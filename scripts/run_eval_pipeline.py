#!/usr/bin/env python3
"""RK1 eval pipeline: base vs SFT on plant + general, checkpoint rule, test, plots.

Resumable: every generated/judged row is appended to --out-dir immediately and
skipped on rerun. Point --out-dir at Google Drive so a dropped runtime loses nothing.

Phases:
  1. baseline samples (configs/eval.yaml baseline_prompts), base + candidates
  2. dev: base, then candidates in order until one passes the checkpoint rule
  3. test: base + selected variant (once, after selection)
  4. summary.json / summary.md / plots
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.io_jsonl import read_jsonl
from src.eval.checkers import check_no_invented_experience
from src.eval.score import aggregate_general, score_answer, transition_counts
from src.train.sft_qlora import load_yaml

PLANT_SUITES = ("plant", "garden")
JUDGE_DIMS = ("factual_correctness", "context_fit", "practicality", "calibrated_uncertainty")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_done(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


# ---------------------------------------------------------------- model


class Runner:
    def __init__(self, model_cfg: dict, adapters: dict[str, Path], batch_size: int, seed: int):
        import torch
        from peft import PeftModel

        from src.train.sft_qlora import load_model_and_tokenizer

        self.torch = torch
        base, tok = load_model_and_tokenizer(model_cfg)
        base.config.use_cache = True
        tok.padding_side = "left"
        names = list(adapters)
        model = PeftModel.from_pretrained(base, str(adapters[names[0]]), adapter_name=names[0])
        for name in names[1:]:
            model.load_adapter(str(adapters[name]), adapter_name=name)
        model.eval()
        self.model, self.tok = model, tok
        self.system = (model_cfg.get("system_prompt") or "").strip()
        self.gen = dict(model_cfg.get("generation") or {})
        self.batch_size = batch_size
        self.seed = seed

    def _generate(self, prompts: list[str], batch_idx: int) -> list[str]:
        torch = self.torch
        texts = []
        for p in prompts:
            msgs = [{"role": "system", "content": self.system}] if self.system else []
            msgs.append({"role": "user", "content": p})
            texts.append(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        torch.manual_seed(self.seed + batch_idx)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=int(self.gen.get("max_new_tokens", 512)),
                do_sample=bool(self.gen.get("do_sample", True)),
                temperature=float(self.gen.get("temperature", 0.2)),
                top_p=float(self.gen.get("top_p", 0.9)),
                pad_token_id=self.tok.pad_token_id,
                use_cache=True,
            )
        n = enc["input_ids"].shape[1]
        return [self.tok.decode(o[n:], skip_special_tokens=True).strip() for o in out]

    def generate(self, variant: str, prompts: list[str], batch_idx: int) -> list[str]:
        if variant == "base":
            with self.model.disable_adapter():
                return self._generate(prompts, batch_idx)
        self.model.set_adapter(variant)
        return self._generate(prompts, batch_idx)


# ---------------------------------------------------------------- suites


def user_text(row: dict) -> str:
    return row["messages"][0]["content"].strip()


def build_suite(name: str, split: str, cfg: dict) -> list[dict[str, Any]]:
    paths = cfg["paths"]
    if name == "baseline":
        return [{"id": p["id"], "prompt": p["content"].strip()} for p in cfg["baseline_prompts"]]
    if name == "general":
        return [
            {"id": r["id"], "prompt": user_text(r), "category": r["category"], "gold": r["gold"], "check": r["check"]}
            for r in read_jsonl(ROOT / paths[f"general_{split}"])
        ]
    if name in PLANT_SUITES:
        rows = read_jsonl(ROOT / paths[f"{name}_{split}"])
        return [{"id": r["id"], "prompt": user_text(r), "reference": r["messages"][1]["content"]} for r in rows]
    raise ValueError(name)


def run_generation(runner: Runner, variant: str, suite: list[dict], path: Path, limit: int | None) -> list[dict]:
    todo_all = suite[:limit] if limit else suite
    done = read_done(path)
    done_ids = {r["id"] for r in done}
    todo = [r for r in todo_all if r["id"] not in done_ids]
    if not todo:
        log(f"  {path.name}: already complete ({len(done)})")
        return [r for r in done if r["id"] in {x["id"] for x in todo_all}]
    log(f"  {path.name}: {len(todo)} to generate ({len(done)} done)")
    bs = runner.batch_size
    for i in range(0, len(todo), bs):
        chunk = todo[i : i + bs]
        t0 = time.time()
        preds = runner.generate(variant, [r["prompt"] for r in chunk], batch_idx=len(done) + i)
        append_jsonl(path, [{**r, "variant": variant, "prediction": p} for r, p in zip(chunk, preds)])
        log(f"    {min(i + bs, len(todo))}/{len(todo)} ({time.time() - t0:.0f}s/batch)")
    keep = {x["id"] for x in todo_all}
    return [r for r in read_done(path) if r["id"] in keep]


def run_judging(judge, rows: list[dict], path: Path) -> list[dict]:
    done = {r["id"]: r for r in read_done(path) if r.get("judge", {}).get("status") == "ok"}
    for r in rows:
        if r["id"] in done:
            continue
        res = None
        for attempt in range(4):
            res = judge.judge(r["prompt"], r["prediction"], r.get("reference"))
            if res.get("status") == "ok":
                break
            log(f"    judge retry {attempt + 1} {r['id']}: {res.get('error', '')[:120]}")
            time.sleep(5 * (attempt + 1))
        if res.get("status") == "ok":
            append_jsonl(path, [{**r, "judge": res}])
            done[r["id"]] = {**r, "judge": res}
    log(f"  {path.name}: judged {len(done)}/{len(rows)}")
    return [done[r["id"]] for r in rows if r["id"] in done]


# ---------------------------------------------------------------- metrics


def general_metrics(rows: list[dict], base_rows: list[dict] | None) -> dict:
    scored = [{**r, "correct": score_answer(r["prediction"], r["gold"], r["check"])} for r in rows]
    m = aggregate_general(scored)
    if base_rows is not None:
        b = [{**r, "correct": score_answer(r["prediction"], r["gold"], r["check"])} for r in base_rows]
        m["transitions_vs_base"] = transition_counts(b, scored)
    return m


def autocheck_metrics(rows: list[dict]) -> dict:
    flags = [check_no_invented_experience(r["prediction"]).passed for r in rows]
    return {"n": len(flags), "pass_rate": sum(flags) / max(1, len(flags))}


def judge_metrics(rows: list[dict], n_expected: int) -> dict:
    if not rows:
        return {"n_ok": 0, "n_expected": n_expected}
    scores = [r["judge"]["scores"] for r in rows]
    return {
        "n_ok": len(rows),
        "n_expected": n_expected,
        "mean_total": sum(r["judge"]["total"] for r in rows) / len(rows),
        "per_dim": {d: sum(s[d] for s in scores) / len(scores) for d in JUDGE_DIMS},
        "critical_error_rate": sum(bool(s["critical_error"]) for s in scores) / len(scores),
    }


def evaluate_split(
    runner: Runner, judge, variant: str, split: str, cfg: dict, out: Path, limit: int | None, results: dict
) -> dict:
    log(f"== {split} / {variant}")
    res: dict[str, Any] = {}
    gen = run_generation(runner, variant, build_suite("general", split, cfg), out / f"general_{split}__{variant}.jsonl", limit)
    base_gen = None if variant == "base" else read_done(out / f"general_{split}__base.jsonl")
    res["general"] = general_metrics(gen, base_gen)
    answers, judged_all = [], []
    for suite in PLANT_SUITES:
        rows = run_generation(runner, variant, build_suite(suite, split, cfg), out / f"{suite}_{split}__{variant}.jsonl", limit)
        answers.extend(rows)
        res[f"autocheck_{suite}"] = autocheck_metrics(rows)
        if judge is not None:
            judged = run_judging(judge, rows, out / f"judge_{suite}_{split}__{variant}.jsonl")
            judged_all.extend(judged)
            res[f"judge_{suite}"] = judge_metrics(judged, len(rows))
    res["autocheck"] = autocheck_metrics(answers)
    if judge is not None:
        res["judge"] = judge_metrics(judged_all, len(answers))
    results.setdefault(split, {})[variant] = res
    log(f"   general acc={res['general']['accuracy']:.3f} no_invented_exp={res['autocheck']['pass_rate']:.2f} "
        f"judge={res.get('judge', {}).get('mean_total', 'n/a')}")
    return res


def passes_rule(base: dict, cand: dict, max_drop_pp: float) -> tuple[bool, dict]:
    drop_pp = (base["general"]["accuracy"] - cand["general"]["accuracy"]) * 100
    general_ok = drop_pp <= max_drop_pp
    if "judge" in base and "mean_total" in base["judge"] and "mean_total" in cand.get("judge", {}):
        plant_ok = cand["judge"]["mean_total"] >= base["judge"]["mean_total"]
        plant_basis = "judge_mean_total_plant+garden"
    else:
        plant_ok = cand["autocheck"]["pass_rate"] >= base["autocheck"]["pass_rate"]
        plant_basis = "no_invented_experience_pass_rate"
    return general_ok and plant_ok, {
        "general_drop_pp": drop_pp,
        "general_ok": general_ok,
        "plant_ok": plant_ok,
        "plant_basis": plant_basis,
    }


# ---------------------------------------------------------------- report


def write_summary(out: Path, summary: dict) -> None:
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [f"# RK1 eval summary\n", f"Selected: **{summary.get('selected', 'pending')}**\n"]
    for split in ("dev", "test"):
        block = summary["results"].get(split)
        if not block:
            continue
        lines.append(f"\n## {split}\n")
        lines.append(
            "| variant | general acc | no invented exp | judge plant | judge garden | judge all (0-8) "
            "| critical err | c→w | w→c |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        nan = float("nan")
        for v, r in block.items():
            j = r.get("judge", {})
            t = r["general"].get("transitions_vs_base", {})
            lines.append(
                f"| {v} | {r['general']['accuracy']:.3f} | {r['autocheck']['pass_rate']:.2f} | "
                f"{r.get('judge_plant', {}).get('mean_total', nan):.2f} | "
                f"{r.get('judge_garden', {}).get('mean_total', nan):.2f} | "
                f"{j.get('mean_total', nan):.2f} | {j.get('critical_error_rate', nan):.2f} | "
                f"{t.get('correct_to_wrong', '-')} | {t.get('wrong_to_correct', '-')} |"
            )
        lines.append("\nPer-category general accuracy:\n")
        cats = sorted({c for r in block.values() for c in r["general"]["per_category"]})
        lines.append("| variant | " + " | ".join(cats) + " |")
        lines.append("|---|" + "---:|" * len(cats))
        for v, r in block.items():
            pc = r["general"]["per_category"]
            lines.append(f"| {v} | " + " | ".join(f"{pc.get(c, 0):.2f}" for c in cats) + " |")
    for d in summary.get("decisions", []):
        lines.append(f"\n- rule check `{d['variant']}`: pass={d['pass']} {json.dumps(d['detail'])}")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(out: Path, summary: dict, adapter_root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = out / "plots"
    plots.mkdir(exist_ok=True)

    states = sorted(adapter_root.glob("checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name.split("-")[1]))
    if states:
        hist = json.loads(states[-1].read_text())["log_history"]
        tr = [(h["step"], h["loss"]) for h in hist if "loss" in h]
        ev = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(*zip(*tr), marker="o", label="train loss")
        if ev:
            ax.plot(*zip(*ev), marker="s", label="eval loss (plant_dev)")
        ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title(f"QLoRA training ({adapter_root.name})"); ax.legend()
        fig.tight_layout(); fig.savefig(plots / "train_loss.png", dpi=120); plt.close(fig)

    for split, block in summary["results"].items():
        variants = list(block)
        cats = sorted({c for r in block.values() for c in r["general"]["per_category"]})
        fig, ax = plt.subplots(figsize=(8, 4))
        w = 0.8 / len(variants)
        for i, v in enumerate(variants):
            pc = block[v]["general"]["per_category"]
            ax.bar([x + i * w for x in range(len(cats))], [pc.get(c, 0) for c in cats], w, label=v)
        ax.set_xticks([x + w * (len(variants) - 1) / 2 for x in range(len(cats))], cats, rotation=20)
        ax.set_ylim(0, 1); ax.set_ylabel("accuracy"); ax.set_title(f"general_{split} per category"); ax.legend()
        fig.tight_layout(); fig.savefig(plots / f"general_{split}_per_category.png", dpi=120); plt.close(fig)

        jv = [v for v in variants if "per_dim" in block[v].get("judge", {})]
        if jv:
            fig, ax = plt.subplots(figsize=(8, 4))
            w = 0.8 / len(jv)
            for i, v in enumerate(jv):
                pd = block[v]["judge"]["per_dim"]
                ax.bar([x + i * w for x in range(len(JUDGE_DIMS))], [pd[d] for d in JUDGE_DIMS], w, label=v)
            ax.set_xticks([x + w * (len(jv) - 1) / 2 for x in range(len(JUDGE_DIMS))], JUDGE_DIMS, rotation=15)
            ax.set_ylim(0, 2); ax.set_ylabel("mean score (0-2)"); ax.set_title(f"judge plant+garden {split}"); ax.legend()
            fig.tight_layout(); fig.savefig(plots / f"judge_plant_{split}.png", dpi=120); plt.close(fig)
    log(f"plots → {plots}")


# ---------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--adapter-root", required=True, help="SFT output dir (final adapter + checkpoint-*)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="Rows per suite (smoke only)")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--skip-test", action="store_true")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    adapter_root = Path(args.adapter_root)
    model_cfg = load_yaml(ROOT / "configs/model.yaml")
    eval_cfg = load_yaml(ROOT / "configs/eval.yaml")
    max_drop = float(eval_cfg["checkpoint_rule"]["max_general_dev_drop_pp"])

    judge = None
    if not args.no_judge:
        from src.eval.openrouter_judge import OpenRouterJudge

        jcfg = load_yaml(ROOT / "configs/judge.yaml")
        jcfg["cache_dir"] = str(out / "judge_cache")
        judge = OpenRouterJudge(jcfg)  # fails fast if OPENROUTER_API_KEY is missing

    # Candidates in order of preference: full SFT, then the softer earlier checkpoint.
    adapters: dict[str, Path] = {"sft_final": adapter_root}
    ckpts = sorted(adapter_root.glob("checkpoint-*"), key=lambda q: int(q.name.split("-")[1]))
    for c in ckpts[:-1]:
        adapters[f"sft_step{c.name.split('-')[1]}"] = c
    candidates = list(adapters)
    log(f"candidates: {candidates}")

    runner = Runner(model_cfg, adapters, args.batch_size, int(eval_cfg.get("seed", 42)))
    summary: dict[str, Any] = {
        "model_id": model_cfg["model_id"],
        "generation": model_cfg.get("generation"),
        "judge_model": None if judge is None else judge.cfg["model"],
        "adapter_root": str(adapter_root),
        "candidates": candidates,
        "limit": args.limit,
        "results": {},
        "decisions": [],
    }

    log("== baseline samples")
    bl = build_suite("baseline", "", eval_cfg)
    for v in ["base", candidates[0]]:
        run_generation(runner, v, bl, out / f"baseline_samples__{v}.jsonl", None)

    base_dev = evaluate_split(runner, judge, "base", "dev", eval_cfg, out, args.limit, summary["results"])
    write_summary(out, summary)
    selected = "base"
    for cand in candidates:
        res = evaluate_split(runner, judge, cand, "dev", eval_cfg, out, args.limit, summary["results"])
        ok, detail = passes_rule(base_dev, res, max_drop)
        summary["decisions"].append({"variant": cand, "pass": ok, "detail": detail})
        write_summary(out, summary)
        log(f"rule {cand}: pass={ok} {detail}")
        if ok:
            selected = cand
            break
    summary["selected"] = selected
    write_summary(out, summary)

    if not args.skip_test:
        test_variants = ["base"] if selected == "base" else ["base", selected]
        # Always report the full SFT on test too if nothing passed, so the report has a base-vs-SFT test table.
        if selected == "base":
            test_variants.append(candidates[0])
        for v in test_variants:
            evaluate_split(runner, judge, v, "test", eval_cfg, out, args.limit, summary["results"])
            write_summary(out, summary)

    make_plots(out, summary, adapter_root)
    write_summary(out, summary)
    log(f"DONE selected={selected} → {out}/summary.md")
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
