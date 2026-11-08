#!/usr/bin/env python3
"""Judge SFT v2 candidates with the selection judge (configs/sft_v2.yaml).

  python scripts/judge_sft_pool.py --calibrate          # dev: SE refs / base / S1, compare with eval judge
  python scripts/judge_sft_pool.py --limit 50           # pilot on the pool
  python scripts/judge_sft_pool.py                      # full pool (resumable via cache)
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.io_jsonl import read_jsonl, write_jsonl
from src.data.target_judge import TargetJudge
from src.train.sft_qlora import load_yaml


def cost_of(res: dict, jcfg: dict) -> float:
    u = res.get("usage") or {}
    if u.get("cost") is not None:
        return float(u["cost"])
    return (
        u.get("prompt_tokens", 0) * jcfg["price_per_m_input"]
        + u.get("completion_tokens", 0) * jcfg["price_per_m_output"]
    ) / 1e6


def run_items(judge: TargetJudge, items: list[tuple[str, str, str]], jcfg: dict) -> dict[str, dict]:
    """items: (key, question, answer). Returns key -> judge result (ok only)."""
    out: dict[str, dict] = {}
    spent = 0.0
    lock = threading.Lock()
    stop = threading.Event()

    def work(item):
        key, q, a = item
        if stop.is_set():
            return key, None
        for attempt in range(4):
            res = judge.judge(q, a)
            if res.get("status") == "ok":
                return key, res
            time.sleep(2 * (attempt + 1))
        print(f"FAILED {key}: {res.get('error')}", flush=True)
        return key, None

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=jcfg["workers"]) as ex:
        futs = [ex.submit(work, it) for it in items]
        for i, fut in enumerate(as_completed(futs), 1):
            key, res = fut.result()
            if res is None:
                continue
            with lock:
                out[key] = res
                spent += cost_of(res, jcfg)
                if spent > jcfg["budget_usd"] and not stop.is_set():
                    print(f"BUDGET STOP at ${spent:.3f}", flush=True)
                    stop.set()
            if i % 200 == 0 or i == len(items):
                print(f"{i}/{len(items)} ok={len(out)} spent=${spent:.3f} {time.time() - t0:.0f}s", flush=True)
    print(f"total spent (incl. cached) ${spent:.3f}")
    return out


def summarize(name: str, rs: list[dict]) -> None:
    s = [r["scores"] for r in rs]
    print(
        f"{name:13} n={len(rs)} total={statistics.mean(r['total'] for r in rs):.2f} "
        f"crit={statistics.mean(x['critical_error'] for x in s):.2f} "
        f"style={statistics.mean(x['assistant_style'] for x in s):.2f} "
        f"fact={statistics.mean(x['factual_correctness'] for x in s):.2f}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    load_dotenv(ROOT / ".env")
    cfg = load_yaml(ROOT / "configs/sft_v2.yaml")
    jcfg = dict(cfg["judge"])
    jcfg["cache_dir"] = str(ROOT / jcfg["cache_dir"])
    judge = TargetJudge(jcfg)
    paths = cfg["paths"]

    if args.calibrate:
        ev = ROOT / "results/eval_s1"
        base = {r["id"]: r for r in read_jsonl(ev / "plant_dev__base.jsonl")}
        sft = {r["id"]: r for r in read_jsonl(ev / "plant_dev__s1_final.jsonl")}
        items = []
        for i, r in base.items():
            items.append((f"se_reference|{i}", r["prompt"], r["reference"]))
            items.append((f"base|{i}", r["prompt"], r["prediction"]))
            items.append((f"s1_final|{i}", r["prompt"], sft[i]["prediction"]))
        res = run_items(judge, items, jcfg)
        rows = [{"key": k, "source": k.split("|")[0], "id": k.split("|")[1], "judge": v} for k, v in res.items()]
        write_jsonl(ROOT / paths["calibration"], rows)
        for src in ("se_reference", "base", "s1_final"):
            summarize(src, [r["judge"] for r in rows if r["source"] == src])
        return

    pool = [r for r in read_jsonl(ROOT / paths["pool"]) if not r["metadata"]["prefilter_reasons"]]
    if args.limit:
        random.Random(0).shuffle(pool)
        pool = pool[: args.limit]
    items = [(r["id"], r["judge_question"], r["messages"][1]["content"]) for r in pool]
    res = run_items(judge, items, jcfg)
    rows = [
        {
            "id": k,
            "model": v.get("model"),
            "rubric_version": jcfg["rubric_version"],
            "scores": v["scores"],
            "total": v["total"],
            "usage": v.get("usage"),
        }
        for k, v in sorted(res.items())
    ]
    out = ROOT / paths["judgments"]
    if args.limit:
        out = out.with_name(f"judgments_pilot{args.limit}.jsonl")
    write_jsonl(out, rows)
    print(f"Wrote {out} ({len(rows)})")
    summarize("pool", list(res.values()))


if __name__ == "__main__":
    main()
