#!/usr/bin/env python3
"""Judge the SFT targets themselves: SE reference answers vs base vs SFT, all without a reference.

Tests whether the training data (best Stack Exchange answers) scores like the SFT model.
Uses plant_dev only; writes judge_reference_check_dev.jsonl + prints means.

  python scripts/judge_references.py --eval-dir results/eval_s1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.io_jsonl import read_jsonl, write_jsonl
from src.eval.openrouter_judge import OpenRouterJudge
from src.train.sft_qlora import load_yaml

DIMS = ("factual_correctness", "context_fit", "practicality", "calibrated_uncertainty")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", required=True)
    args = p.parse_args()
    eval_dir = Path(args.eval_dir)

    load_dotenv(ROOT / ".env")
    cfg = load_yaml(ROOT / "configs/judge.yaml")
    cfg["cache_dir"] = str(eval_dir / "judge_cache")
    judge = OpenRouterJudge(cfg)

    base = {r["id"]: r for r in read_jsonl(eval_dir / "plant_dev__base.jsonl")}
    sft = {r["id"]: r for r in read_jsonl(eval_dir / "plant_dev__s1_final.jsonl")}
    sources = {
        "se_reference": {i: r["reference"] for i, r in base.items()},
        "base": {i: r["prediction"] for i, r in base.items()},
        "s1_final": {i: sft[i]["prediction"] for i in base},
    }
    out, totals = [], {}
    for name, answers in sources.items():
        for i, ans in answers.items():
            res = judge.judge(base[i]["prompt"], ans, None)
            if res.get("status") != "ok":
                print("error", name, i, res.get("error"))
                continue
            out.append({"id": i, "source": name, "answer": ans, "judge": res})
            totals.setdefault(name, []).append(res)
    write_jsonl(eval_dir / "judge_reference_check_dev.jsonl", out)
    for name, rs in totals.items():
        dims = {d: statistics.mean(r["scores"][d] for r in rs) for d in DIMS}
        crit = statistics.mean(bool(r["scores"]["critical_error"]) for r in rs)
        print(
            f"{name:13} n={len(rs)} mean_total={statistics.mean(r['total'] for r in rs):.2f} "
            f"crit={crit:.2f} " + " ".join(f"{d[:6]}={v:.2f}" for d, v in dims.items())
        )


if __name__ == "__main__":
    main()
