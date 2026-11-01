#!/usr/bin/env python3
"""Score generation JSONL against general gold / format checker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.io_jsonl import read_jsonl
from src.eval.checkers import check_three_numbered_steps
from src.eval.score import aggregate_general, score_answer, transition_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="JSONL with id, prediction, and gold/check for general")
    parser.add_argument("--baseline", default=None, help="Optional baseline predictions for transitions")
    parser.add_argument("--mode", choices=["general", "format"], default="general")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.predictions)
    if args.mode == "format":
        scored = []
        for r in rows:
            pred = r.get("prediction") or r.get("response") or ""
            cr = check_three_numbered_steps(pred)
            scored.append({**r, "format_pass": cr.passed, "format_detail": cr.detail})
        summary = {
            "n": len(scored),
            "pass_rate": sum(1 for r in scored if r["format_pass"]) / max(1, len(scored)),
        }
    else:
        scored = []
        for r in rows:
            pred = r.get("prediction") or r.get("response") or ""
            ok = score_answer(pred, r["gold"], r["check"])
            scored.append({**r, "correct": ok})
        summary = aggregate_general(scored)
        if args.baseline:
            base = read_jsonl(args.baseline)
            # ensure correct flags
            base_scored = []
            for r in base:
                pred = r.get("prediction") or r.get("response") or ""
                base_scored.append({**r, "correct": score_answer(pred, r["gold"], r["check"])})
            summary["transitions"] = transition_counts(base_scored, scored)

    out = Path(args.out) if args.out else ROOT / "reports/metrics" / f"score_{args.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "n_rows": len(scored)}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
