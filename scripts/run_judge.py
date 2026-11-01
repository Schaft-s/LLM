#!/usr/bin/env python3
"""Run OpenRouter judge on plant answer JSONL (question/reference/response fields)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.io_jsonl import read_jsonl, write_jsonl
from src.eval.openrouter_judge import OpenRouterJudge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL with question, response, optional reference")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    cfg = yaml.safe_load((ROOT / "configs/judge.yaml").read_text(encoding="utf-8"))
    # Make cache path absolute
    cfg["cache_dir"] = str(ROOT / cfg["cache_dir"])
    judge = OpenRouterJudge(cfg)
    rows = read_jsonl(ROOT / args.input if not Path(args.input).is_absolute() else args.input)
    if args.limit:
        rows = rows[: args.limit]
    out_rows = []
    for row in rows:
        q = row.get("question") or row.get("prompt")
        if not q and row.get("messages"):
            q = row["messages"][0]["content"]
        a = row.get("response") or row.get("prediction") or row.get("answer")
        if a is None and row.get("messages") and len(row["messages"]) > 1:
            a = row["messages"][1]["content"]
        ref = row.get("reference")
        result = judge.judge(q, a or "", ref, dry_run=args.dry_run)
        out_rows.append({**row, "judge": result})
        print(row.get("id"), result.get("status"), result.get("total"))

    out = args.output or str(ROOT / "reports/metrics/judge_results.jsonl")
    write_jsonl(out, out_rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
