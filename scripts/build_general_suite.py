#!/usr/bin/env python3
"""Build general_dev.jsonl and general_test.jsonl."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.io_jsonl import write_jsonl
from src.eval.general_suite import build_general_splits


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs/eval.yaml").read_text(encoding="utf-8"))
    splits = build_general_splits(cfg)
    meta = splits.pop("_meta")
    for name, rows in splits.items():
        path = ROOT / cfg["paths"][name]
        # Training-facing fields + gold for scorers
        write_jsonl(path, rows)
        print(f"Wrote {path} ({len(rows)})")
    meta_path = ROOT / "data/manifests/general_suite_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(meta)


if __name__ == "__main__":
    main()
