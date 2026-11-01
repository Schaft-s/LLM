#!/usr/bin/env python3
"""Remove spot-review denylist IDs from plant splits (group-safe)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.io_jsonl import read_jsonl, write_jsonl

DENY_IDS = {
    "gardening-q20502-a20503",
    "gardening-q17501-a17675",
    "gardening-q22777-a22778",
    "gardening-q53109-a53110",
}


def main() -> None:
    deny_groups = set()
    for eid in DENY_IDS:
        qpart = eid.split("-q", 1)[1]
        qid = qpart.split("-a", 1)[0]
        deny_groups.add(f"gardening-q{qid}")

    report = {"removed": {}}
    for name in ("plant_train", "plant_dev", "plant_test"):
        path = ROOT / f"data/processed/{name}.jsonl"
        rows = read_jsonl(path)
        kept = [r for r in rows if r["group_id"] not in deny_groups]
        removed = len(rows) - len(kept)
        write_jsonl(path, kept)
        report["removed"][name] = removed
        print(f"{name}: removed {removed}, left {len(kept)}")

    out = ROOT / "reports/spot_denylist_applied.json"
    out.write_text(json.dumps({"deny_ids": sorted(DENY_IDS), **report}, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
