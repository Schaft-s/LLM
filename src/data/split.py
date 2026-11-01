"""Group-aware train/dev/test split by question_id."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


def group_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[rec["group_id"]].append(rec)
    return dict(groups)


def split_groups(
    records: list[dict[str, Any]],
    *,
    seed: int,
    dev_size: int,
    test_size: int,
) -> dict[str, list[dict[str, Any]]]:
    groups = group_records(records)
    group_ids = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    if len(group_ids) < dev_size + test_size + 1:
        raise ValueError(
            f"Not enough groups ({len(group_ids)}) for "
            f"dev={dev_size}, test={test_size}, and train."
        )

    test_ids = set(group_ids[:test_size])
    dev_ids = set(group_ids[test_size : test_size + dev_size])
    train_ids = set(group_ids[test_size + dev_size :])

    def flatten(ids: set[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for gid in sorted(ids):
            rows.extend(groups[gid])
        return rows

    return {
        "plant_train": flatten(train_ids),
        "plant_dev": flatten(dev_ids),
        "plant_test": flatten(test_ids),
    }


def assert_no_group_leakage(splits: dict[str, list[dict[str, Any]]]) -> None:
    sets = {name: {r["group_id"] for r in rows} for name, rows in splits.items()}
    overlap_td = sets["plant_train"] & sets["plant_dev"]
    overlap_tt = sets["plant_train"] & sets["plant_test"]
    overlap_dt = sets["plant_dev"] & sets["plant_test"]
    if overlap_td or overlap_tt or overlap_dt:
        raise AssertionError(
            f"Group leakage: train∩dev={len(overlap_td)}, "
            f"train∩test={len(overlap_tt)}, dev∩test={len(overlap_dt)}"
        )
