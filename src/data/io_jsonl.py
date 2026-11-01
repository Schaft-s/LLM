"""JSONL helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def strip_for_training(row: dict[str, Any]) -> dict[str, Any]:
    """Keep id/group_id/messages; metadata stays in sidecar."""
    return {
        "id": row["id"],
        "group_id": row["group_id"],
        "messages": row["messages"],
    }
