"""Score general-task answers against gold."""

from __future__ import annotations

import json
import re
from typing import Any


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def extract_json_obj(text: str) -> dict | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None


def score_answer(pred: str, gold: str, check: str) -> bool:
    pred = pred or ""
    gold = gold or ""
    if check == "exact":
        # Allow gold to appear as the whole answer or sole token
        p = _normalize_ws(pred)
        g = _normalize_ws(gold)
        if p == g:
            return True
        # first line / last integer-like token fallback for verbose models
        first = _normalize_ws(pred.splitlines()[0]) if pred.strip() else ""
        return first == g
    if check == "exact_ci":
        return _normalize_ws(pred).lower() == _normalize_ws(gold).lower() or (
            _normalize_ws(pred.splitlines()[0]).lower() == _normalize_ws(gold).lower()
            if pred.strip()
            else False
        )
    if check == "choice_letter":
        g = gold.strip().upper()
        # Prefer explicit lone letter
        m = re.search(r"\b([A-E])\b", pred.upper())
        if not m:
            return False
        # If multiple letters, take the first standalone after Answer: if present
        m2 = re.search(r"(?:answer\s*[:=]\s*)([A-E])\b", pred.upper())
        letter = m2.group(1) if m2 else m.group(1)
        return letter == g
    if check == "json_obj":
        pred_obj = extract_json_obj(pred)
        try:
            gold_obj = json.loads(gold)
        except json.JSONDecodeError:
            return False
        return pred_obj == gold_obj
    raise ValueError(f"unknown check: {check}")


def looks_like_plant_refusal(text: str) -> bool:
    t = (text or "").lower()
    cues = [
        "only answer questions about plants",
        "i only help with plants",
        "as a plant",
        "outside my plant",
        "not a plant-related",
    ]
    return any(c in t for c in cues)


def aggregate_general(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cat: dict[str, list[bool]] = {}
    refusals = 0
    for r in rows:
        by_cat.setdefault(r["category"], []).append(bool(r.get("correct")))
        if looks_like_plant_refusal(r.get("prediction", "")):
            refusals += 1
    per_cat = {k: (sum(v) / len(v) if v else 0.0) for k, v in sorted(by_cat.items())}
    all_flags = [c for vs in by_cat.values() for c in vs]
    return {
        "n": len(all_flags),
        "accuracy": (sum(all_flags) / len(all_flags) if all_flags else 0.0),
        "per_category": per_cat,
        "plant_topic_refusals": refusals,
    }


def transition_counts(base_rows: list[dict], sft_rows: list[dict]) -> dict[str, int]:
    b = {r["id"]: bool(r.get("correct")) for r in base_rows}
    s = {r["id"]: bool(r.get("correct")) for r in sft_rows}
    ids = set(b) & set(s)
    c2w = w2c = both_c = both_w = 0
    for i in ids:
        if b[i] and not s[i]:
            c2w += 1
        elif not b[i] and s[i]:
            w2c += 1
        elif b[i] and s[i]:
            both_c += 1
        else:
            both_w += 1
    return {
        "correct_to_wrong": c2w,
        "wrong_to_correct": w2c,
        "both_correct": both_c,
        "both_wrong": both_w,
    }
