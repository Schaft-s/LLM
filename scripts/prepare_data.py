#!/usr/bin/env python3
"""Build plant_train/dev/test JSONL from Gardening SE dump."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.filters import build_candidates
from src.data.io_jsonl import strip_for_training, write_jsonl
from src.data.parse_se import load_questions_and_answers
from src.data.split import assert_no_group_leakage, split_groups


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_spot_review(kept: list[dict], dropped: list[dict], seed: int, n: int = 25) -> list[dict]:
    """Diversify by matched tag / drop reason for a 25-example spot review."""
    rng = random.Random(seed)
    sample: list[dict] = []
    by_tag: dict[str, list[dict]] = {}
    for row in kept:
        tag = (row["metadata"].get("matched_tags") or ["untagged"])[0]
        by_tag.setdefault(tag, []).append(row)
    tags = sorted(by_tag)
    rng.shuffle(tags)
    for tag in tags:
        if len(sample) >= 15:
            break
        pick = rng.choice(by_tag[tag])
        if pick["id"] not in {s["id"] for s in sample}:
            sample.append({**pick, "_spot_bucket": f"keep:{tag}"})

    by_reason: dict[str, list[dict]] = {}
    for row in dropped:
        reason = (row["metadata"].get("drop_reasons") or ["other"])[0]
        by_reason.setdefault(reason, []).append(row)
    for reason in sorted(by_reason):
        if len(sample) >= n:
            break
        pick = rng.choice(by_reason[reason])
        if pick["id"] not in {s["id"] for s in sample}:
            sample.append({**pick, "_spot_bucket": f"drop:{reason}"})

    while len(sample) < n and kept:
        pick = rng.choice(kept)
        if pick["id"] not in {s["id"] for s in sample}:
            sample.append({**pick, "_spot_bucket": "keep:extra"})
        else:
            break
    return sample[:n]


def annotate_spot(sample: list[dict]) -> list[dict]:
    """Attach provisional keep/drop labels for human confirmation."""
    out = []
    for row in sample:
        meta = row["metadata"]
        provisional = "keep" if meta.get("review_status") == "auto_accepted" else "drop"
        out.append(
            {
                "id": row["id"],
                "group_id": row["group_id"],
                "bucket": row.get("_spot_bucket"),
                "provisional": provisional,
                "decision": provisional,  # editable during review
                "reason": ",".join(meta.get("drop_reasons") or ["auto_ok"]),
                "tags": meta.get("tags"),
                "matched_tags": meta.get("matched_tags"),
                "user": row["messages"][0]["content"][:800],
                "assistant": row["messages"][1]["content"][:800],
                "urls": meta.get("source_urls"),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/data.yaml"))
    args = parser.parse_args()
    cfg = load_config(Path(args.config))

    posts_xml = ROOT / cfg["source"]["posts_xml"]
    if not posts_xml.exists():
        raise SystemExit(f"Missing {posts_xml}; extract the SE dump first.")

    print(f"Loading {posts_xml} ...")
    questions, answers = load_questions_and_answers(posts_xml)
    print(f"Questions={len(questions)} answer_parents={len(answers)}")

    tags = set(cfg["expanded_tags"])
    fcfg = cfg["filters"]
    candidates = build_candidates(
        questions,
        answers,
        tags,
        drop_img=fcfg["drop_img_markup"],
        drop_closed=fcfg["drop_closed"],
        min_question_chars=fcfg["min_question_chars"],
        min_answer_chars=fcfg["min_answer_chars"],
        apply_sufficiency=fcfg["drop_image_dependent_heuristics"],
    )

    # Stage counts matching audit path before sufficiency heuristics
    technical_kept = [
        c
        for c in candidates
        if not ({"closed", "img_markup"} & set(c.drop_reasons))
        and c.answer.score > 0
    ]
    kept_cands = [c for c in candidates if c.kept]
    dropped_cands = [c for c in candidates if not c.kept]

    kept_rows = [c.to_record() for c in kept_cands]
    dropped_rows = [c.to_record() for c in dropped_cands]

    reason_counts = Counter()
    for c in dropped_cands:
        for r in c.drop_reasons:
            reason_counts[r] += 1

    report = {
        "questions_total": len(questions),
        "expanded_tag_candidates_with_positive_answer": len(candidates),
        "technical_filter_kept_like_audit": len(technical_kept),
        "after_sufficiency_kept": len(kept_rows),
        "dropped": len(dropped_rows),
        "drop_reason_counts": dict(reason_counts),
        "audit_expected_technical": 838,
    }
    print(json.dumps(report, indent=2))

    paths = cfg["paths"]
    write_jsonl(ROOT / paths["candidates"], kept_rows)

    spot = annotate_spot(select_spot_review(kept_rows, dropped_rows, cfg["split"]["seed"], 25))
    write_jsonl(ROOT / paths["spot_review"], spot)

    scfg = cfg["split"]
    # Cap split sizes if corpus is smaller than planned
    n_groups = len({r["group_id"] for r in kept_rows})
    dev_size = min(scfg["plant_dev_size"], max(1, n_groups // 10))
    test_size = min(scfg["plant_test_size"], max(1, n_groups // 10))
    while n_groups < dev_size + test_size + 50 and (dev_size > 20 or test_size > 20):
        # keep at least ~50 train groups when possible
        if dev_size >= test_size and dev_size > 20:
            dev_size -= 1
        elif test_size > 20:
            test_size -= 1
        else:
            break

    splits = split_groups(
        kept_rows,
        seed=scfg["seed"],
        dev_size=dev_size,
        test_size=test_size,
    )
    assert_no_group_leakage(splits)

    for name, rows in splits.items():
        out = ROOT / paths[name]
        write_jsonl(out, [strip_for_training(r) for r in rows])
        print(f"Wrote {out} ({len(rows)} rows, {len({r['group_id'] for r in rows})} groups)")

    sidecar = []
    for name, rows in splits.items():
        for r in rows:
            meta = dict(r["metadata"])
            meta["split"] = name
            sidecar.append({"id": r["id"], "group_id": r["group_id"], "metadata": meta})
    write_jsonl(ROOT / paths["metadata_sidecar"], sidecar)

    report["split_sizes"] = {
        name: {
            "rows": len(rows),
            "groups": len({r["group_id"] for r in rows}),
        }
        for name, rows in splits.items()
    }
    report["spot_review_path"] = paths["spot_review"]
    report_path = ROOT / paths["filter_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
