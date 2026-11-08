#!/usr/bin/env python3
"""Select SFT v2 targets from judged pool and report corpus statistics.

  python scripts/select_sft_data.py
"""

from __future__ import annotations

import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.io_jsonl import read_jsonl, strip_for_training, write_jsonl
from src.eval.checkers import check_no_invented_experience
from src.train.sft_qlora import load_yaml

NUMBERED_RE = re.compile(r"(?m)^\s*(?:1[.)])\s+\S[\s\S]*?^\s*2[.)]\s+\S")
BULLET_RE = re.compile(r"(?m)^\s*[-*]\s+\S")
# judge said "text sufficient" but hedged about the removed image in its rationale
IMAGE_HEDGE_RE = re.compile(
    r"\brel(?:y|ies|ying) on (?:the )?(?:removed )?(?:images?|photos?)|mostly sufficient|"
    r"image-dependent|depends? (?:partly )?on (?:the )?(?:removed )?(?:images?|photos?)|without the (?:image|photo)",
    re.I,
)
FIRST_PERSON_RE = re.compile(
    r"\b(I think|I would|I'd|I have|I've|in my experience|I'm no expert|I use|I grow|my own)\b"
)


def style_stats(rows: list[dict]) -> dict:
    ans = [r["messages"][1]["content"] for r in rows]
    n = max(1, len(ans))
    return {
        "n": len(ans),
        "numbered_list": round(sum(bool(NUMBERED_RE.search(a)) for a in ans) / n, 3),
        "bullets": round(sum(bool(BULLET_RE.search(a)) for a in ans) / n, 3),
        "first_person": round(sum(bool(FIRST_PERSON_RE.search(a)) for a in ans) / n, 3),
        "no_invented_experience_pass": round(sum(check_no_invented_experience(a).passed for a in ans) / n, 3),
        "answer_words_median": statistics.median(len(a.split()) for a in ans) if ans else 0,
        "tokens_median": statistics.median(r["metadata"]["n_tokens"] for r in rows) if rows else 0,
    }


def main() -> None:
    cfg = load_yaml(ROOT / "configs/sft_v2.yaml")
    s, paths = cfg["select"], cfg["paths"]
    pool = {r["id"]: r for r in read_jsonl(ROOT / paths["pool"])}
    judged = []
    for j in read_jsonl(ROOT / paths["judgments"]):
        r = pool[j["id"]]
        r["judge"] = j["scores"]
        r["judge_total"] = j["total"]
        judged.append(r)

    def ok(r, step):
        j = r["judge"]
        return {
            "text_sufficient": j["question_text_sufficient"]
            and not (r["metadata"]["question_has_image"] and IMAGE_HEDGE_RE.search(j["rationale"])),
            "self_contained": j["answer_self_contained"],
            "on_topic": j["domain"] != "off_topic",
            "no_critical": not j["critical_error"],
            "factual": j["factual_correctness"] >= s["min_factual"],
            "total": r["judge_total"] >= s["min_total"],
            "style": j["assistant_style"] >= s["min_style"],
            "length": r["metadata"]["n_tokens"] <= s["max_train_tokens"],
        }[step]

    steps = ["text_sufficient", "self_contained", "on_topic", "no_critical", "factual", "total", "style", "length"]
    funnel = {"judged": len(judged)}
    funnel_img = {"judged": sum(r["metadata"]["question_has_image"] for r in judged)}
    cur = judged
    for st in steps:
        cur = [r for r in cur if ok(r, st)]
        funnel[st] = len(cur)
        funnel_img[st] = sum(r["metadata"]["question_has_image"] for r in cur)

    by_q: dict[str, list[dict]] = {}
    for r in cur:
        by_q.setdefault(r["group_id"], []).append(r)
    for rs in by_q.values():
        rs.sort(key=lambda r: (-r["judge_total"], -r["judge"]["assistant_style"], -r["metadata"]["answer_score"]))

    # garden eval holdout: one row per question, best selected answer as the judge reference
    garden_groups = sorted(g for g, rs in by_q.items() if rs[0]["judge"]["domain"] != "indoor_container")
    random.Random(s["seed"]).shuffle(garden_groups)
    n_dev, n_test = s["garden_dev_size"], s["garden_test_size"]
    holdout = {"garden_dev": garden_groups[:n_dev], "garden_test": garden_groups[n_dev : n_dev + n_test]}
    for name, groups in holdout.items():
        write_jsonl(ROOT / paths[name], [strip_for_training(by_q[g][0]) for g in sorted(groups)])
    held = {g for gs in holdout.values() for g in gs}

    selected = []
    for g, rs in by_q.items():
        if g not in held:
            selected.extend(rs[: s["max_answers_per_question"]])
    selected.sort(key=lambda r: r["id"])
    funnel["per_question_cap_minus_garden_holdout"] = len(selected)
    funnel_img["per_question_cap_minus_garden_holdout"] = sum(r["metadata"]["question_has_image"] for r in selected)

    write_jsonl(ROOT / paths["train"], [strip_for_training(r) for r in selected])
    sidecar = ROOT / paths["train"].replace(".jsonl", "_metadata.jsonl")
    write_jsonl(
        sidecar,
        [
            {"id": r["id"], "group_id": r["group_id"], "metadata": r["metadata"], "judge": r["judge"],
             "judge_total": r["judge_total"]}
            for r in selected
        ],
    )

    def dist(rows, key):
        return dict(Counter(key(r) for r in rows).most_common())

    totals = [r["judge_total"] for r in judged]
    report = {
        "funnel": funnel,
        "funnel_image_questions": funnel_img,
        "selected_questions": len({r["group_id"] for r in selected}),
        "judged_total_hist": dict(sorted(Counter(totals).items())),
        "judged_mean_total": round(statistics.mean(totals), 2),
        "domain_judged": dist(judged, lambda r: r["judge"]["domain"]),
        "domain_selected": dist(selected, lambda r: r["judge"]["domain"]),
        "answer_rank_judged": dist(judged, lambda r: min(r["metadata"]["answer_rank"], 4)),
        "answer_rank_selected": dist(selected, lambda r: min(r["metadata"]["answer_rank"], 4)),
        "mean_total_by_rank": {
            k: round(statistics.mean(r["judge_total"] for r in judged if min(r["metadata"]["answer_rank"], 4) == k), 2)
            for k in (1, 2, 3, 4)
        },
        "mean_total_by_se_score": {
            b: round(statistics.mean(r["judge_total"] for r in rows), 2)
            for b, rows in {
                "1": [r for r in judged if r["metadata"]["answer_score"] == 1],
                "2-3": [r for r in judged if 2 <= r["metadata"]["answer_score"] <= 3],
                "4-9": [r for r in judged if 4 <= r["metadata"]["answer_score"] <= 9],
                "10+": [r for r in judged if r["metadata"]["answer_score"] >= 10],
            }.items()
            if rows
        },
        "accepted_share_judged": round(statistics.mean(r["metadata"]["is_accepted"] for r in judged), 3),
        "accepted_share_selected": round(statistics.mean(r["metadata"]["is_accepted"] for r in selected), 3),
        "style_judged": style_stats(judged),
        "style_selected": style_stats(selected),
        "assistant_style_selected": dist(selected, lambda r: r["judge"]["assistant_style"]),
        "calibrated_uncertainty_selected": dist(selected, lambda r: r["judge"]["calibrated_uncertainty"]),
        "thresholds": s,
    }
    (ROOT / paths["selection_report"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {paths['train']} ({len(selected)}) and {sidecar}")


if __name__ == "__main__":
    main()
