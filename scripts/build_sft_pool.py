#!/usr/bin/env python3
"""Build the SFT v2 candidate pool: every positive-score answer on the full Gardening SE dump.

Deterministic prefilters only; quality is decided later by the LLM judge.
Needs the Qwen tokenizer (run with .venv-colab):

  .venv-colab/bin/python scripts/build_sft_pool.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.html_clean import has_img_markup, html_to_text
from src.data.io_jsonl import read_jsonl, write_jsonl
from src.data.parse_se import load_questions_and_answers
from src.train.sft_qlora import load_yaml

IMG_TAG_RE = re.compile(r"<\s*img\b[^>]*>", re.I)
IMGUR_RE = re.compile(r"imgur\.com", re.I)
# Applied only to questions that had images: any hint that the text points at the picture.
Q_IMAGE_REF_RE = re.compile(
    r"\b(pictures?|pictured|photos?|photographs?|pics?|images?|imgur|attached|attachment|"
    r"see below|shown|look(?:s)? like this|like this one|as you can see|you can see|close-?ups?|"
    r"screenshots?|this (?:one|plant|tree|bug|insect|thing|spot|growth|leaf|flower|mushroom|fungus|weed)|"
    r"these (?:plants|leaves|spots|bugs|insects|things|ones)|what is (?:this|it)|what are these)\b",
    re.I,
)
A_IMAGE_REF_RE = re.compile(
    r"\b((?:your|the|this|that|first|second|last) (?:pictures?|photos?|pics?|images?)|"
    r"in the pic|from the (?:picture|photo|image)|I can see|can be seen|judging (?:by|from) the)\b",
    re.I,
)


def question_has_image(body: str) -> bool:
    return has_img_markup(body) or bool(IMGUR_RE.search(body or ""))


def main() -> None:
    cfg = load_yaml(ROOT / "configs/sft_v2.yaml")
    pcfg = cfg["pool"]
    system_prompt = load_yaml(ROOT / "configs/model.yaml")["system_prompt"].strip()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", cache_dir=str(ROOT / ".cache/hf"))

    holdout = set()
    for f in pcfg["holdout_files"]:
        holdout |= {r["group_id"] for r in read_jsonl(ROOT / f)}

    questions, answers = load_questions_and_answers(ROOT / "data/raw/Posts.xml")
    drop_tags = set(pcfg["drop_tags"])
    rows, reasons_count = [], Counter()
    for qid, q in sorted(questions.items()):
        q_img = question_has_image(q.body)
        q_text = html_to_text(q.body)
        title = (q.title or "").strip()
        user = f"{title}\n\n{q_text}".strip()
        judge_q = f"{title}\n\n{html_to_text(IMG_TAG_RE.sub(' [IMAGE REMOVED] ', q.body))}".strip()
        q_reasons = []
        if q.closed_date:
            q_reasons.append("closed")
        if f"gardening-q{qid}" in holdout:
            q_reasons.append("eval_holdout")
        if drop_tags & set(q.tags):
            q_reasons.append("tag_identification")
        if q_img and Q_IMAGE_REF_RE.search(user):
            q_reasons.append("q_refers_to_image")
        for a in answers.get(qid, []):
            if a.score < pcfg["min_answer_score"]:
                continue
            a_text = html_to_text(a.body)
            reasons = list(q_reasons)
            if has_img_markup(a.body):
                reasons.append("a_img")
            if len(a_text) < pcfg["min_answer_chars"]:
                reasons.append("short_answer")
            if A_IMAGE_REF_RE.search(a_text):
                reasons.append("a_refers_to_image")
            messages = [{"role": "user", "content": user}, {"role": "assistant", "content": a_text}]
            n_tokens = len(
                tok.apply_chat_template(
                    [{"role": "system", "content": system_prompt}, *messages], tokenize=True
                )
            )
            if n_tokens > pcfg["max_total_tokens"]:
                reasons.append("too_long")
            for r in reasons:
                reasons_count[r] += 1
            rows.append(
                {
                    "id": f"gardening-q{qid}-a{a.id}",
                    "group_id": f"gardening-q{qid}",
                    "messages": messages,
                    "judge_question": judge_q,
                    "metadata": {
                        "question_id": qid,
                        "answer_id": a.id,
                        "tags": q.tags,
                        "question_score": q.score,
                        "answer_score": a.score,
                        "is_accepted": q.accepted_answer_id == a.id,
                        "question_has_image": q_img,
                        "n_tokens": n_tokens,
                        "licenses": [q.content_license, a.content_license],
                        "source_urls": [
                            f"https://gardening.stackexchange.com/questions/{qid}",
                            f"https://gardening.stackexchange.com/a/{a.id}",
                        ],
                        "prefilter_reasons": reasons,
                    },
                }
            )

    # rank among positive answers of the same question (1 = top score)
    by_q: dict[int, list[dict]] = {}
    for r in rows:
        by_q.setdefault(r["metadata"]["question_id"], []).append(r)
    for rs in by_q.values():
        rs.sort(key=lambda r: (-r["metadata"]["answer_score"], r["metadata"]["answer_id"]))
        for i, r in enumerate(rs, 1):
            r["metadata"]["answer_rank"] = i

    paths = cfg["paths"]
    write_jsonl(ROOT / paths["pool"], rows)
    passed = [r for r in rows if not r["metadata"]["prefilter_reasons"]]
    report = {
        "questions_total": len(questions),
        "answers_total": sum(len(v) for v in answers.values()),
        "answers_score_ge_min": len(rows),
        "prefilter_passed": len(passed),
        "prefilter_passed_questions": len({r["group_id"] for r in passed}),
        "prefilter_passed_with_image_question": sum(
            r["metadata"]["question_has_image"] for r in passed
        ),
        "drop_reason_counts": dict(reasons_count.most_common()),
    }
    (ROOT / paths["pool_report"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
