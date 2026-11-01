"""Candidate filters and text-sufficiency heuristics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.data.html_clean import has_img_markup, html_to_text
from src.data.parse_se import Post

IMAGE_REF_RE = re.compile(
    r"\b(pictur(?:e|ed)|photos?|images?|see below|shown below|look like this|"
    r"as (?:you can )?see|attached|imgur)\b|!\[|stack\.imgur",
    re.IGNORECASE,
)
IDENTIFY_RE = re.compile(
    r"\b(what (?:is|are) th(?:is|ese)|identify|identification|"
    r"what plant(?: is)? (?:this|is)|species (?:is|are) this)\b",
    re.IGNORECASE,
)
OUTDOOR_AG_RE = re.compile(
    r"\b(farm|acre|orchard|field crop|livestock|tractor)\b",
    re.IGNORECASE,
)


@dataclass
class Candidate:
    question: Post
    answer: Post
    question_text: str
    answer_text: str
    matched_tags: list[str]
    drop_reasons: list[str]

    @property
    def kept(self) -> bool:
        return not self.drop_reasons

    def to_record(self) -> dict[str, Any]:
        qid = self.question.id
        aid = self.answer.id
        user_content = self.question_text
        if self.question.title:
            user_content = f"{self.question.title.strip()}\n\n{self.question_text}".strip()
        return {
            "id": f"gardening-q{qid}-a{aid}",
            "group_id": f"gardening-q{qid}",
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": self.answer_text},
            ],
            "metadata": {
                "source": "gardening.stackexchange.com",
                "source_revision": "archive.org-stackexchange-gardening-2024-04",
                "question_id": qid,
                "answer_id": aid,
                "source_urls": [
                    f"https://gardening.stackexchange.com/questions/{qid}",
                    f"https://gardening.stackexchange.com/a/{aid}",
                ],
                "attribution": [
                    {
                        "role": "question",
                        "owner_user_id": self.question.owner_user_id,
                        "license": self.question.content_license,
                    },
                    {
                        "role": "answer",
                        "owner_user_id": self.answer.owner_user_id,
                        "license": self.answer.content_license,
                    },
                ],
                "licenses": [
                    self.question.content_license,
                    self.answer.content_license,
                ],
                "language": "en",
                "tags": self.question.tags,
                "matched_tags": self.matched_tags,
                "answer_score": self.answer.score,
                "answer_origin": "human",
                "text_sufficient": "image_dependent" not in self.drop_reasons,
                "review_status": "auto_accepted" if self.kept else "auto_dropped",
                "drop_reasons": self.drop_reasons,
            },
        }


def best_positive_answer(answers: list[Post]) -> Post | None:
    if not answers:
        return None
    best = max(answers, key=lambda a: (a.score, -a.id))
    if best.score <= 0:
        return None
    return best


def sufficiency_reasons(
    question_html: str,
    answer_html: str,
    question_text: str,
    answer_text: str,
    min_q: int,
    min_a: int,
) -> list[str]:
    reasons: list[str] = []
    combined_q = f"{question_text}"
    if len(question_text) < min_q:
        reasons.append("short_question")
    if len(answer_text) < min_a:
        reasons.append("short_answer")
    if IMAGE_REF_RE.search(question_html) or IMAGE_REF_RE.search(question_text):
        if IDENTIFY_RE.search(question_text) or IDENTIFY_RE.search(question_html):
            reasons.append("image_dependent")
        elif re.search(r"\b(this plant|these leaves|the photo)\b", question_text, re.I):
            reasons.append("image_dependent")
    if IDENTIFY_RE.search(question_text) and not re.search(
        r"\b(water|light|soil|repot|fertiliz|humid|drain)\b", question_text, re.I
    ):
        # bare identification asks without care context
        if "image_dependent" not in reasons:
            reasons.append("identification_only")
    if OUTDOOR_AG_RE.search(question_text):
        reasons.append("outdoor_agriculture")
    if not answer_text.strip() or answer_text.strip().lower() in {"deleted", "n/a"}:
        reasons.append("empty_answer")
    _ = combined_q  # kept for clarity / future heuristics
    return reasons


def build_candidates(
    questions: dict[int, Post],
    answers_by_parent: dict[int, list[Post]],
    expanded_tags: set[str],
    *,
    drop_img: bool = True,
    drop_closed: bool = True,
    min_question_chars: int = 40,
    min_answer_chars: int = 80,
    apply_sufficiency: bool = True,
) -> list[Candidate]:
    out: list[Candidate] = []
    for qid, q in questions.items():
        matched = sorted(set(q.tags) & expanded_tags)
        if not matched:
            continue
        ans = best_positive_answer(answers_by_parent.get(qid, []))
        if ans is None:
            continue
        reasons: list[str] = []
        if drop_closed and q.closed_date:
            reasons.append("closed")
        if drop_img and (has_img_markup(q.body) or has_img_markup(ans.body)):
            reasons.append("img_markup")
        q_text = html_to_text(q.body)
        a_text = html_to_text(ans.body)
        if apply_sufficiency:
            reasons.extend(
                sufficiency_reasons(
                    q.body, ans.body, q_text, a_text, min_question_chars, min_answer_chars
                )
            )
        # Deduplicate reasons while preserving order
        seen: set[str] = set()
        uniq: list[str] = []
        for r in reasons:
            if r not in seen:
                seen.add(r)
                uniq.append(r)
        out.append(
            Candidate(
                question=q,
                answer=ans,
                question_text=q_text,
                answer_text=a_text,
                matched_tags=matched,
                drop_reasons=uniq,
            )
        )
    return out
