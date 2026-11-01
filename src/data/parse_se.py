"""Parse Gardening Stack Exchange Posts.xml."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class Post:
    id: int
    post_type_id: int
    body: str
    score: int
    owner_user_id: str | None = None
    content_license: str | None = None
    creation_date: str | None = None
    title: str | None = None
    tags: list[str] = field(default_factory=list)
    parent_id: int | None = None
    accepted_answer_id: int | None = None
    closed_date: str | None = None


def parse_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    # SE dump format: |houseplants|sunlight|
    parts = [p for p in raw.strip().split("|") if p]
    return parts


def iter_posts(posts_xml: Path | str) -> Iterator[Post]:
    path = Path(posts_xml)
    # iterparse keeps memory bounded for ~56MB Posts.xml
    for _event, elem in ET.iterparse(path, events=("end",)):
        if elem.tag != "row":
            continue
        attrs = elem.attrib
        post_type = int(attrs["PostTypeId"])
        post = Post(
            id=int(attrs["Id"]),
            post_type_id=post_type,
            body=attrs.get("Body", ""),
            score=int(attrs.get("Score", "0")),
            owner_user_id=attrs.get("OwnerUserId"),
            content_license=attrs.get("ContentLicense"),
            creation_date=attrs.get("CreationDate"),
            title=attrs.get("Title"),
            tags=parse_tags(attrs.get("Tags")),
            parent_id=int(attrs["ParentId"]) if attrs.get("ParentId") else None,
            accepted_answer_id=(
                int(attrs["AcceptedAnswerId"]) if attrs.get("AcceptedAnswerId") else None
            ),
            closed_date=attrs.get("ClosedDate"),
        )
        yield post
        elem.clear()


def load_questions_and_answers(
    posts_xml: Path | str,
) -> tuple[dict[int, Post], dict[int, list[Post]]]:
    questions: dict[int, Post] = {}
    answers_by_parent: dict[int, list[Post]] = {}
    for post in iter_posts(posts_xml):
        if post.post_type_id == 1:
            questions[post.id] = post
        elif post.post_type_id == 2 and post.parent_id is not None:
            answers_by_parent.setdefault(post.parent_id, []).append(post)
    return questions, answers_by_parent
