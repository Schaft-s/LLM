"""Automatic format checkers."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


STEP_RE = re.compile(
    r"(?:^|\n)\s*(?:(?:step\s*)?([1-3])[\.\)\:]|(?:^|\n)\s*([1-3])[\.\)\:])\s*(.+)",
    re.IGNORECASE,
)


def _extract_numbered_steps(text: str) -> dict[int, str]:
    found: dict[int, str] = {}
    # Prefer line-start numbered items
    for line in (text or "").splitlines():
        m = re.match(r"\s*(?:step\s*)?(\d+)[\.\)\:]\s*(.+)$", line, re.IGNORECASE)
        if not m:
            continue
        n = int(m.group(1))
        body = m.group(2).strip()
        if body and n not in found:
            found[n] = body
    return found


def check_three_numbered_steps(text: str) -> CheckResult:
    """Pass iff exactly steps 1,2,3 exist, non-empty, and not near-duplicates."""
    steps = _extract_numbered_steps(text)
    if set(steps) != {1, 2, 3}:
        return CheckResult(
            name="three_numbered_steps",
            passed=False,
            detail=f"expected steps 1/2/3, found {sorted(steps)}",
        )
    bodies = [steps[i] for i in (1, 2, 3)]
    if any(len(b) < 3 for b in bodies):
        return CheckResult(
            name="three_numbered_steps",
            passed=False,
            detail="one or more steps too short/empty",
        )
    norm = [re.sub(r"\W+", " ", b.lower()).strip() for b in bodies]
    if len(set(norm)) < 3:
        return CheckResult(
            name="three_numbered_steps",
            passed=False,
            detail="duplicated step content",
        )
    return CheckResult(name="three_numbered_steps", passed=True, detail="ok")


def check_nonempty(text: str, min_chars: int = 1) -> CheckResult:
    ok = bool(text) and len(text.strip()) >= min_chars
    return CheckResult("nonempty", ok, "ok" if ok else "empty_or_too_short")


def check_no_severe_repetition(text: str, min_line_len: int = 20) -> CheckResult:
    lines = [ln.strip() for ln in (text or "").splitlines() if len(ln.strip()) >= min_line_len]
    if len(lines) >= 3 and len(set(lines)) == 1:
        return CheckResult("no_severe_repetition", False, "identical long lines repeated")
    return CheckResult("no_severe_repetition", True, "ok")
