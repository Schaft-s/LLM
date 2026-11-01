"""HTML → plain text for Stack Exchange posts."""

from __future__ import annotations

import html
import re

from html.parser import HTMLParser


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = False
        self._li_index = 0
        self._in_ol = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in {"script", "style"}:
            self._skip = True
            return
        if t in {"p", "br", "div", "tr", "h1", "h2", "h3", "h4", "hr"}:
            self.parts.append("\n")
        elif t == "li":
            if self._in_ol:
                self._li_index += 1
                self.parts.append(f"\n{self._li_index}. ")
            else:
                self.parts.append("\n- ")
        elif t == "ol":
            self._in_ol = True
            self._li_index = 0
        elif t == "ul":
            self._in_ol = False

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in {"script", "style"}:
            self._skip = False
        elif t in {"p", "div", "tr", "li"}:
            self.parts.append("\n")
        elif t == "ol":
            self._in_ol = False
            self._li_index = 0

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


_WS_RE = re.compile(r"[ \t]+")
_BLANK_RE = re.compile(r"\n{3,}")


def html_to_text(raw: str) -> str:
    if not raw:
        return ""
    unescaped = html.unescape(raw)
    parser = _TextExtractor()
    parser.feed(unescaped)
    parser.close()
    text = "".join(parser.parts)
    text = _WS_RE.sub(" ", text)
    text = _BLANK_RE.sub("\n\n", text)
    return text.strip()


def has_img_markup(raw_html: str) -> bool:
    return bool(re.search(r"<\s*img\b", raw_html or "", flags=re.IGNORECASE))
