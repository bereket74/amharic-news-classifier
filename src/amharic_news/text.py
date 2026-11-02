from __future__ import annotations

import re
import unicodedata

_ZERO_WIDTH = re.compile("[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_SPACE = re.compile(r"\s+")


def normalize_inference_text(value: str | None) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    text = _ZERO_WIDTH.sub("", text)
    return _SPACE.sub(" ", text).strip()


def join_headline_and_article(headline: str | None, article: str | None) -> str:
    title = normalize_inference_text(headline)
    body = normalize_inference_text(article)
    if title and body:
        return f"{title}\n{body}"
    return title or body
