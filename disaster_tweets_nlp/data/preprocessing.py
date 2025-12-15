from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = text.lower()
    text = _URL_RE.sub(" <url> ", text)
    text = _MENTION_RE.sub(" <user> ", text)
    text = text.replace("#", " #")
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def build_model_text(
    raw_text: str,
    keyword: str | float | None,
    location: str | float | None,
    use_keyword: bool,
    use_location: bool,
) -> str:
    parts: list[str] = [raw_text]
    if use_keyword and isinstance(keyword, str) and keyword.strip():
        parts.append(f"keyword: {keyword}")
    if use_location and isinstance(location, str) and location.strip():
        parts.append(f"location: {location}")
    return clean_text(" | ".join(parts))
