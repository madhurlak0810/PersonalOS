"""Helpers every job board adapter uses to turn a board's JSON into a payload.

Board responses are untrusted input in two senses. Their *text* is written by
whoever published the posting and is carried as data, never read for
instructions. Their *shape* is whatever the board sent: a field documented as
an object can arrive as a string, a list as null. The accessors here never
raise on a wrong shape -- they answer "absent" -- so one malformed posting
cannot fail a search, and `DictPostingNormalizer` makes the final call on
whether what is left is a posting.
"""

import html
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from personalos.domain.job_search import SearchProfile

#: Tags whose end marks a line break when HTML is flattened to text.
_BLOCK_TAGS = re.compile(r"</?(?:p|div|br|li|ul|ol|h[1-6]|tr)\b[^>]*>", re.IGNORECASE)
_ANY_TAG = re.compile(r"<[^>]+>")
_NON_TEXT = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)


def html_to_text(content: str) -> str:
    """Flatten board HTML to plain text, one line per block.

    Unescaped before tags are removed and once more after: Greenhouse
    entity-escapes its HTML, and entities inside the text survive the first
    pass. Script and style bodies are dropped, not flattened.
    """
    text = _NON_TEXT.sub(" ", html.unescape(content))
    text = _ANY_TAG.sub("", _BLOCK_TAGS.sub("\n", text))
    lines = (" ".join(line.split()) for line in html.unescape(text).splitlines())
    return "\n".join(line for line in lines if line)


def mapping(value: Any) -> Mapping[str, Any]:
    """`value` if it is a JSON object, else an empty one."""
    return value if isinstance(value, Mapping) else {}


def items(value: Any) -> Sequence[Any]:
    """`value` if it is a JSON array, else an empty one."""
    return value if isinstance(value, list) else []


def text(value: Any) -> str | None:
    """`value` if it is non-blank text, else `None`."""
    return value if isinstance(value, str) and value.strip() else None


def number(value: Any) -> int | None:
    """`value` as a whole number if it is a JSON number, else `None`."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


def epoch_ms_to_iso(value: Any) -> str | None:
    """A millisecond Unix timestamp as an ISO-8601 UTC string, or `None`."""
    ms = number(value)
    if ms is None or ms <= 0:
        return None
    try:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def is_remote(*signals: Any) -> bool:
    """True when any signal is `True` or text mentioning remote work."""
    return any(
        signal is True or (isinstance(signal, str) and "remote" in signal.lower())
        for signal in signals
    )


def unique(values: Sequence[str | None]) -> list[str]:
    """The non-blank values, first occurrence kept, in order."""
    seen: dict[str, None] = {}
    for value in values:
        if value and value not in seen:
            seen[value] = None
    return list(seen)


def search_terms(profile: SearchProfile) -> list[str]:
    """The profile's roles and keywords, lower-cased, blanks dropped."""
    return [term.lower() for term in (*profile.target_roles, *profile.keywords) if term.strip()]


def matches(payload: Mapping[str, Any], profile: SearchProfile) -> bool:
    """True when a posting mentions any of the profile's roles or keywords.

    A profile with neither matches everything. This is a coarse recall filter
    standing in for the search endpoint most boards do not have; precision is
    `hard_filter` and the scorer's job.
    """
    terms = search_terms(profile)
    if not terms:
        return True
    haystack = f"{payload.get('title') or ''} {payload.get('description') or ''}".lower()
    return any(term in haystack for term in terms)


__all__ = [
    "epoch_ms_to_iso",
    "html_to_text",
    "is_remote",
    "items",
    "mapping",
    "matches",
    "number",
    "search_terms",
    "text",
    "unique",
]
