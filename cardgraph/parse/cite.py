"""Best-effort parsing of debate cite lines.

A cite in the wild looks like any of:

    Ember 24 (Ember Climate, "Global Electricity Review 2024", 5/8/24, https://...)
    Blackhurst et al. 2023 -- Prof of Civil Eng, Carnegie Mellon [Michael ...]
    Goldman Sachs '25
    Hoffmann, Allianz Research, 4-2-2026, "Data centers and the grid"

There is no standard. We extract what we can and always keep `raw`.
"""

from __future__ import annotations

import re

from ..models import Cite

_URL = re.compile(r"https?://[^\s\)\]\>,]+")

# "24", "'24", "2024", "24-25" -- capture the year token
_YEAR = re.compile(
    r"(?<![\d/-])(?:'|‘|’)?((?:19|20)\d{2}|\d{2})(?![\d])"
)

# Leading author chunk: capitalized words, possibly with "et al." and commas,
# stopping at the first year / paren / quote.
_AUTHOR = re.compile(
    r"^\s*(?:\[[^\]]*\]\s*)?"
    r"([A-Z][\w.'’-]*(?:\s+(?:et\s+al\.?|and|&|de|van|von|der))?"
    r"(?:[,\s]+[A-Z][\w.'’-]*){0,3})"
)

_QUOTED_TITLE = re.compile(r"[\"“]([^\"”]{6,200})[\"”]")

_CRED_HINTS = re.compile(
    r"(prof(?:essor)?\b|ph\.?d|j\.?d\b|m\.?d\b|senior fellow|director\b|"
    r"analyst\b|research(?:er)?\b|economist\b|lecturer\b|chair\b)",
    re.IGNORECASE,
)


def _normalize_year(tok: str) -> int | None:
    try:
        v = int(tok)
    except ValueError:
        return None
    if v >= 1900:
        return v
    # two-digit: 90-99 -> 1990s, else 2000s
    return 1900 + v if v >= 90 else 2000 + v


# Words that begin a sentence, not a name. A cite line that starts with one of
# these is almost always analytic prose that was misidentified upstream; letting
# it through fills the author index with "The" and "I" and quietly corrupts
# every source-concentration finding that counts distinct authors.
_NOT_AUTHORS = {
    "the", "a", "an", "i", "we", "they", "he", "she", "it", "this", "that",
    "these", "those", "there", "here", "and", "but", "or", "if", "as", "in",
    "on", "at", "by", "for", "of", "to", "from", "with", "my", "our", "your",
    "his", "her", "their", "its", "no", "not", "first", "second", "third",
    "next", "thus", "therefore", "however", "moreover", "furthermore", "also",
    "resolved", "contention", "value", "criterion", "framework", "observation",
    "prefer", "because", "since", "when", "while", "all", "any", "every",
}


def _plausible_author(author: str) -> bool:
    """Reject sentence-openers and single letters masquerading as surnames."""
    if not author:
        return False
    tokens = [t.strip(".,").lower() for t in author.split() if t.strip(".,")]
    if not tokens:
        return False
    if tokens[0] in _NOT_AUTHORS:
        return False
    # a lone initial ("A.", "I") is not an attribution
    if len(tokens) == 1 and len(tokens[0]) <= 2:
        return False
    return True


def parse_cite(raw: str) -> Cite:
    raw = (raw or "").strip()
    c = Cite(raw=raw)
    if not raw:
        return c

    m = _URL.search(raw)
    if m:
        c.url = m.group(0).rstrip(".,;")

    # Year: prefer one near the start (that is the debate-convention year),
    # falling back to any year in the line.
    head = raw[:80]
    ym = _YEAR.search(head) or _YEAR.search(raw)
    if ym:
        c.year = _normalize_year(ym.group(1))

    am = _AUTHOR.match(raw)
    if am:
        author = am.group(1).strip(" ,")
        # Guard against swallowing an all-caps publication as the author.
        ok = (author and not author.isupper()) or len(author.split()) <= 3
        if ok and _plausible_author(author):
            c.author = author

    tm = _QUOTED_TITLE.search(raw)
    if tm:
        c.title = tm.group(1).strip()

    cm = _CRED_HINTS.search(raw)
    if cm:
        # take the clause containing the credential hint
        start = raw.rfind(",", 0, cm.start()) + 1
        end = raw.find(",", cm.end())
        end = end if end != -1 else min(len(raw), cm.end() + 60)
        c.credentials = raw[start:end].strip(" ,[]")

    # Publication: text inside the first paren that is not the URL/title
    pm = re.search(r"\(([^)]{3,120})\)", raw)
    if pm:
        inner = pm.group(1)
        inner = _URL.sub("", inner)
        inner = _QUOTED_TITLE.sub("", inner)
        inner = inner.strip(" ,;")
        if inner and not inner.replace("/", "").replace("-", "").isdigit():
            c.publication = inner.split(",")[0].strip() or None

    return c


def looks_like_cite(text: str) -> bool:
    """Heuristic for 'this paragraph is a cite, not tag or body'.

    Cites are short, contain a year near the front, and usually carry a URL,
    a quoted title, or a parenthetical. Body paragraphs are long prose.
    """
    t = (text or "").strip()
    if not (8 <= len(t) <= 500):
        return False
    if not _YEAR.search(t[:80]):
        return False
    signals = 0
    if _URL.search(t):
        signals += 1
    if _QUOTED_TITLE.search(t):
        signals += 1
    if "(" in t and ")" in t:
        signals += 1
    if _CRED_HINTS.search(t):
        signals += 1
    if len(t) < 200:
        signals += 1
    return signals >= 2
