"""Canonicalizing author names for grouping.

Debate cites name the same person a dozen ways. From one real archive ingest:

    "Nick Bostrom"                "Bostrom"
    "Bostrum, Nick. University"   "Bostrom 02"

Every check that counts *distinct sources* silently gets this wrong, and it gets
it wrong in the dangerous direction. `source_concentration` fires when a
position leans on too few authors; if one author is counted as three, the
position looks better sourced than it is and the finding never fires. A
false negative in a tool whose job is finding weaknesses is worse than a false
positive, because nothing tells you it happened.

Two layers, deliberately separate:

`author_key` is conservative and cheap -- casefold, strip institutional tails
and honorifics, resolve "Last, First" to the surname. It merges the variants
that are literally the same string wearing different punctuation.

`group_authors` is fuzzy and only ever applied *within one position* (a handful
of names), where a false merge is visible and cheap. It catches the typo pairs
("Bostrom"/"Bostrum") that no amount of normalization will unify, using the
shared-prefix and edit-distance behavior of actual misspelled surnames rather
than a general string metric.
"""

from __future__ import annotations

import re
import unicodedata

# Words that trail a name and describe where they work, not who they are.
_INSTITUTION = {
    "university", "univ", "college", "school", "institute", "institution",
    "department", "dept", "professor", "prof", "phd", "ph", "jd", "md",
    "director", "fellow", "senior", "associate", "assistant", "chair",
    "research", "researcher", "analyst", "economist", "scientist", "lecturer",
    "of", "at", "the", "for", "and", "in", "on", "law", "review", "press",
    "et", "al", "inc", "llc", "corp", "center", "centre", "programme",
    "program", "staff", "writer", "correspondent", "editor", "reporter",
}

_HONORIFIC = {"dr", "mr", "mrs", "ms", "prof", "sir", "rev", "hon"}


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def author_key(name: str | None) -> str:
    """A canonical key for an author name. Empty string when unusable."""
    if not name:
        return ""
    s = _strip_accents(str(name)).lower()

    # "Bostrum, Nick. University" -> surname is what precedes the comma
    if "," in s:
        s = s.split(",", 1)[0]

    # Keep alphanumerics. Stripping digits merges any two names that differ
    # only by a trailing number, and over-merging is the failure that matters
    # here: it makes a single-source position look diverse and silences the one
    # check meant to catch that.
    s = re.sub(r"[^a-z0-9\s'-]", " ", s)
    tokens = [t for t in s.split() if t]
    tokens = [t for t in tokens if t not in _HONORIFIC]
    tokens = [t for t in tokens if t not in _INSTITUTION]
    # drop bare initials ("j.", "a") and the year that trails most debate cites
    tokens = [t for t in tokens if len(t) > 1 and not t.isdigit()]
    if not tokens:
        return ""

    # For a plain "Nick Bostrom", the surname is the last token; for a lone
    # "Bostrom" it is the only one. Either way the last token is the key, which
    # is also how debate cites are spoken and indexed.
    return tokens[-1]


def _similar(a: str, b: str) -> bool:
    """Are these two surname keys plausibly the same misspelled name?

    Not a general string metric. Real variants in this corpus differ by one or
    two characters at the end ("bostrom"/"bostrum") or by a dropped letter, and
    they always share a long prefix. Requiring a shared prefix before allowing
    an edit-distance match keeps "tickell" and "mitchell" apart, which a bare
    distance threshold does not.
    """
    if a == b:
        return True
    if len(a) < 5 or len(b) < 5:
        return False          # short names collide too easily to merge
    # A difference in a digit is never a misspelling -- it is a deliberate
    # distinction (identifiers, disambiguated names). Only letters get the
    # benefit of the typo doubt.
    if any(c.isdigit() for c in a) or any(c.isdigit() for c in b):
        return False
    if abs(len(a) - len(b)) > 1:
        return False
    prefix = 0
    for x, y in zip(a, b):
        if x != y:
            break
        prefix += 1
    if prefix < max(4, min(len(a), len(b)) - 2):
        return False
    # at most one substitution or deletion after the shared prefix
    return _edits_within_one(a, b)


def _edits_within_one(a: str, b: str) -> bool:
    if a == b:
        return True
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x != y) <= 1
    if len(a) > len(b):
        a, b = b, a
    # b is exactly one longer: check a is b with one deletion
    for i in range(len(b)):
        if a == b[:i] + b[i + 1:]:
            return True
    return False


def group_authors(names) -> dict[str, str]:
    """Map each raw name to a canonical key, merging near-identical surnames.

    Intended for the small set of names attached to one position. Merging across
    a whole corpus with this would be reckless.
    """
    keys = {n: author_key(n) for n in names}
    canon: dict[str, str] = {}
    for key in sorted({k for k in keys.values() if k}, key=lambda x: (len(x), x)):
        for existing in canon:
            if _similar(key, existing):
                canon[key] = canon[existing]
                break
        else:
            canon[key] = key
    return {name: canon.get(k, k) for name, k in keys.items()}


def distinct_authors(names) -> set[str]:
    """How many genuinely different people are cited here."""
    return {v for v in group_authors(names).values() if v}
