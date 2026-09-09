"""Core data model for cardgraph.

The unit of debate evidence is a *card*: a tag (the claim the debater asserts),
a cite (the source), a body (the full quoted passage), and -- critically -- the
*read text*: the subset of the body that is actually spoken aloud in a round,
marked in Verbatim by underlining and/or highlighting.

Almost every naive debate-evidence tool indexes the body. That is wrong. The
body is context; the read text is the argument. Retrieval quality lives or dies
on this distinction, so it is a first-class field here.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Side(str, Enum):
    AFF = "aff"
    NEG = "neg"
    BOTH = "both"
    UNKNOWN = "unknown"


class NodeKind(str, Enum):
    """Levels of the Verbatim outline hierarchy, plus our own rollups."""

    POCKET = "pocket"        # Heading 1 -- e.g. "Politics DA"
    HAT = "hat"              # Heading 2 -- e.g. "1NC"
    BLOCK = "block"          # Heading 3 -- e.g. "AT: Link Turn"
    TAG = "tag"              # Heading 4 -- the card's claim
    CONTENTION = "contention"  # rollup we infer for case-level structure


@dataclass
class Cite:
    """A parsed source line. Debate cites are wildly inconsistent; every field
    here is best-effort and `raw` is always preserved verbatim."""

    raw: str
    author: str | None = None
    year: int | None = None
    publication: str | None = None
    title: str | None = None
    url: str | None = None
    credentials: str | None = None

    def short(self) -> str:
        if self.author and self.year:
            return f"{self.author} {str(self.year)[-2:]}"
        if self.author:
            return self.author
        return (self.raw or "")[:40]


@dataclass
class Card:
    """One piece of evidence."""

    tag: str
    cite: Cite
    body: str
    read_text: str
    # provenance
    source_id: str = ""
    source_path: str = ""
    ordinal: int = 0
    # outline context, outermost first: ["Politics DA", "1NC", "AT: Link Turn"]
    path: list[str] = field(default_factory=list)
    side: Side = Side.UNKNOWN
    # derived
    card_id: str = ""
    emphasis_text: str = ""   # doubly-marked (bold+underline / highlighted)
    read_ratio: float = 0.0   # len(read_text) / len(body)
    warrant_flags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.card_id:
            self.card_id = self.compute_id()
        if self.body:
            self.read_ratio = round(len(self.read_text) / max(len(self.body), 1), 4)

    def compute_id(self) -> str:
        """Content hash, keyed on cite + read_text rather than the tag: the same
        card gets re-tagged constantly as it moves between teams, and we want
        those to collapse to one id so re-cutting is visible.

        The exception is a card with no read text. Those are not evidence yet --
        an unfilled outline placeholder, or a file whose marking the parser
        missed -- and hashing them on content alone makes every one of them
        identical, so a 33-tag skeleton silently collapses into a single row.
        Fall back to position, which is unique by construction.
        """
        if self.read_text.strip():
            basis = f"{self.cite.raw}||{self.read_text}"
        else:
            basis = f"{self.source_id}||{self.ordinal}||{self.tag}||{self.cite.raw}"
        return hashlib.sha1(basis.encode("utf-8", "ignore")).hexdigest()[:16]

    @property
    def block(self) -> str | None:
        return self.path[-1] if self.path else None

    @property
    def pocket(self) -> str | None:
        return self.path[0] if self.path else None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["side"] = self.side.value
        return d


@dataclass
class OutlineNode:
    """A node in the document outline (pocket / hat / block). Cards hang off
    these. This is what lets the UI show 'every contention and block' as a tree
    rather than a flat search index."""

    title: str
    kind: NodeKind
    path: list[str] = field(default_factory=list)
    source_id: str = ""
    card_ids: list[str] = field(default_factory=list)
    children: list["OutlineNode"] = field(default_factory=list)

    @property
    def node_id(self) -> str:
        basis = "||".join([self.source_id, *self.path, self.title])
        return hashlib.sha1(basis.encode("utf-8", "ignore")).hexdigest()[:16]


@dataclass
class Source:
    """A document we ingested."""

    source_id: str
    path: str
    title: str
    origin: str = "local"       # adapter name: openev, caselist-archive, local
    license: str = "unknown"
    fetched_at: str | None = None
    url: str | None = None
    card_count: int = 0


# ---------------------------------------------------------------------------
# Block-name conventions. Debate blocks are named with a small, stable set of
# prefixes. Recognizing them is what turns a pile of headings into a graph:
# "AT: Ratepayer Harm" is an *answer to* a thing named "Ratepayer Harm".
# ---------------------------------------------------------------------------

ANSWER_PREFIXES = [
    r"^a[/\\]?t:?\s+",          # AT:  A/T:  AT
    r"^at\s+the\s+",
    r"^answers?\s+to:?\s+",
    r"^ans\s+",
    r"^2ac\s+at:?\s+",
    r"^1ar\s+at:?\s+",
    r"^-{0,2}\s*at\s*-{0,2}\s+",
]
_ANSWER_RE = re.compile("|".join(ANSWER_PREFIXES), re.IGNORECASE)

FRONTLINE_PREFIXES = re.compile(
    r"^(2ac|1ar|2ar|1nc|2nc|1nr|block|overview|ov)\b[\s:—-]*", re.IGNORECASE
)


def answers_target(heading: str) -> str | None:
    """If `heading` is an answer block, return the thing it answers.

    >>> answers_target("AT: Ratepayer Harm")
    'Ratepayer Harm'
    >>> answers_target("Uniqueness") is None
    True
    """
    m = _ANSWER_RE.match(heading.strip())
    if not m:
        return None
    target = heading[m.end():].strip(" :-—–")
    return target or None


def strip_speech_prefix(heading: str) -> str:
    return FRONTLINE_PREFIXES.sub("", heading.strip()).strip(" :-—–")


def infer_side(path: list[str], title: str = "") -> Side:
    """Guess aff/neg from outline context. Crude on purpose -- it is a filter
    hint in the UI, never something we assert as fact."""
    hay = " ".join([*path, title]).lower()
    aff_hits = len(re.findall(r"\b(1ac|2ac|1ar|2ar|aff|affirmative|advantage)\b", hay))
    neg_hits = len(re.findall(r"\b(1nc|2nc|1nr|2nr|neg|negative|da|disad|cp|counterplan|kritik|\bk\b)\b", hay))
    if aff_hits > neg_hits:
        return Side.AFF
    if neg_hits > aff_hits:
        return Side.NEG
    return Side.UNKNOWN
