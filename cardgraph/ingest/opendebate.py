"""OpenDebateEvidence: the caselist corpus, already published as a dataset.

This adapter is the answer to the question the rest of `ingest/` exists to
avoid. OpenCaseList is login-gated and `policy.py` refuses to scrape it. But the
same evidence was published as a research dataset -- ~3.5M cards spanning
2014-2022, PII-anonymized, released *with the OpenCaseList project's explicit
blessing* (Hardy et al., "OpenDebateEvidence", arXiv:2406.14657). So the data is
available through the front door, and no part of this project needs to go
through a login it was not invited past.

The schema also happens to be a near-exact match for `models.Card`, including
the one field that matters most:

    tag       -> Card.tag              the debater's claim
    fullcite  -> Card.cite.raw         (falls back to `cite`)
    fulltext  -> Card.body             the full quoted passage
    spoken    -> Card.read_text        THE UNDERLINED PORTION, pre-extracted
    pocket/hat/block -> Card.path      the outline hierarchy
    side      -> Card.side             "A"/"N"

`spoken` is why this is worth building. Everywhere else in this project the read
text has to be recovered from .docx run properties, which is the hardest and
most failure-prone step in the pipeline. Here it arrives already separated.

Two variants:
    Yusuf5/OpenCaselist                     full, ~4.8M rows
    Hellisotherpeople/OpenCaseList-Deduplicated   ~78% of near-dupes removed

Prefer the deduplicated one unless you specifically want duplicate counts as a
quality signal (`duplicateCount` is a decent proxy for how many teams thought a
card was worth reading).

Streaming is the default and non-negotiable at this scale: the full dataset does
not fit in memory and you almost never want all of it. Filter by topic, year or
event and take what you need.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from ..models import Card, Cite, NodeKind, OutlineNode, Side
from ..parse.cite import parse_cite

FULL = "Yusuf5/OpenCaselist"
DEDUPED = "Hellisotherpeople/OpenCaseList-Deduplicated"

CITATION = ("Hardy et al., OpenDebateEvidence: A Massive-Scale Argument Mining "
            "and Summarization Dataset, arXiv:2406.14657")

_SIDE = {"a": Side.AFF, "aff": Side.AFF, "affirmative": Side.AFF,
         "n": Side.NEG, "neg": Side.NEG, "negative": Side.NEG}


def _clean(s) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class DatasetFilter:
    """What slice to pull. Applied row-wise while streaming."""

    query: str | None = None          # substring match over tag/fulltext
    year_min: int | None = None
    year_max: int | None = None
    event: str | None = None          # "ld" | "cx"
    side: str | None = None           # "A" | "N"
    min_duplicate_count: int | None = None
    require_spoken: bool = True       # rows with no read text are near-useless

    def matches(self, row: dict) -> bool:
        if self.require_spoken and not _clean(row.get("spoken")):
            return False
        if self.event and _clean(row.get("event")).lower() != self.event.lower():
            return False
        if self.side and _clean(row.get("side")).upper() != self.side.upper():
            return False
        year = row.get("year")
        if isinstance(year, int):
            if self.year_min and year < self.year_min:
                return False
            if self.year_max and year > self.year_max:
                return False
        dc = row.get("duplicateCount")
        if self.min_duplicate_count and isinstance(dc, int):
            if dc < self.min_duplicate_count:
                return False
        if self.query:
            q = self.query.lower()
            hay = " ".join(str(row.get(k) or "") for k in
                           ("tag", "fulltext", "spoken", "cite")).lower()
            if q not in hay:
                return False
        return True


def row_to_card(row: dict, source_id: str) -> Card | None:
    """Map one dataset row onto a Card. Returns None for unusable rows."""
    tag = _clean(row.get("tag"))
    spoken = _clean(row.get("spoken"))
    body = _clean(row.get("fulltext")) or spoken
    if not tag or not (spoken or body):
        return None

    raw_cite = _clean(row.get("fullcite")) or _clean(row.get("cite"))
    cite: Cite = parse_cite(raw_cite) if raw_cite else Cite(raw="")
    if cite.year is None and isinstance(row.get("year"), int):
        cite.year = row["year"]

    path = [_clean(row.get(k)) for k in ("pocket", "hat", "block")]
    path = [p for p in path if p]

    side = _SIDE.get(_clean(row.get("side")).lower(), Side.UNKNOWN)

    card = Card(
        tag=tag, cite=cite, body=body, read_text=spoken,
        source_id=source_id,
        source_path=f"{source_id}#{row.get('id')}",
        ordinal=int(row.get("id") or 0) if str(row.get("id") or "").isdigit() else 0,
        path=path, side=side,
    )
    return card


def stream_cards(
    dataset: str = DEDUPED,
    *,
    filt: DatasetFilter | None = None,
    limit: int = 5000,
    split: str = "train",
    progress=None,
):
    """Yield Cards from the dataset, streaming.

    Requires `pip install datasets` and network access to huggingface.co. It is
    a genuine dependency on a third-party host: if your environment blocks it
    (some corporate and sandboxed networks do), the failure is a clean
    ImportError or connection error here rather than a silent empty ingest.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "OpenDebateEvidence ingest needs the `datasets` package: "
            "pip install datasets") from exc

    say = progress or (lambda *_: None)
    filt = filt or DatasetFilter()
    source_id = "ode:" + hashlib.sha1(dataset.encode()).hexdigest()[:8]

    say(f"streaming {dataset} (split={split})")
    ds = load_dataset(dataset, split=split, streaming=True)

    kept = 0
    for i, row in enumerate(ds):
        if kept >= limit:
            break
        if not filt.matches(row):
            continue
        card = row_to_card(dict(row), source_id)
        if card is None:
            continue
        kept += 1
        if kept % 500 == 0:
            say(f"  {kept} cards kept from {i + 1} rows scanned")
        yield card


def build_outline(cards: list[Card], source_id: str,
                  title: str = "OpenDebateEvidence") -> OutlineNode:
    """Rebuild the pocket/hat/block tree from the flat rows."""
    root = OutlineNode(title=title, kind=NodeKind.POCKET, path=[],
                       source_id=source_id)
    index: dict[tuple, OutlineNode] = {(): root}

    def node_for(path: list[str]) -> OutlineNode:
        key = tuple(path)
        if key in index:
            return index[key]
        parent = node_for(path[:-1])
        kind = [NodeKind.POCKET, NodeKind.HAT, NodeKind.BLOCK][min(len(path) - 1, 2)]
        n = OutlineNode(title=path[-1], kind=kind, path=list(path),
                        source_id=source_id)
        parent.children.append(n)
        index[key] = n
        return n

    for c in cards:
        (node_for(c.path) if c.path else root).card_ids.append(c.card_id)
    return root


def ingest(store, dataset: str = DEDUPED, *, filt: DatasetFilter | None = None,
           limit: int = 5000, progress=None) -> dict:
    """Stream, convert and store. Returns a small report."""
    from ..models import Source

    say = progress or (lambda *_: None)
    source_id = "ode:" + hashlib.sha1(dataset.encode()).hexdigest()[:8]
    cards = list(stream_cards(dataset, filt=filt, limit=limit, progress=say))
    if not cards:
        return {"cards": 0, "added": 0, "note": "no rows matched the filter"}

    store.add_source(Source(
        source_id=source_id, path=dataset, title=f"OpenDebateEvidence ({dataset})",
        origin="opendebateevidence", license=f"research dataset; cite {CITATION}",
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        url=f"https://huggingface.co/datasets/{dataset}", card_count=len(cards),
    ))
    added = store.add_cards(cards)
    store.add_outline(build_outline(cards, source_id))
    with_read = sum(1 for c in cards if c.read_text)
    say(f"{added} new cards ({with_read}/{len(cards)} with read text)")
    return {"cards": len(cards), "added": added, "with_read_text": with_read,
            "source_id": source_id, "citation": CITATION}
