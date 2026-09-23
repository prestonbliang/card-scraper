"""Research packet export.

Turns a set of pinned or searched cards into a single shareable packet:
a Markdown brief grouped by side, a BibTeX bibliography, a CSV spreadsheet,
and a manifest that records where every card came from and how healthy the
underlying source is.

Retrieval metadata is not a factual verdict. The manifest reports
traceability (citation present, source recorded, read text preserved) so a
reader can audit the evidence chain themselves; it deliberately says nothing
about whether the claim inside the card is true.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone

APP_VERSION = "0.2.0"
SIDES = ("aff", "neg", "unknown")
SIDE_LABELS = {"aff": "Affirmative", "neg": "Negative", "unknown": "Unclassified"}


class PacketError(ValueError):
    """Raised when a packet request is invalid or produces nothing usable."""


@dataclass(frozen=True)
class PacketCard:
    """The subset of a card row a packet needs, already validated."""

    card_id: str
    tag: str
    cite_raw: str
    cite_author: str | None
    cite_year: int | None
    cite_pub: str | None
    cite_url: str | None
    block: str | None
    pocket: str | None
    side: str
    read_text: str
    body: str
    source_id: str
    source_title: str
    source_origin: str
    source_url: str | None
    source_fetched_at: str | None
    source_refresh_failed_at: str | None
    source_refresh_error: str | None
    read_ratio: float
    disclosed_only: bool
    score: float | None = None
    # Precomputed freshness from search hits; None means derive from the row.
    _freshness: str | None = None
    _age_days: int | None = None

    @property
    def side_label(self) -> str:
        return SIDE_LABELS.get(self.side, SIDE_LABELS["unknown"])

    @property
    def evidence_status(self) -> str:
        """Traceability only: citation, recorded source, and readable text."""
        has_citation = bool(self.cite_raw.strip())
        has_source = bool(self.source_id.strip())
        has_read = bool(self.read_text.strip()) or self.disclosed_only
        if has_citation and has_source and has_read:
            return "traceable"
        if has_citation and has_source:
            return "attributed"
        return "needs-review"

    @property
    def source_freshness(self) -> str:
        if self._freshness is not None:
            return self._freshness
        if self.source_refresh_failed_at:
            return "refresh-failed"
        if not self.source_fetched_at:
            return "unknown"
        try:
            stamp = datetime.fromisoformat(
                self.source_fetched_at.replace("Z", "+00:00"))
            age = max(0, (datetime.now(timezone.utc) - stamp).days)
        except (TypeError, ValueError):
            return "unknown"
        threshold = 90 if self.source_origin in {"opendebateevidence", "dataset"} else 30
        return "stale" if age >= threshold else "fresh"

    @property
    def source_age_days(self) -> int | None:
        if self._age_days is not None:
            return self._age_days
        if not self.source_fetched_at:
            return None
        try:
            stamp = datetime.fromisoformat(
                self.source_fetched_at.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
        return max(0, (datetime.now(timezone.utc) - stamp).days)


CARD_QUERY = """
SELECT c.card_id, c.tag, c.cite_raw, c.cite_author, c.cite_year, c.cite_pub,
       c.cite_url, c.block, c.pocket, c.side, c.read_text, c.body,
       c.read_ratio, c.disclosed_only,
       s.source_id AS src_id, s.title AS src_title, s.origin AS src_origin,
       s.url AS src_url, s.fetched_at AS src_fetched_at,
       s.last_refresh_failed_at AS src_refresh_failed_at,
       s.last_refresh_error AS src_refresh_error
FROM cards c LEFT JOIN sources s ON s.source_id = c.source_id
WHERE c.card_id IN ({placeholders})
"""


def collect_cards(store, *, card_ids: list[str] | None = None,
                  query: str | None = None, k: int = 25,
                  smart: bool = False, side: str | None = None,
                  author: str | None = None, year_min: int | None = None,
                  year_max: int | None = None, block: str | None = None,
                  min_read_ratio: float | None = None,
                  source: str | None = None,
                  mode: str = "balanced") -> list[PacketCard]:
    """Resolve a packet's cards either by explicit ids or by search."""
    if card_ids and query:
        raise PacketError("pass card ids or a query, not both")
    if card_ids:
        cleaned = [cid.strip() for cid in card_ids if cid and cid.strip()]
        if not cleaned:
            raise PacketError("no usable card ids were provided")
        if len(cleaned) > 500:
            raise PacketError("a packet supports at most 500 cards")
        placeholders = ",".join("?" * len(cleaned))
        rows = store.conn.execute(
            CARD_QUERY.format(placeholders=placeholders), cleaned).fetchall()
        found = {r["card_id"]: dict(r) for r in rows}
        missing = [cid for cid in cleaned if cid not in found]
        if missing:
            raise PacketError(
                "card ids not in the index: " + ", ".join(missing[:8])
                + ("…" if len(missing) > 8 else ""))
        ordered = [found[cid] for cid in dict.fromkeys(cleaned)]
        return [_packet_card(row) for row in ordered]

    if not query or not query.strip():
        raise PacketError("pass card ids or a search query")
    from .index.search import SearchEngine  # local import avoids a cycle
    engine = SearchEngine(store)
    engine.build()
    hits = engine.search(query.strip(), k=max(1, min(k, 500)),
                         side=side, author=author, year_min=year_min,
                         year_max=year_max, block=block,
                         min_read_ratio=min_read_ratio, source=source,
                         mode=mode)
    if not hits:
        raise PacketError("the search returned no cards to packet")
    return [_packet_card_from_hit(hit) for hit in hits]


def _packet_card(row: dict) -> PacketCard:
    return PacketCard(
        card_id=row["card_id"], tag=row["tag"] or "Untitled",
        cite_raw=row["cite_raw"] or "",
        cite_author=row.get("cite_author"), cite_year=row.get("cite_year"),
        cite_pub=row.get("cite_pub"), cite_url=row.get("cite_url"),
        block=row.get("block"), pocket=row.get("pocket"),
        side=row["side"] or "unknown", read_text=row["read_text"] or "",
        body=row["body"] or "",
        source_id=row.get("src_id") or "",
        source_title=row.get("src_title") or "",
        source_origin=row.get("src_origin") or "",
        source_url=row.get("src_url"),
        source_fetched_at=row.get("src_fetched_at"),
        source_refresh_failed_at=row.get("src_refresh_failed_at"),
        source_refresh_error=row.get("src_refresh_error"),
        read_ratio=float(row.get("read_ratio") or 0.0),
        disclosed_only=bool(row.get("disclosed_only")),
    )


def _packet_card_from_hit(hit) -> PacketCard:
    return PacketCard(
        card_id=hit.card_id, tag=hit.tag or "Untitled",
        cite_raw=hit.cite_raw or "", cite_author=hit.cite_author,
        cite_year=hit.cite_year, cite_pub=None, cite_url=None,
        block=hit.block, pocket=None, side=hit.side or "unknown",
        read_text=hit.read_text or "", body="",
        source_id=hit.source_id or "", source_title=hit.source_title or "",
        source_origin=hit.source_origin or "", source_url=hit.source_url,
        source_fetched_at=None, source_refresh_failed_at=None,
        source_refresh_error=hit.source_refresh_error,
        read_ratio=float(hit.read_ratio or 0.0),
        disclosed_only=bool(hit.disclosed_only), score=hit.score,
        _freshness=hit.source_freshness, _age_days=hit.source_age_days,
    )


def _clean_text(value: str, limit: int = 6000) -> str:
    return (value or "").strip()[:limit]


def markdown_brief(cards: list[PacketCard], title: str = "Card Scraper research packet") -> str:
    """A readable brief grouped by side, ordered as the cards were given."""
    out: list[str] = [
        f"# {title}", "",
        f"{len(cards)} card{'s' if len(cards) != 1 else ''} · generated "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · "
        "retrieval metadata is not a factual verdict.", "",
    ]
    for side in SIDES:
        group = [c for c in cards if c.side == side]
        if not group:
            continue
        out += [f"## {SIDE_LABELS[side]}", ""]
        for index, card in enumerate(group, 1):
            block = f" — {card.block}" if card.block else ""
            out.append(f"### {index}. {card.tag}{block}")
            out.append("")
            if card.cite_raw:
                out.append(f"**Citation:** {card.cite_raw}")
            if card.source_title or card.source_url:
                where = card.source_title or card.source_id
                if card.source_url:
                    where += f" — {card.source_url}"
                out.append(f"**Source:** {where}")
            flags: list[str] = [card.evidence_status, f"source: {card.source_freshness}"]
            out.append(f"**Evidence health:** {' · '.join(flags)}")
            out.append("")
            read = _clean_text(card.read_text)
            if read:
                out.append(f"> {read}")
            elif card.disclosed_only:
                out.append("> (source disclosed only the ends of this card)")
            else:
                out.append("> (no read text was parsed for this card)")
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def _bibtex_key(card: PacketCard, taken: set[str]) -> str:
    author = re.sub(r"[^A-Za-z]", "", (card.cite_author or "unknown").split()[-1] or "unknown").lower() or "unknown"
    year = str(card.cite_year) if card.cite_year else "nodate"
    base = f"{author}{year}"
    key = base
    suffix = 0
    while key in taken:
        suffix += 1
        key = f"{base}{chr(ord('a') + suffix - 1) if suffix <= 26 else suffix}"
    taken.add(key)
    return key


def bibtex_bibliography(cards: list[PacketCard]) -> str:
    """One entry per card. Cards without an author/year still get a key so a
    reader can spot missing metadata instead of silently losing the card."""
    taken: set[str] = set()
    out: list[str] = ["% Bibliography generated by Card Scraper", ""]
    for card in cards:
        key = _bibtex_key(card, taken)
        fields = [
            f"  title = {{{card.tag}}}",
            f"  howpublished = {{{card.source_title or card.source_id or 'unrecorded source'}}}",
        ]
        if card.cite_raw.strip():
            fields.insert(1, f"  note = {{{card.cite_raw}}}")
        if card.cite_author:
            fields.insert(0, f"  author = {{{card.cite_author}}}")
        if card.cite_year:
            fields.insert(1, f"  year = {{{card.cite_year}}}")
        if card.cite_pub:
            fields.append(f"  publisher = {{{card.cite_pub}}}")
        url = card.cite_url or card.source_url
        if url:
            fields.append(f"  url = {{{url}}}")
        out.append(f"@misc{{{key},")
        out.append(",\n".join(fields))
        out += ["}", ""]
    return "\n".join(out).rstrip() + "\n"


CSV_COLUMNS = [
    "card_id", "side", "tag", "block", "cite_raw", "cite_author", "cite_year",
    "evidence_status", "source_id", "source_title", "source_url",
    "source_freshness", "read_ratio", "read_text",
]


def csv_sheet(cards: list[PacketCard]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for card in cards:
        writer.writerow([
            card.card_id, card.side, card.tag, card.block or "",
            card.cite_raw, card.cite_author or "", card.cite_year or "",
            card.evidence_status, card.source_id, card.source_title,
            card.source_url or "", card.source_freshness,
            card.read_ratio, _clean_text(card.read_text, 2000),
        ])
    return buffer.getvalue()


def provenance_manifest(cards: list[PacketCard]) -> dict:
    sources: dict[str, dict] = {}
    for card in cards:
        entry = sources.setdefault(card.source_id or "unrecorded", {
            "source_id": card.source_id or None,
            "title": card.source_title or None,
            "origin": card.source_origin or None,
            "url": card.source_url,
            "freshness": card.source_freshness,
            "refresh_failed": bool(card.source_refresh_failed_at),
            "refresh_error": card.source_refresh_error,
            "cards": [],
        })
        entry["cards"].append(card.card_id)
    statuses = [c.evidence_status for c in cards]
    return {
        "packet_version": 1,
        "generator": f"Card Scraper {APP_VERSION}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "card_count": len(cards),
        "evidence_status_counts": {
            name: statuses.count(name)
            for name in ("traceable", "attributed", "needs-review")
            if statuses.count(name)
        },
        "cards": [c.card_id for c in cards],
        "sources": sources,
        "disclaimer": (
            "Retrieval metadata describes traceability, not truth. Verify "
            "citations against the original sources and respect each "
            "source's license before redistributing."
        ),
    }


def build_packet(store, *, card_ids: list[str] | None = None,
                 query: str | None = None, **kw) -> tuple[str, bytes, dict]:
    """Assemble the packet ZIP. Returns (suggested_filename, zip_bytes, manifest)."""
    cards = collect_cards(store, card_ids=card_ids, query=query, **kw)
    manifest = provenance_manifest(cards)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.md", markdown_brief(cards))
        z.writestr("bibliography.bib", bibtex_bibliography(cards))
        z.writestr("cards.csv", csv_sheet(cards))
        z.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()[:12]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"card-scraper-packet-{stamp}-{digest}.zip", payload, manifest


def write_packet(store, path: str, **kw) -> tuple[str, dict]:
    """Write a packet to a file path atomically; used by the CLI."""
    filename, payload, manifest = build_packet(store, **kw)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    fd, staged = tempfile.mkstemp(prefix="card-scraper-packet-", suffix=".zip")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, os.path.abspath(path))
    finally:
        if os.path.exists(staged):
            os.remove(staged)
    return filename, manifest
