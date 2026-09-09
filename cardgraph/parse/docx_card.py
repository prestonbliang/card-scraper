"""Parse a Verbatim-style debate .docx into cards and an outline tree.

Design notes
------------
A card file is a flat paragraph stream that encodes a tree. The encoding is:

    Heading 1   pocket   "Politics DA"
    Heading 2   hat      "1NC"
    Heading 3   block    "AT: Link Turn"
    Heading 4   tag      "Data center buildout collapses ratepayer support"
    Cite        cite     "Ember 24 (...)"
    Normal      body     the quoted passage, with the read portion underlined

We walk the stream with a small state machine, maintaining a path stack for the
first three levels and accumulating a card at level four.

Real files break these rules constantly: missing cites, tags applied as bold
Normal text, bodies split across dozens of paragraphs, tables, blank headings.
The parser is written to degrade rather than throw -- a file that yields cards
with empty `read_text` is a warning, not a crash, and `parse_report` surfaces
those counts so you can tell whether an ingest actually worked.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import docx
from docx.document import Document as _Doc
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from ..models import Card, NodeKind, OutlineNode, Side, infer_side
from .cite import looks_like_cite, parse_cite
from .styles import is_cite_style, outline_level, resolve_marks

# A tag is a claim sentence: short-ish, no trailing citation apparatus.
_MAX_TAG_LEN = 400
_MIN_BODY_LEN = 60


@dataclass
class ParseReport:
    """What actually happened. Print this after every ingest."""

    path: str = ""
    paragraphs: int = 0
    cards: int = 0
    cards_without_cite: int = 0
    cards_without_read_text: int = 0
    outline_nodes: int = 0
    used_fallback: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def read_text_coverage(self) -> float:
        if not self.cards:
            return 0.0
        return round(1 - self.cards_without_read_text / self.cards, 4)

    def summary(self) -> str:
        return (
            f"{os.path.basename(self.path)}: {self.cards} cards, "
            f"read-text coverage {self.read_text_coverage:.0%}, "
            f"cite coverage {1 - self.cards_without_cite / max(self.cards, 1):.0%}"
            + (" [fallback parser]" if self.used_fallback else "")
        )


def _iter_block_items(parent):
    """Yield paragraphs and tables in document order (python-docx does not)."""
    if isinstance(parent, _Doc):
        parent_elm = parent.element.body
    else:
        parent_elm = parent._element
    for child in parent_elm.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


ELISION = " … "


def _stitch(spans: list[tuple[str, bool]]) -> str:
    """Join marked spans, inserting an elision marker wherever unmarked text
    was skipped.

    This matters more than it looks. Underlined text is non-contiguous by
    design -- the debater cuts *through* the paragraph -- so naive
    concatenation welds the end of one clause onto the start of the next and
    produces a sentence the author never wrote ("...ratepayer baseresidential
    rate impacts..."). That corrupts both display and embeddings. The marker
    keeps the seams visible.
    """
    out: list[str] = []
    skipped = False
    for text, is_marked in spans:
        if is_marked:
            if out and skipped:
                out.append(ELISION)
            out.append(text)
            skipped = False
        elif text.strip():
            skipped = True
    return _clean("".join(out))


def _paragraph_marks(paragraph) -> tuple[str, str, str]:
    """Return (full_text, read_text, emphasis_text) for one paragraph."""
    full: list[str] = []
    read_spans: list[tuple[str, bool]] = []
    emph_spans: list[tuple[str, bool]] = []
    for run in paragraph.runs:
        txt = run.text
        if not txt:
            continue
        full.append(txt)
        marks = resolve_marks(run, paragraph)
        read_spans.append((txt, marks.is_read))
        emph_spans.append((txt, marks.is_emphasis))
    return "".join(full), _stitch(read_spans), _stitch(emph_spans)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _join_read(chunks: list[str]) -> str:
    """Join read fragments. Underlined spans are non-contiguous by nature, so
    we join with a space and collapse -- this reconstructs roughly what the
    debater actually says."""
    return _clean(" ".join(c for c in chunks if c.strip()))


class _CardAccumulator:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.tag: str = ""
        self.cite_raw: str = ""
        self.body_parts: list[str] = []
        self.read_parts: list[str] = []
        self.emph_parts: list[str] = []
        self.path: list[str] = []

    @property
    def active(self) -> bool:
        return bool(self.tag)

    def build(self, source_id: str, source_path: str, ordinal: int) -> Card | None:
        body = _clean(" ".join(self.body_parts))
        if not self.tag:
            return None
        if len(body) < _MIN_BODY_LEN and not self.cite_raw:
            return None
        return Card(
            tag=_clean(self.tag),
            cite=parse_cite(self.cite_raw),
            body=body,
            read_text=_join_read(self.read_parts),
            emphasis_text=_join_read(self.emph_parts),
            source_id=source_id,
            source_path=source_path,
            ordinal=ordinal,
            path=list(self.path),
            side=infer_side(self.path, self.tag),
        )


def parse_docx(path: str, source_id: str | None = None) -> tuple[list[Card], OutlineNode, ParseReport]:
    """Parse one .docx. Returns (cards, outline_root, report)."""
    source_id = source_id or os.path.basename(path)
    report = ParseReport(path=path)
    doc = docx.Document(path)

    root = OutlineNode(title=os.path.basename(path), kind=NodeKind.POCKET,
                       path=[], source_id=source_id)
    node_by_path: dict[tuple[str, ...], OutlineNode] = {(): root}

    def node_for(path: list[str]) -> OutlineNode:
        key = tuple(path)
        if key in node_by_path:
            return node_by_path[key]
        parent = node_for(path[:-1])
        kind = [NodeKind.POCKET, NodeKind.HAT, NodeKind.BLOCK][min(len(path) - 1, 2)]
        node = OutlineNode(title=path[-1], kind=kind, path=list(path), source_id=source_id)
        parent.children.append(node)
        node_by_path[key] = node
        return node

    cards: list[Card] = []
    acc = _CardAccumulator()
    stack: list[str] = []
    expecting_cite = False
    ordinal = 0

    def flush() -> None:
        nonlocal ordinal
        if not acc.active:
            acc.reset()
            return
        card = acc.build(source_id, path, ordinal)
        acc.reset()
        if card is None:
            return
        ordinal += 1
        cards.append(card)
        target = node_for(card.path) if card.path else root
        target.card_ids.append(card.card_id)
        if not card.cite.raw:
            report.cards_without_cite += 1
        if not card.read_text:
            report.cards_without_read_text += 1

    for item in _iter_block_items(doc):
        if isinstance(item, Table):
            # Cards do appear inside tables (blocks pasted from other files).
            for row in item.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        full, read, emph = _paragraph_marks(p)
                        if full.strip():
                            acc.body_parts.append(full)
                            acc.read_parts.append(read)
                            acc.emph_parts.append(emph)
            continue

        p = item
        report.paragraphs += 1
        lvl = outline_level(p)
        full, read, emph = _paragraph_marks(p)
        text = _clean(full)

        if lvl is not None and lvl <= 3:
            flush()
            expecting_cite = False
            if text:
                stack = stack[: lvl - 1] + [text]
                node_for(stack)
            continue

        if lvl == 4:
            flush()
            if text:
                acc.tag = text[:_MAX_TAG_LEN]
                acc.path = list(stack)
                expecting_cite = True
            continue

        if not text:
            continue

        if acc.active and expecting_cite and (is_cite_style(p) or looks_like_cite(text)):
            acc.cite_raw = text
            expecting_cite = False
            continue

        if acc.active:
            expecting_cite = False
            acc.body_parts.append(full)
            acc.read_parts.append(read)
            acc.emph_parts.append(emph)

    flush()

    if not cards:
        report.used_fallback = True
        cards, root = _fallback_parse(doc, source_id, path, report)

    report.cards = len(cards)
    report.outline_nodes = _count_nodes(root)
    if report.cards and report.read_text_coverage < 0.5:
        report.warnings.append(
            "Under half of cards have read text -- this file probably marks the "
            "read portion in a way the style resolver missed. Inspect one card "
            "before trusting search results from this source."
        )
    return cards, root, report


def _count_nodes(node: OutlineNode) -> int:
    return 1 + sum(_count_nodes(c) for c in node.children)


def _fallback_parse(doc, source_id: str, path: str, report: ParseReport):
    """For files with no heading styles at all.

    Anchor on cite lines: a paragraph that looks like a cite, preceded by a
    short paragraph (the tag) and followed by prose (the body).
    """
    paras = [p for p in doc.paragraphs]
    texts = [_clean(p.text) for p in paras]
    cards: list[Card] = []
    root = OutlineNode(title=os.path.basename(path), kind=NodeKind.POCKET,
                       path=[], source_id=source_id)

    i, ordinal = 0, 0
    while i < len(texts):
        if texts[i] and looks_like_cite(texts[i]) and i > 0:
            tag = ""
            for j in range(i - 1, max(-1, i - 4), -1):
                if texts[j] and len(texts[j]) <= _MAX_TAG_LEN and not looks_like_cite(texts[j]):
                    tag = texts[j]
                    break
            body_parts, read_parts, emph_parts = [], [], []
            k = i + 1
            while k < len(texts) and not looks_like_cite(texts[k]):
                full, read, emph = _paragraph_marks(paras[k])
                if full.strip():
                    body_parts.append(full)
                    read_parts.append(read)
                    emph_parts.append(emph)
                k += 1
            body = _clean(" ".join(body_parts))
            if tag and len(body) >= _MIN_BODY_LEN:
                card = Card(
                    tag=tag, cite=parse_cite(texts[i]), body=body,
                    read_text=_join_read(read_parts),
                    emphasis_text=_join_read(emph_parts),
                    source_id=source_id, source_path=path, ordinal=ordinal,
                    path=[], side=Side.UNKNOWN,
                )
                ordinal += 1
                cards.append(card)
                root.card_ids.append(card.card_id)
                if not card.read_text:
                    report.cards_without_read_text += 1
            i = k
            continue
        i += 1
    return cards, root


def parse_any(path: str, source_id: str | None = None):
    """Dispatch on extension. .docx today; .html/.txt hooks for wiki dumps."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".docx", ".docm"):
        return parse_docx(path, source_id)
    raise ValueError(f"no parser for {ext} (path={path})")
