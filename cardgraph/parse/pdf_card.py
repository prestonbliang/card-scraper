"""Conservative PDF ingestion for public evidence releases.

PDFs do not preserve a reliable equivalent of Word underlining, so this parser
never claims to know what was read aloud. It creates one searchable evidence
unit per non-empty page, preserves the extracted page text as body, and marks
read_text empty. That is intentionally less clever than inventing card
boundaries or read marks from a PDF's visual layout.
"""

from __future__ import annotations

import os

from ..models import Card, NodeKind, OutlineNode, infer_side
from .cite import looks_like_cite, parse_cite
from .docx_card import ParseReport

_MIN_PAGE_TEXT = 40
_MAX_TAG_LEN = 400


def parse_pdf(path: str, source_id: str | None = None):
    """Return page-level cards, an outline root, and a transparent report."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF support requires pypdf; install the project dependencies"
        ) from exc

    source_id = source_id or os.path.basename(path)
    report = ParseReport(path=path)
    root = OutlineNode(title=os.path.basename(path), kind=NodeKind.POCKET,
                       path=[], source_id=source_id)
    cards: list[Card] = []
    reader = PdfReader(path)

    for page_no, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").replace("\x00", "")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        body = " ".join(lines).strip()
        if len(body) < _MIN_PAGE_TEXT:
            continue

        if lines and looks_like_cite(lines[0]):
            tag = f"PDF page {page_no}"
            cite_raw = lines[0]
            body = " ".join(lines[1:]).strip() or lines[0]
        else:
            tag = lines[0][:_MAX_TAG_LEN] if lines else f"PDF page {page_no}"
            cite_raw = next((line for line in lines[1:4] if looks_like_cite(line)), "")
            if cite_raw:
                body = " ".join(line for line in lines[1:] if line != cite_raw).strip()

        if len(body) < _MIN_PAGE_TEXT:
            body = " ".join(lines)
        page_path = [f"Page {page_no}"]
        card = Card(
            tag=tag,
            cite=parse_cite(cite_raw),
            body=body,
            read_text="",
            source_id=source_id,
            source_path=path,
            ordinal=len(cards),
            path=page_path,
            side=infer_side([], tag),
            disclosed_only=True,
        )
        cards.append(card)
        page_node = OutlineNode(
            title=page_path[0], kind=NodeKind.BLOCK, path=page_path,
            source_id=source_id, card_ids=[card.card_id],
        )
        root.children.append(page_node)

    report.cards = len(cards)
    report.cards_without_read_text = len(cards)
    report.paragraphs = len(reader.pages)
    report.outline_nodes = 1 + len(cards)
    if cards:
        report.warnings.append(
            "PDF text imported at page level; PDF layout does not reliably "
            "preserve card boundaries or read marking, so read text is left "
            "empty. Search uses the extracted page body and results should be "
            "verified against the original PDF."
        )
    return cards, root, report
