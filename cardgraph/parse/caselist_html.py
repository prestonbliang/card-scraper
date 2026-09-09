"""Parse archived caselist wiki pages (XWiki HTML).

The published caselist archives are ~40,000 `.htm` files, not `.docx`, and they
encode something structurally different from a card file. Worth being precise
about the difference, because getting it wrong produces an index that looks fine
and is quietly lying.

A page is one team's aff or neg, and looks like this::

    <h4 class="title ..." name="title"><span>Minimata AC</span></h4>   <- entry
    <div name="entry">
      <p>Tournament: Woodward | Round: 1 | Opponent: NA | Judge: NA</p>
      <h4 class="wikigeneratedheader"><span>Consequences are what matter</span></h4>
      <p><strong>Harris 10 ~Sam, CEO Project Reason...~</strong><br/>
         Here is my starting point: all questions of value<br/>
         AND<br/>
         , there is no version of moral realism that...</p>
    </div>

**The disclosure convention matters.** Caselist rules require the first and last
few words of what you read, not the whole card -- the literal string ``AND`` on
its own line marks the omitted middle. So a caselist card has no full body and
no underlining. What you get is the tag, the cite, and the two ends of the read
text.

That is a real limitation, and the honest thing is to represent it rather than
paper over it:

* the disclosed fragments become ``read_text``, joined with the same elision
  marker the .docx parser uses for non-contiguous underlined spans -- the two
  conventions mean the same thing, so they render identically;
* ``body`` is set to the same text, because no more exists. It is not a
  truncation we chose;
* ``disclosed_only=True`` is set on every card, and the analysis layer reads it
  to suppress ``no_read_text`` / ``thin_read`` findings. Without that flag,
  ingesting the archive would generate one "this card has nothing underlined"
  finding per card, forever, all of them true and all of them useless.
"""

from __future__ import annotations

import html
import os
import re

from lxml import etree

from ..models import Card, NodeKind, OutlineNode, Side, infer_side
from .cite import looks_like_cite, parse_cite
from .docx_card import ELISION, ParseReport

# The caselist elision marker: a line containing only "AND" (occasionally
# "and", or padded) standing in for the omitted middle of a card.
_AND_LINE = re.compile(r"^\s*(AND|\.\.\.|…)\s*$", re.IGNORECASE)

_ROUND_LINE = re.compile(
    r"Tournament:\s*(?P<tournament>[^|]*)\|?\s*Round:\s*(?P<round>[^|]*)\|?"
    r"\s*Opponent:\s*(?P<opponent>[^|]*)\|?\s*Judge:\s*(?P<judge>[^|]*)",
    re.IGNORECASE)

# XWiki wraps cite text in tildes where the original used brackets.
_TILDE = re.compile(r"~+")

_MIN_BODY = 25


def _clean(s: str) -> str:
    s = html.unescape(s or "")
    s = s.replace(" ", " ")
    s = _TILDE.sub("", s)
    return re.sub(r"[ \t\r\f\v]+", " ", s).strip()


def _text_of(el) -> str:
    return _clean("".join(el.itertext()))


def _lines_of(el) -> list[str]:
    """Flatten an element to lines, treating <br/> as a line break.

    lxml gives <br/> as an empty element with a tail, so the text of a
    cite-then-body paragraph arrives as one run unless we split on it
    explicitly.
    """
    parts: list[str] = []
    if el.text:
        parts.append(el.text)
    for child in el:
        if child.tag in ("br",):
            parts.append("\n")
        else:
            parts.append("".join(child.itertext()))
        if child.tail:
            parts.append(child.tail)
    blob = html.unescape("".join(parts)).replace(" ", " ")
    blob = _TILDE.sub("", blob)
    return [re.sub(r"\s+", " ", ln).strip() for ln in blob.split("\n")]


def _strong_text(el) -> str:
    """The bolded lead of a paragraph, which is where the cite usually lives."""
    for tag in ("strong", "b"):
        found = el.find(f".//{tag}")
        if found is not None:
            t = _clean("".join(found.itertext()))
            if t:
                return t
    return ""


def _is_entry_header(el) -> bool:
    return (el.get("name") == "title"
            or "title" in (el.get("class") or "").split())


def _is_card_header(el) -> bool:
    cls = (el.get("class") or "").split()
    return "wikigeneratedheader" in cls and not _is_entry_header(el)


def _split_cite_and_body(lines: list[str], strong: str = "") -> tuple[str, str]:
    """Given the lines of a card paragraph, return (cite, read_text).

    Getting this wrong is how the author index fills with junk. Taking line 0 as
    the cite whenever a paragraph had more than one line -- which is what this
    did originally -- eats the first line of every *analytic* block ("I affirm.",
    "A framework centered around plans is best because...") and hands it to the
    cite parser, which dutifully extracts "I" and "A" as author names. Measured
    on the real archive that produced 58 cards attributed to pronouns, with
    "The" the 1st and "I" the 2nd most-cited "author" in the corpus.

    So the cite must be positively identified, in priority order:
      1. the paragraph's bolded lead, which is where XWiki puts it;
      2. a first line that independently looks like a cite.
    Otherwise the paragraph is all disclosed text and has no cite.
    """
    lines = [ln for ln in lines if ln]
    if not lines:
        return "", ""
    cite = ""
    rest = lines
    strong = (strong or "").strip()
    if strong and lines[0].startswith(strong[:40]):
        cite, rest = lines[0], lines[1:]
    elif looks_like_cite(lines[0]):
        cite, rest = lines[0], lines[1:]

    out: list[str] = []
    for ln in rest:
        if _AND_LINE.match(ln):
            if out:
                out.append(ELISION.strip())
            continue
        out.append(ln)
    text = " ".join(out)
    text = re.sub(rf"\s*{re.escape(ELISION.strip())}\s*", ELISION, text)
    return cite, text.strip()


def parse_caselist_html(path: str, source_id: str | None = None
                        ) -> tuple[list[Card], OutlineNode, ParseReport]:
    """Parse one archived caselist page."""
    source_id = source_id or os.path.basename(path)
    report = ParseReport(path=path)

    with open(path, "rb") as fh:
        raw = fh.read()
    parser = etree.HTMLParser(recover=True)
    tree = etree.fromstring(raw, parser)
    if tree is None:
        return [], OutlineNode(title=os.path.basename(path),
                               kind=NodeKind.POCKET, source_id=source_id), report

    # school / page from the file path: .../hsld13/bin/Apple+Valley/Boals+Aff.htm
    parts = [p.replace("+", " ") for p in path.replace("\\", "/").split("/")]
    page = os.path.splitext(parts[-1])[0] if parts else "page"
    school = parts[-2] if len(parts) >= 2 else ""
    root = OutlineNode(title=f"{school} {page}".strip() or page,
                       kind=NodeKind.POCKET, source_id=source_id)
    nodes: dict[tuple, OutlineNode] = {(): root}

    def node_for(p: list[str]) -> OutlineNode:
        key = tuple(p)
        if key in nodes:
            return nodes[key]
        parent = node_for(p[:-1])
        kind = [NodeKind.POCKET, NodeKind.HAT, NodeKind.BLOCK][min(len(p) - 1, 2)]
        n = OutlineNode(title=p[-1], kind=kind, path=list(p), source_id=source_id)
        parent.children.append(n)
        nodes[key] = n
        return n

    side = Side.AFF if re.search(r"\baff\b", page, re.I) else (
        Side.NEG if re.search(r"\bneg\b", page, re.I) else Side.UNKNOWN)

    cards: list[Card] = []
    ordinal = 0

    # Walk headers in document order; each h4 is either an entry title (a
    # position) or a card tag under the current entry.
    entry = ""
    round_ctx = ""
    for el in tree.iter():
        tag = str(el.tag).lower() if isinstance(el.tag, str) else ""
        if tag not in ("h1", "h2", "h3", "h4", "h5", "p"):
            continue
        report.paragraphs += 1

        if tag.startswith("h"):
            text = _text_of(el)
            if not text:
                continue
            if _is_entry_header(el):
                entry = text
                round_ctx = ""
                node_for([school or page, page, entry][-3:])
                continue
            if _is_card_header(el):
                # collect the paragraph(s) following this header
                lines: list[str] = []
                sib = el.getnext()
                while sib is not None and (
                        not isinstance(sib.tag, str)
                        or str(sib.tag).lower() not in ("h1", "h2", "h3", "h4", "h5")):
                    if isinstance(sib.tag, str) and str(sib.tag).lower() == "p":
                        lines.extend(_lines_of(sib))
                    sib = sib.getnext()

                strong = ""
                sib2 = el.getnext()
                while sib2 is not None and (
                        not isinstance(sib2.tag, str)
                        or str(sib2.tag).lower() not in ("h1", "h2", "h3", "h4", "h5")):
                    if isinstance(sib2.tag, str) and str(sib2.tag).lower() == "p":
                        strong = _strong_text(sib2)
                        if strong:
                            break
                    sib2 = sib2.getnext()
                cite_raw, read = _split_cite_and_body(lines, strong)
                if not read or len(read) < _MIN_BODY:
                    continue
                path_parts = [p for p in [school or page, page, entry] if p]
                card = Card(
                    tag=text[:400],
                    cite=parse_cite(cite_raw),
                    # No fuller body exists: the caselist discloses ends only.
                    body=read,
                    read_text=read,
                    source_id=source_id,
                    source_path=path,
                    ordinal=ordinal,
                    path=path_parts,
                    side=side if side is not Side.UNKNOWN
                    else infer_side(path_parts, text),
                    disclosed_only=True,
                    round_context=round_ctx,
                )
                ordinal += 1
                cards.append(card)
                node_for(path_parts).card_ids.append(card.card_id)
                if not card.cite.raw:
                    report.cards_without_cite += 1
                if not card.read_text:
                    report.cards_without_read_text += 1
            continue

        # <p> — pick up the round context line for the current entry
        text = _text_of(el)
        m = _ROUND_LINE.search(text)
        if m:
            round_ctx = " | ".join(
                f"{k.title()}: {v.strip()}" for k, v in m.groupdict().items()
                if v and v.strip() and v.strip().upper() != "NA")

    report.cards = len(cards)
    report.outline_nodes = sum(1 for _ in nodes)
    if cards and report.cards_without_cite / len(cards) > 0.6:
        report.warnings.append(
            "Most cards on this page have no parseable cite. Caselist cite "
            "formatting is team-authored and inconsistent; the tag and "
            "disclosed text are still usable.")
    return cards, root, report


def is_caselist_page(path: str) -> bool:
    """Cheap check so a directory of mixed HTML does not get misparsed."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(60_000).decode("utf-8", "replace")
    except OSError:
        return False
    return ('name="title"' in head and "wikigeneratedheader" in head) or \
           ("tblCites" in head)
