"""Resolving *what a run actually looks like* in a Verbatim .docx.

This is fiddly and it is the whole ballgame, so it gets its own module.

python-docx exposes `run.font.underline` etc., but those return ``None`` when
the property is inherited rather than set directly on the run -- which is the
common case, because Verbatim applies formatting through character styles
("Style Emphasis", "Style Underline") and through the paragraph style. A parser
that trusts the direct properties silently reads ~0% of a real card file as
"highlighted", which is a failure mode that looks like success: you get cards
out, they just have empty read_text.

So we resolve in three passes: direct run properties -> the run's character
style (walking its basedOn chain) -> the paragraph style. First hit wins.
"""

from __future__ import annotations

from dataclasses import dataclass

from docx.oxml.ns import qn

# Verbatim's default style names, plus the common team-local variants we have
# seen in the wild. Order matters: longer/more specific first.
TAG_STYLE_HINTS = ("heading 4", "style tag", "tag", "cardtag", "card tag")
CITE_STYLE_HINTS = ("cite", "style cite", "citation", "card cite")
BLOCK_STYLE_HINTS = ("heading 3", "style block", "block")
HAT_STYLE_HINTS = ("heading 2", "style hat", "hat")
POCKET_STYLE_HINTS = ("heading 1", "style pocket", "pocket")
BODY_STYLE_HINTS = ("normal", "card", "style card", "body text", "cardtext")

EMPHASIS_STYLE_HINTS = ("emphasis", "style emphasis", "emphatic", "bold underline")
UNDERLINE_STYLE_HINTS = ("underline", "style underline", "read", "style read")


@dataclass(frozen=True)
class Marks:
    """The marks a run carries, after inheritance is resolved."""

    underline: bool = False
    highlight: bool = False
    bold: bool = False
    strike: bool = False

    @property
    def is_read(self) -> bool:
        """Spoken aloud in round. Underline is the near-universal convention;
        some teams highlight instead of (or as well as) underlining."""
        return (self.underline or self.highlight) and not self.strike

    @property
    def is_emphasis(self) -> bool:
        """Doubly marked -- the words the debater leans on. Usually bold on top
        of underline, or highlight on top of underline."""
        return self.is_read and (self.bold or (self.underline and self.highlight))


def _rpr_marks(rPr) -> dict[str, bool | None]:
    """Read explicit formatting off an <w:rPr> element."""
    out: dict[str, bool | None] = {"underline": None, "highlight": None,
                                   "bold": None, "strike": None}
    if rPr is None:
        return out

    u = rPr.find(qn("w:u"))
    if u is not None:
        val = u.get(qn("w:val"))
        out["underline"] = val not in ("none", "0", "false")

    hl = rPr.find(qn("w:highlight"))
    if hl is not None:
        val = hl.get(qn("w:val"))
        out["highlight"] = val not in (None, "none")

    # Word 2013+ also does highlighting via <w14:shd>/<w:shd> fill.
    shd = rPr.find(qn("w:shd"))
    if shd is not None:
        fill = shd.get(qn("w:fill"))
        if fill and fill.lower() not in ("auto", "ffffff", "000000"):
            out["highlight"] = True

    b = rPr.find(qn("w:b"))
    if b is not None:
        out["bold"] = b.get(qn("w:val")) not in ("0", "false")

    st = rPr.find(qn("w:strike"))
    if st is not None:
        out["strike"] = st.get(qn("w:val")) not in ("0", "false")

    return out


def _style_chain(style, limit: int = 6):
    """Yield a style and its basedOn ancestors."""
    seen = set()
    cur = style
    while cur is not None and len(seen) < limit:
        sid = getattr(cur, "style_id", None) or id(cur)
        if sid in seen:
            break
        seen.add(sid)
        yield cur
        cur = getattr(cur, "base_style", None)


def _name_implies(name: str) -> dict[str, bool | None]:
    """Some teams encode 'this is read' purely in the style *name*, with the
    actual formatting living in a theme we may not be able to resolve. Trust
    the name as a last resort."""
    n = (name or "").strip().lower()
    out: dict[str, bool | None] = {"underline": None, "highlight": None,
                                   "bold": None, "strike": None}
    if any(h in n for h in EMPHASIS_STYLE_HINTS):
        out["underline"] = True
        out["bold"] = True
    elif any(h in n for h in UNDERLINE_STYLE_HINTS):
        out["underline"] = True
    return out


def resolve_marks(run, paragraph) -> Marks:
    """Resolve a run's effective marks: direct rPr, then character style chain
    (including style names), then paragraph style chain."""
    resolved: dict[str, bool | None] = {"underline": None, "highlight": None,
                                        "bold": None, "strike": None}

    def absorb(candidate: dict[str, bool | None]) -> None:
        for k, v in candidate.items():
            if resolved[k] is None and v is not None:
                resolved[k] = v

    # 1. direct
    absorb(_rpr_marks(run._element.rPr))

    # 2. run's character style
    try:
        for st in _style_chain(run.style):
            absorb(_rpr_marks(getattr(st.element, "rPr", None)))
            absorb(_name_implies(getattr(st, "name", "")))
    except (AttributeError, KeyError):
        pass

    # 3. paragraph style
    try:
        for st in _style_chain(paragraph.style):
            absorb(_rpr_marks(getattr(st.element, "rPr", None)))
    except (AttributeError, KeyError):
        pass

    return Marks(
        underline=bool(resolved["underline"]),
        highlight=bool(resolved["highlight"]),
        bold=bool(resolved["bold"]),
        strike=bool(resolved["strike"]),
    )


def style_name(paragraph) -> str:
    try:
        return (paragraph.style.name or "").strip().lower()
    except (AttributeError, KeyError):
        return ""


def outline_level(paragraph) -> int | None:
    """Return 1-4 for pocket/hat/block/tag, or None if not a heading.

    Prefers the paragraph's `outlineLvl` (which survives style renaming) and
    falls back to style-name matching.
    """
    pPr = paragraph._element.pPr
    if pPr is not None:
        lvl = pPr.find(qn("w:outlineLvl"))
        if lvl is not None:
            try:
                v = int(lvl.get(qn("w:val")))
                if 0 <= v <= 3:
                    return v + 1
            except (TypeError, ValueError):
                pass

    name = style_name(paragraph)
    if not name:
        return None
    for lvl, hints in enumerate(
        (POCKET_STYLE_HINTS, HAT_STYLE_HINTS, BLOCK_STYLE_HINTS, TAG_STYLE_HINTS), start=1
    ):
        if name in hints:
            return lvl
    # "heading 1".."heading 4"
    if name.startswith("heading "):
        try:
            v = int(name.split()[-1])
            if 1 <= v <= 4:
                return v
        except ValueError:
            pass
    return None


def is_cite_style(paragraph) -> bool:
    return style_name(paragraph) in CITE_STYLE_HINTS
