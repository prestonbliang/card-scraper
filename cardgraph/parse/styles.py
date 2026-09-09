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

# qn() re-parses a "w:xxx" prefix on every call and this module calls it tens of
# thousands of times per document. Resolve the tags we use once.
_U, _HL, _SHD, _B, _STRIKE = (qn("w:u"), qn("w:highlight"), qn("w:shd"),
                              qn("w:b"), qn("w:strike"))
_VAL, _FILL = qn("w:val"), qn("w:fill")
_PSTYLE, _RSTYLE = qn("w:pStyle"), qn("w:rStyle")
_STYLE, _STYLEID, _NAME, _BASEDON, _RPR = (qn("w:style"), qn("w:styleId"),
                                           qn("w:name"), qn("w:basedOn"),
                                           qn("w:rPr"))

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

    u = rPr.find(_U)
    if u is not None:
        val = u.get(_VAL)
        out["underline"] = val not in ("none", "0", "false")

    hl = rPr.find(_HL)
    if hl is not None:
        val = hl.get(_VAL)
        out["highlight"] = val not in (None, "none")

    # Word 2013+ also does highlighting via <w14:shd>/<w:shd> fill.
    shd = rPr.find(_SHD)
    if shd is not None:
        fill = shd.get(_FILL)
        if fill and fill.lower() not in ("auto", "ffffff", "000000"):
            out["highlight"] = True

    b = rPr.find(_B)
    if b is not None:
        out["bold"] = b.get(_VAL) not in ("0", "false")

    st = rPr.find(_STRIKE)
    if st is not None:
        out["strike"] = st.get(_VAL) not in ("0", "false")

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


# ---------------------------------------------------------------------------
# Style index
#
# Resolving a run's marks means walking its character style, that style's
# basedOn ancestors, and the paragraph style. Doing that through python-docx's
# `run.style` property re-resolves the same handful of styles once per run:
# profiling a real 5,500-run card file showed 11,137 style lookups costing 5.2
# of 5.6 seconds, 93% of parse time, because `Styles.default_for` rescans every
# style definition on every call. Across the published archive that is the
# difference between eight hours and four minutes.
#
# The style table is invariant for the life of a document, so it gets resolved
# exactly once into {style_id: marks} and every run after that is a dict hit.
# ---------------------------------------------------------------------------

_EMPTY: dict[str, bool | None] = {"underline": None, "highlight": None,
                                  "bold": None, "strike": None}


class StyleIndex:
    """Pre-resolved formatting for every style in one document."""

    # Two tables, and the split is load-bearing. A style *name* like "Style
    # Bold Underline" is treated as evidence of underlining when the theme is
    # unresolvable -- but only for CHARACTER styles, which is where the original
    # per-run resolution applied it. Applying it to paragraph styles too changes
    # what counts as read text on ~2% of real cards, which was caught by diffing
    # this optimization's output against the pre-optimization parser over 3,600
    # cards from the archive.
    __slots__ = ("_marks", "_marks_no_name", "_names")

    def __init__(self, styles_element) -> None:
        raw: dict[str, dict] = {}
        based: dict[str, str] = {}
        self._names: dict[str, str] = {}

        if styles_element is not None:
            for st in styles_element.findall(_STYLE):
                sid = st.get(_STYLEID)
                if not sid:
                    continue
                raw[sid] = _rpr_marks(st.find(_RPR))
                name_el = st.find(_NAME)
                if name_el is not None:
                    self._names[sid] = name_el.get(_VAL) or ""
                base = st.find(_BASEDON)
                if base is not None:
                    val = base.get(_VAL)
                    if val:
                        based[sid] = val

        # Flatten each basedOn chain once, nearest ancestor winning.
        self._marks: dict[str, dict] = {}
        self._marks_no_name: dict[str, dict] = {}
        for sid in raw:
            merged = dict(_EMPTY)
            plain = dict(_EMPTY)
            cur, hops = sid, 0
            while cur is not None and hops < 6:
                props = raw.get(cur, _EMPTY)
                for k, v in props.items():
                    if merged[k] is None and v is not None:
                        merged[k] = v
                    if plain[k] is None and v is not None:
                        plain[k] = v
                for k, v in _name_implies(self._names.get(cur, "")).items():
                    if merged[k] is None and v is not None:
                        merged[k] = v
                cur = based.get(cur)
                hops += 1
            self._marks[sid] = merged
            self._marks_no_name[sid] = plain

    def marks_for(self, style_id: str | None, use_names: bool = True) -> dict:
        """`use_names=False` for paragraph styles: explicit formatting only."""
        if not style_id:
            return _EMPTY
        table = self._marks if use_names else self._marks_no_name
        return table.get(style_id, _EMPTY)


_INDEX_CACHE: dict[int, StyleIndex] = {}


def style_index_for(part) -> StyleIndex | None:
    """Build (or reuse) the StyleIndex for a document part."""
    if part is None:
        return None
    try:
        element = part.styles.element
    except Exception:
        return None
    key = id(element)
    idx = _INDEX_CACHE.get(key)
    if idx is None:
        idx = StyleIndex(element)
        # Bounded: a long-running ingest opens thousands of documents and we do
        # not want their style tables living forever.
        if len(_INDEX_CACHE) > 64:
            _INDEX_CACHE.clear()
        _INDEX_CACHE[key] = idx
    return idx


def _direct_style_id(element) -> str | None:
    """Read w:rStyle / w:pStyle straight off the XML, skipping python-docx's
    style resolution entirely."""
    if element is None:
        return None
    pPr = getattr(element, "pPr", None)
    if pPr is not None:
        st = pPr.find(_PSTYLE)
        return st.get(_VAL) if st is not None else None
    rPr = getattr(element, "rPr", None)
    if rPr is not None:
        st = rPr.find(_RSTYLE)
        return st.get(_VAL) if st is not None else None
    return None


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


def resolve_marks(run, paragraph, index: "StyleIndex | None" = None) -> Marks:
    """Resolve a run's effective marks: direct rPr, then the character style
    chain (including style names), then the paragraph style chain.

    `index` is the document's pre-resolved style table. It is optional so this
    stays callable with two arguments, but callers in a loop should pass it --
    without it every run re-resolves the same styles through python-docx, which
    is 93% of parse time on a real card file.
    """
    resolved: dict[str, bool | None] = {"underline": None, "highlight": None,
                                        "bold": None, "strike": None}

    def absorb(candidate: dict[str, bool | None]) -> None:
        for k, v in candidate.items():
            if resolved[k] is None and v is not None:
                resolved[k] = v

    # 1. direct run properties always win
    absorb(_rpr_marks(run._element.rPr))

    if index is None:
        index = style_index_for(getattr(run, "part", None))

    if index is not None:
        # 2. character style (names may imply marks), then
        # 3. paragraph style (explicit formatting only) -- both pre-flattened
        absorb(index.marks_for(_direct_style_id(run._element)))
        absorb(index.marks_for(_direct_style_id(paragraph._element),
                               use_names=False))
    else:
        # Fallback: no style part available (synthetic runs in tests).
        try:
            for st in _style_chain(run.style):
                absorb(_rpr_marks(getattr(st.element, "rPr", None)))
                absorb(_name_implies(getattr(st, "name", "")))
        except (AttributeError, KeyError):
            pass
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
