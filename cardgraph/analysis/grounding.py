"""Grounding: making the model cite real cards, and checking that it did.

This is the module that decides whether the analysis layer is worth anything.
A model asked to critique a debate position will happily produce fluent,
plausible, completely invented criticism — citing a card that does not exist,
or quoting a sentence the card does not contain. That output is worse than no
output, because it reads exactly like the real thing.

Two mechanisms:

**Citation by index, not by id.** Cards are presented to the model as ``[C1]``,
``[C2]`` … rather than as 16-hex ids. Models copy short integer labels reliably
and long hex strings unreliably, and a mis-copied hex id is indistinguishable
from a fabricated one. Indices are mapped back to real ids afterwards, and an
index outside the presented range is a hard error we can detect exactly.

**Quote verification.** Every quote the model attributes to a card is checked as
a normalized substring of that card's actual text. Near-misses are tolerated
(whitespace, ellipsis, smart quotes); inventions are not.

A finding that survives with no valid citation is not deleted silently — it is
marked ``grounded=False``, demoted, and annotated, so you can see what the model
claimed and that it could not back it up. Silently dropping it would hide a
prompt regression; keeping it unmarked would launder a hallucination.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .schema import Finding, Severity

CITE_RE = re.compile(r"\[?\bC(\d{1,4})\b\]?")


def normalize(text: str) -> str:
    """Aggressive normalization for substring comparison: NFKD, lowercase,
    collapse whitespace, strip punctuation that varies between the .docx and
    the model's rendering of it (smart quotes, dashes, ellipses)."""
    t = unicodedata.normalize("NFKD", text or "")
    t = t.replace("’", "'").replace("‘", "'")
    t = t.replace("“", '"').replace("”", '"')
    t = re.sub(r"[–—…]", " ", t)
    t = t.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class CardRef:
    """A card as presented to the model."""

    index: int
    card_id: str
    tag: str
    cite: str
    read_text: str
    body: str = ""

    @property
    def label(self) -> str:
        return f"C{self.index}"

    def render(self, max_read: int = 900) -> str:
        read = self.read_text or "(nothing underlined in this card)"
        if len(read) > max_read:
            read = read[:max_read] + " …[truncated]"
        return (f"[{self.label}]\n"
                f"TAG: {self.tag}\n"
                f"CITE: {self.cite or '(no cite)'}\n"
                f"READ: {read}")

    def haystack(self) -> str:
        return normalize(" ".join([self.tag, self.read_text, self.body, self.cite]))


class CardContext:
    """The set of cards a model call may cite, plus the verifier for what it
    cited. One instance per prompt."""

    def __init__(self, cards: list[dict], include_body: bool = False):
        self.refs: list[CardRef] = []
        self.by_index: dict[int, CardRef] = {}
        self.by_id: dict[str, CardRef] = {}
        for i, c in enumerate(cards, start=1):
            ref = CardRef(
                index=i,
                card_id=c["card_id"],
                tag=c.get("tag") or "",
                cite=c.get("cite_raw") or "",
                read_text=c.get("read_text") or "",
                body=(c.get("body") or "") if include_body else "",
            )
            self.refs.append(ref)
            self.by_index[i] = ref
            self.by_id[ref.card_id] = ref

    def __len__(self) -> int:
        return len(self.refs)

    def render(self, max_read: int = 900) -> str:
        return "\n\n".join(r.render(max_read) for r in self.refs)

    # -- resolution --------------------------------------------------------

    def resolve(self, tokens: list[str] | None) -> tuple[list[str], list[str]]:
        """Map model-emitted labels to real card ids.

        Returns (card_ids, problems). Accepts "C3", "[C3]", "3", and a real
        card_id if the model somehow produced one.
        """
        ids: list[str] = []
        problems: list[str] = []
        seen: set[str] = set()
        for tok in tokens or []:
            tok = str(tok).strip()
            if not tok:
                continue
            if tok in self.by_id:
                if tok not in seen:
                    seen.add(tok)
                    ids.append(tok)
                continue
            m = CITE_RE.fullmatch(tok) or CITE_RE.search(tok)
            if not m:
                problems.append(f"uninterpretable citation {tok!r}")
                continue
            idx = int(m.group(1))
            ref = self.by_index.get(idx)
            if ref is None:
                problems.append(
                    f"cited C{idx} but only C1..C{len(self.refs)} were provided")
                continue
            if ref.card_id not in seen:
                seen.add(ref.card_id)
                ids.append(ref.card_id)
        return ids, problems

    def verify_quote(self, quote: str, card_ids: list[str],
                     min_len: int = 12) -> tuple[bool, str]:
        """Check a quote appears in at least one of the cited cards.

        Exact normalized substring first; then a token-overlap fallback so a
        model that drops a stop word or joins two underlined fragments is not
        punished for it. Anything below the overlap floor is treated as
        invented.
        """
        q = normalize(quote)
        if len(q) < min_len:
            return True, "quote too short to verify"
        for cid in card_ids:
            ref = self.by_id.get(cid)
            if ref is None:
                continue
            hay = ref.haystack()
            if q in hay:
                return True, ""
            qt = q.split()
            if len(qt) >= 4:
                hayset = set(hay.split())
                overlap = sum(1 for w in qt if w in hayset) / len(qt)
                if overlap >= 0.85:
                    return True, f"near match ({overlap:.0%} token overlap)"
        return False, "quote does not appear in any cited card"


def ground_finding(finding: Finding, ctx: CardContext,
                   raw_citations: list[str] | None = None) -> Finding:
    """Resolve citations, verify quotes, and demote what cannot be backed."""
    notes: list[str] = []
    ids, problems = ctx.resolve(raw_citations if raw_citations is not None
                                else finding.card_ids)
    notes.extend(problems)
    finding.card_ids = ids

    kept_quotes: list[str] = []
    for q in finding.quotes:
        ok, why = ctx.verify_quote(q, ids)
        if ok:
            kept_quotes.append(q)
            if why:
                notes.append(f"quote accepted: {why}")
        else:
            notes.append(f"dropped unverifiable quote: {q[:70]!r} ({why})")
    finding.quotes = kept_quotes

    if not ids:
        finding.grounded = False
        finding.confidence = min(finding.confidence, 0.3)
        if finding.severity.rank > Severity.MINOR.rank:
            finding.severity = Severity.MINOR
        notes.append("no valid card citation — demoted; treat as a prompt to "
                     "look, not as a finding")
    elif any("dropped unverifiable quote" in n for n in notes):
        finding.confidence = min(finding.confidence, 0.6)

    finding.grounding_notes = notes
    return finding


def grounding_report(findings: list[Finding]) -> dict:
    """Aggregate stats. Worth printing after every model-backed run: a sudden
    drop in the grounded rate is how you notice a prompt regression before it
    reaches anyone's speech doc."""
    total = len(findings)
    grounded = sum(1 for f in findings if f.grounded)
    dropped_quotes = sum(
        1 for f in findings for n in f.grounding_notes
        if n.startswith("dropped unverifiable quote"))
    bad_cites = sum(
        1 for f in findings for n in f.grounding_notes
        if "but only C1" in n or "uninterpretable" in n)
    return {
        "findings": total,
        "grounded": grounded,
        "grounded_rate": round(grounded / total, 3) if total else 1.0,
        "dropped_quotes": dropped_quotes,
        "invalid_citations": bad_cites,
    }
