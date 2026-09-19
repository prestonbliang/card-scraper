"""Model-backed analysis: internal link chains, warrant critique, block generation.

Three prompts, each doing something the deterministic layer structurally cannot.

Counting sources is arithmetic. Deciding whether "large-load contracts are
firmed by gas" plus "firmed gas defers retirement" actually gets you to "the
emissions land inside the transition window" is reading comprehension over an
argument, and that is what a model is for.

Prompt discipline used throughout, because the failure mode here is confident
invention rather than error:

* Cards are presented as ``[C1]``…``[Cn]`` and the model cites those labels.
  Every citation is resolved and verified afterwards (see grounding.py).
* The model is given the deterministic findings first, and told not to repeat
  them. That stops it burning attention rediscovering "only two sources" and
  pushes it toward things only it can see.
* ``kind`` is a closed enum. A model allowed to invent categories produces
  twenty near-synonyms and an unusable UI.
* It is explicitly instructed that "no card supports this" is a correct and
  valuable answer. Without that, models manufacture support, because the
  implied task is to find some.
"""

from __future__ import annotations

import re

from ..llm import LLM, LLMError
from .grounding import CardContext, ground_finding
from .schema import (KINDS, ChainLink, Finding, GeneratedBlock, Severity)

MODEL_KINDS = ["chain_break", "warrant_mismatch", "hidden_assumption",
               "impact_gap", "turn_exposure", "definitional_weakness"]

SYSTEM = """You analyze competitive debate evidence files.

You are doing the job of a good opponent doing prep against these files: find \
what actually breaks in a round, not what is stylistically imperfect.

Rules you must follow:
- Cite cards by their bracket label exactly as given: C1, C2, C3. Never invent \
a label outside the range you were shown.
- Quote only text that literally appears in the card you cite. Quotes are \
verified against the source and unverifiable ones are discarded.
- "No card in this set supports that step" is a correct, useful answer. Say it \
plainly. Do not manufacture support that is not there.
- Judge a card by its READ text (the underlined portion actually spoken), not \
by what the tag promises.
- Be specific. "The link is weak" is worthless. "Nothing here connects \
retirement deferral to emissions inside the 2030 window; C3 establishes the \
deferral and stops" is useful.
- Be concise. No preamble, no restating the task."""


CHAIN_SCHEMA = {
    "type": "object",
    "required": ["thesis", "chain"],
    "properties": {
        "thesis": {"type": "string"},
        "chain": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["step", "claim", "status", "cards"],
                "properties": {
                    "step": {"type": "integer"},
                    "claim": {"type": "string"},
                    "status": {"type": "string",
                               "enum": ["supported", "weak", "missing"]},
                    "cards": {"type": "array", "items": {"type": "string"}},
                    "note": {"type": "string"},
                },
            },
        },
    },
}

FINDINGS_SCHEMA = {
    "type": "object",
    "required": ["findings"],
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["kind", "severity", "title", "detail", "cards"],
                "properties": {
                    "kind": {"type": "string", "enum": MODEL_KINDS},
                    "severity": {"type": "string",
                                 "enum": ["critical", "major", "minor"]},
                    "title": {"type": "string"},
                    "detail": {"type": "string"},
                    "fix": {"type": "string"},
                    "cards": {"type": "array", "items": {"type": "string"}},
                    "quotes": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                },
            },
        }
    },
}

BLOCKS_SCHEMA = {
    "type": "object",
    "required": ["blocks"],
    "properties": {
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["title", "argument", "search_query", "priority"],
                "properties": {
                    "title": {"type": "string"},
                    "argument": {"type": "string"},
                    "search_query": {"type": "string"},
                    "priority": {"type": "string",
                                 "enum": ["critical", "major", "minor"]},
                },
            },
        }
    },
}


def _sev(value: str) -> Severity:
    try:
        return Severity(value)
    except ValueError:
        return Severity.MINOR


def _prior_findings_block(prior: list[Finding], limit: int = 8) -> str:
    if not prior:
        return "(none)"
    lines = []
    for f in sorted(prior, key=lambda x: x.sort_key)[:limit]:
        lines.append(f"- [{f.severity.value}] {f.kind}: {f.title}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------

def extract_chain(llm: LLM, title: str, side: str, cards: list[dict],
                  max_cards: int = 24) -> tuple[str, list[ChainLink], list[str]]:
    """Reconstruct the internal link chain and mark which steps carry cards.

    Returns (thesis, links, warnings). A link whose status is 'missing' with no
    citations is the single most actionable output of this whole module: it is
    the step the position asserts and does not prove.
    """
    ctx = CardContext(cards[:max_cards])
    if not len(ctx):
        return "", [], ["no cards"]

    user = f"""Position: {title}
Side: {side}

Cards available:

{ctx.render()}

Reconstruct the argument this position is making as a numbered causal chain \
from its first premise to its terminal impact. Typically 3-6 steps.

For each step:
- `claim`: the step, stated as one plain sentence.
- `cards`: the labels of cards whose READ text establishes that step. Empty \
list if none do.
- `status`: "supported" if a card's read text clearly establishes it, "weak" \
if a card gestures at it but the read text does not carry it, "missing" if no \
card here addresses it at all.
- `note`: for weak/missing, one sentence on exactly what is absent.

Include steps the position *needs* even when no card covers them. A chain that \
only lists the steps you have cards for is useless — the gap is the finding."""

    try:
        data = llm.json(SYSTEM, user, CHAIN_SCHEMA, max_tokens=2500)
    except LLMError as exc:
        return "", [], [f"chain extraction failed: {exc}"]

    warnings: list[str] = []
    links: list[ChainLink] = []
    for raw in data.get("chain", []):
        ids, problems = ctx.resolve(raw.get("cards") or [])
        warnings.extend(problems)
        status = raw.get("status", "missing")
        # A model that marks a step "supported" and cites nothing is
        # contradicting itself; trust the citation, not the label.
        if status == "supported" and not ids:
            status = "missing"
            warnings.append(
                f"step {raw.get('step')} claimed supported with no citation — "
                f"downgraded to missing")
        links.append(ChainLink(
            step=int(raw.get("step") or len(links) + 1),
            claim=(raw.get("claim") or "").strip(),
            status=status, card_ids=ids, note=(raw.get("note") or "").strip(),
        ))
    links.sort(key=lambda link: link.step)
    return (data.get("thesis") or "").strip(), links, warnings


def chain_findings(title: str, links: list[ChainLink]) -> list[Finding]:
    """Turn broken chain links into findings. Deterministic given the chain —
    the model judged the links, this just reports them consistently."""
    out: list[Finding] = []
    for link in links:
        if link.status == "supported":
            continue
        sev = Severity.CRITICAL if link.status == "missing" else Severity.MAJOR
        # A missing first or last step is less fatal than a missing middle one:
        # the ends are often conceded, the internal links never are.
        if link.status == "missing" and link.step in (1, len(links)):
            sev = Severity.MAJOR
        out.append(Finding(
            kind="chain_break" if link.status == "missing" else "warrant_mismatch",
            severity=sev,
            title=f"{title}: step {link.step} is {link.status} — {link.claim[:70]}",
            detail=(link.note or
                    f"No card in this position establishes: {link.claim}"),
            fix=(f"Cut a card for: {link.claim}" if link.status == "missing"
                 else f"Extend the highlighting or re-cut so the read text "
                      f"carries: {link.claim}"),
            card_ids=link.card_ids,
            analyzer="llm:chain",
            confidence=0.75 if link.status == "missing" else 0.65,
            evidence={"step": link.step, "status": link.status},
        ))
    return out


def critique_position(llm: LLM, title: str, side: str, cards: list[dict],
                      prior: list[Finding], max_cards: int = 24) -> tuple[list[Finding], list[str]]:
    """Adversarial read of a position, grounded in its cards."""
    ctx = CardContext(cards[:max_cards])
    if not len(ctx):
        return [], ["no cards"]

    kinds_doc = "\n".join(f"- {k}: {KINDS[k]}" for k in MODEL_KINDS)
    user = f"""Position: {title}
Side: {side}

Cards:

{ctx.render()}

Already found by automated checks — do NOT repeat these, they are handled:
{_prior_findings_block(prior)}

Find what a prepared opponent exploits that the automated checks cannot see. \
Use only these kinds:
{kinds_doc}

Return at most 5 findings, best first. For each, `cards` must contain the \
labels the finding is about, and `quotes` (optional) must be verbatim spans \
from those cards. A finding you cannot tie to a specific card is not worth \
returning.

Prefer one precise finding over five vague ones. If the position is genuinely \
solid, return fewer findings or none."""

    try:
        data = llm.json(SYSTEM, user, FINDINGS_SCHEMA, max_tokens=3000,
                        salvage_key="findings")
    except LLMError as exc:
        return [], [f"critique failed: {exc}"]

    out: list[Finding] = []
    for raw in data.get("findings", []):
        kind = raw.get("kind")
        if kind not in KINDS:
            continue
        f = Finding(
            kind=kind, severity=_sev(raw.get("severity", "minor")),
            title=(raw.get("title") or "").strip()[:160],
            detail=(raw.get("detail") or "").strip(),
            fix=(raw.get("fix") or "").strip(),
            quotes=[q for q in (raw.get("quotes") or []) if isinstance(q, str)],
            confidence=float(raw.get("confidence") or 0.7),
            analyzer="llm:critique",
        )
        out.append(ground_finding(f, ctx, raw.get("cards")))
    return out, []


def dedupe_blocks(blocks: list[GeneratedBlock],
                  threshold: float = 0.6) -> list[GeneratedBlock]:
    """Collapse proposals that are the same argument wearing two titles.

    Models reliably emit "AT: X — Selection Bias" and "AT: X — Undisclosed
    Methodology" as separate items. They are one block, and listing both makes
    the output look padded. Jaccard over content tokens of title+argument,
    keeping the higher-priority version.
    """
    kept: list[GeneratedBlock] = []
    for blk in sorted(blocks, key=lambda b: -b.priority.rank):
        toks = _content_tokens(f"{blk.title} {blk.argument}")
        dup = False
        for other in kept:
            otoks = _content_tokens(f"{other.title} {other.argument}")
            if not toks or not otoks:
                continue
            j = len(toks & otoks) / len(toks | otoks)
            if j >= threshold:
                dup = True
                break
        if not dup:
            kept.append(blk)
    return kept


_STOPWORDS = {"the", "a", "an", "of", "and", "to", "for", "on", "in", "at",
              "is", "it", "that", "this", "not", "but", "as", "by", "with",
              "from", "card", "read", "text", "claim", "argument", "aff",
              "neg", "does", "never", "only", "which", "what", "than", "its"}


def _content_tokens(text: str) -> frozenset[str]:
    words = re.findall(r"[a-z]{4,}", (text or "").lower())
    return frozenset(w for w in words if w not in _STOPWORDS)


def generate_blocks(llm: LLM, title: str, side: str, cards: list[dict],
                    existing_answers: list[str], search_fn=None,
                    max_cards: int = 18,
                    small_corpus: bool = False) -> tuple[list[GeneratedBlock], list[str]]:
    """Propose the answer blocks this position is missing.

    `search_fn(query) -> [(card_id, score)]` checks the corpus for each
    proposal, which is what separates a useful output from a wish list: the
    result distinguishes "you own a card for this, it is just filed elsewhere"
    from "you need to go cut this".

    It must be a *coverage* check, not a ranked search — see
    `SearchEngine.covers`. Ranked search returns its top k for any query at all,
    so a wish-list item nothing supports comes back looking covered, and the
    tool then tells you that you are prepared when you are not.
    """
    ctx = CardContext(cards[:max_cards])
    if not len(ctx):
        return [], ["no cards"]

    have = "\n".join(f"- {t}" for t in existing_answers) or "(none)"
    opponent = "negative" if side == "aff" else "affirmative"
    user = f"""Position: {title} (read by the {side})
Cards this position runs:

{ctx.render(max_read=450)}

Answer blocks that already exist against it:
{have}

You are the {opponent}. Name the standard, high-quality answers to this \
position that are NOT already covered by the existing blocks above.

For each:
- `title`: how the block would be filed, e.g. "AT: Ratepayer Harm — Fuel Cost Confound"
- `argument`: the actual argument in 2-3 sentences, specific enough to cut cards for
- `search_query`: a natural-language query to find supporting evidence
- `priority`: how much trouble this causes if unanswered

At most 5. Real arguments a competent opponent would actually run, not a \
taxonomy of everything imaginable."""

    try:
        data = llm.json(SYSTEM, user, BLOCKS_SCHEMA, max_tokens=2500,
                        salvage_key="blocks")
    except LLMError as exc:
        return [], [f"block generation failed: {exc}"]

    out: list[GeneratedBlock] = []
    for raw in data.get("blocks", []):
        blk = GeneratedBlock(
            against=title,
            title=(raw.get("title") or "").strip()[:160],
            argument=(raw.get("argument") or "").strip(),
            search_query=(raw.get("search_query") or "").strip(),
            priority=_sev(raw.get("priority", "minor")),
        )
        if search_fn and blk.search_query:
            try:
                matches = search_fn(blk.search_query)
            except Exception:
                matches = []
            blk.matched_card_ids = [m[0] for m in matches][:5]
            blk.match_scores = [round(float(m[1]), 3) for m in matches][:5]
        blk.classify(small_corpus=small_corpus)
        out.append(blk)
    return dedupe_blocks(out), []


def summarize_position(llm: LLM, title: str, thesis: str,
                       findings: list[Finding]) -> str:
    """One-paragraph plain-language verdict. Purely presentational — every
    claim in it is already backed by a grounded finding above."""
    if not findings:
        return ""
    body = "\n".join(
        f"- [{f.severity.value}] {f.title}: {f.detail}" for f in
        sorted(findings, key=lambda x: x.sort_key)[:8])
    user = (f"Position: {title}\nThesis: {thesis or '(not extracted)'}\n\n"
            f"Findings:\n{body}\n\n"
            f"Write one paragraph, 3-4 sentences, telling the debater what is "
            f"actually wrong with this position and what to fix first. Plain "
            f"declarative prose, no headers, no lists, no jargon. Do not "
            f"introduce any claim not present in the findings above.")
    try:
        return llm.text(SYSTEM, user, max_tokens=400).strip()
    except LLMError:
        return ""
