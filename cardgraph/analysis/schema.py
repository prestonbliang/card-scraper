"""Types for the analysis layer.

Everything an analyzer produces is a `Finding`, whether it came from arithmetic
or from a model. That is deliberate: the consumer should not have to care which,
and a shared type forces the model-backed analyzers to meet the same bar the
deterministic ones meet -- a specific claim, the cards it is about, and a fix.

The field that does the most work is `card_ids`. A finding that cites no cards
is an opinion, and opinions about your evidence are worth nothing. The grounding
verifier drops or demotes them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, ClassVar, Literal


class Severity(str, Enum):
    CRITICAL = "critical"   # the position loses to a competent opponent
    MAJOR = "major"         # a real hole a prepared opponent will find
    MINOR = "minor"         # worth fixing when you have time
    INFO = "info"           # observation, not a defect

    @property
    def rank(self) -> int:
        return {"critical": 3, "major": 2, "minor": 1, "info": 0}[self.value]


# The weakness taxonomy. Keeping these as a closed vocabulary rather than free
# text is what lets the UI group them, the tests assert on them, and the model
# be told exactly what it is allowed to claim.
KINDS: dict[str, str] = {
    # deterministic
    "source_concentration": "The position rests on too few distinct sources; one indict collapses it.",
    "recency_decay": "Key evidence is old relative to the rest of the corpus on this question.",
    "power_tagging": "The tag claims more than the underlined text supports.",
    "qualification_gap": "No cited author qualification on a load-bearing card.",
    "no_read_text": "Card has no underlined portion -- nothing to read in round.",
    "thin_read": "Very little of the card is underlined; the warrant may not be in what you read.",
    "answer_gap": "A block exists elsewhere answering this, and you have no response to it.",
    "unanswered_position": "You have cards on this position but no answer blocks at all.",
    "self_contradiction": "Two of your own positions make incompatible claims.",
    "duplicate_bloat": "The same evidence appears many times under different tags.",
    "single_card_contention": "An entire contention rests on one card.",
    # model-backed
    "chain_break": "A step in the causal chain has no card supporting it.",
    "warrant_mismatch": "The card's reasoning does not establish what the tag asserts.",
    "hidden_assumption": "The argument requires an unstated premise an opponent can attack.",
    "impact_gap": "The internal link chain does not reach the claimed impact.",
    "turn_exposure": "The argument is structurally vulnerable to a specific turn.",
    "definitional_weakness": "A key term is doing undefined work.",
    "missing_block": "A standard answer to this position has no block in your files.",
}


@dataclass
class Finding:
    kind: str
    severity: Severity
    title: str
    detail: str
    fix: str = ""
    card_ids: list[str] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)
    quotes: list[str] = field(default_factory=list)
    confidence: float = 1.0
    analyzer: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    # set by the grounding verifier
    grounded: bool = True
    grounding_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d

    @property
    def sort_key(self) -> tuple:
        return (-self.severity.rank, -self.confidence, self.kind)


@dataclass
class ChainLink:
    """One step of an internal link chain: 'buildout raises rates' ->
    'rate increases raise energy burden' -> 'energy burden causes shutoffs'."""

    step: int
    claim: str
    status: Literal["supported", "weak", "missing"] = "missing"
    card_ids: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ContentionAnalysis:
    node_id: str
    title: str
    side: str
    thesis: str = ""
    # One-paragraph plain-language verdict. Purely presentational: every claim
    # in it is already carried by a grounded finding below, so it adds no new
    # assertions -- it just says which one to fix first, in prose.
    verdict: str = ""
    chain: list[ChainLink] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    card_count: int = 0
    distinct_sources: int = 0
    newest_year: int | None = None
    oldest_year: int | None = None
    # scores in [0,1]
    support_score: float = 0.0     # how well the chain is carried by real cards
    exposure_score: float = 0.0    # how much a prepared opponent can hit
    priority: float = 0.0          # what to fix first

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id, "title": self.title, "side": self.side,
            "thesis": self.thesis, "verdict": self.verdict,
            "chain": [c.to_dict() for c in self.chain],
            "findings": [f.to_dict() for f in sorted(self.findings,
                                                     key=lambda x: x.sort_key)],
            "card_count": self.card_count,
            "distinct_sources": self.distinct_sources,
            "newest_year": self.newest_year, "oldest_year": self.oldest_year,
            "support_score": round(self.support_score, 3),
            "exposure_score": round(self.exposure_score, 3),
            "priority": round(self.priority, 3),
        }


@dataclass
class GeneratedBlock:
    """A block you do not have and probably should.

    `matched_card_ids` is what makes this useful rather than noise: for each
    proposed answer we search the corpus, so the output distinguishes "you
    already own a card for this, it is just filed elsewhere" from "you need to
    go cut this", and gives a search query for the latter.
    """

    against: str
    title: str
    argument: str
    matched_card_ids: list[str] = field(default_factory=list)
    match_scores: list[float] = field(default_factory=list)
    search_query: str = ""
    coverage: Literal["have", "partial", "none"] = "none"
    priority: Severity = Severity.MINOR

    # Above this cosine we are willing to say you have the argument covered.
    STRONG: ClassVar[float] = 0.60

    @property
    def have_it(self) -> bool:
        """Back-compat boolean. Only true for a strong match -- never for a
        partial one, because "partial" exists precisely so that a weak
        similarity does not get reported as coverage."""
        return self.coverage == "have"

    @property
    def best_score(self) -> float:
        return max(self.match_scores) if self.match_scores else 0.0

    def classify(self, small_corpus: bool = False) -> None:
        if not self.matched_card_ids:
            self.coverage = "none"
        elif self.best_score >= self.STRONG and not small_corpus:
            self.coverage = "have"
        else:
            self.coverage = "partial"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["priority"] = self.priority.value
        d["have_it"] = self.have_it
        d["best_score"] = round(self.best_score, 3)
        return d


@dataclass
class CorpusReport:
    contentions: list[ContentionAnalysis] = field(default_factory=list)
    corpus_findings: list[Finding] = field(default_factory=list)
    generated_blocks: list[GeneratedBlock] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    llm_provider: str = "none"
    llm_usage: str = ""
    warnings: list[str] = field(default_factory=list)

    def all_findings(self) -> list[Finding]:
        out = list(self.corpus_findings)
        for c in self.contentions:
            out.extend(c.findings)
        return sorted(out, key=lambda f: f.sort_key)

    def to_dict(self) -> dict:
        return {
            "contentions": [c.to_dict() for c in self.contentions],
            "corpus_findings": [f.to_dict() for f in sorted(
                self.corpus_findings, key=lambda x: x.sort_key)],
            "generated_blocks": [b.to_dict() for b in self.generated_blocks],
            "stats": self.stats, "llm_provider": self.llm_provider,
            "llm_usage": self.llm_usage, "warnings": self.warnings,
        }
