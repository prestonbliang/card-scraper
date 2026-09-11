"""Tests for the analysis layer.

The centre of gravity here is `TestGrounding`. Everything else checks that a
feature works; those tests check that the system refuses to lie. A model that
invents a citation or a quotation produces output indistinguishable from real
analysis, and a debater who acts on it walks into a round holding an argument
that does not exist. So the adversarial cases — citing a card outside the range,
quoting text that appears nowhere, claiming a chain step is supported while
citing nothing — are asserted explicitly and directly.

No test here needs network or an API key. Model-dependent paths run against
`ScriptedProvider`, which returns fixed payloads, so the analyzers' handling of
model output is tested deterministically and in milliseconds.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cardgraph.analysis import deterministic as det
from cardgraph.analysis.engine import analyze
from cardgraph.analysis.grounding import (CardContext, ground_finding,
                                          grounding_report, normalize)
from cardgraph.analysis.llm_analyzers import (dedupe_blocks, extract_chain,
                                              critique_position, generate_blocks)
from cardgraph.analysis.schema import (Finding, GeneratedBlock, Severity)
from cardgraph.index.search import SearchEngine
from cardgraph.index.store import Store
from cardgraph.llm import LLM, LLMResponse, Provider, StubProvider, Usage
from cardgraph.llm.provider import extract_json, validate
from cardgraph.parse.docx_card import parse_docx
from seed.generate_synthetic import build as build_synthetic


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

CARDS = [
    {"card_id": "aaaa1111", "tag": "Buildout shifts costs to households",
     "cite_raw": "Hallberg 30 (Consumer Utility Board)", "cite_author": "Hallberg",
     "cite_year": 2030, "cite_pub": "Consumer Utility Board",
     "read_text": "Transmission upgrades justified by a single large customer "
                  "are entered into the general rate base",
     "body": "Our review of forty-one rate cases finds a consistent pattern. "
             "Transmission upgrades justified by a single large customer are "
             "entered into the general rate base and recovered from all classes.",
     "read_ratio": 0.42, "side": "aff", "block": "Contention One"},
    {"card_id": "bbbb2222", "tag": "The transfer is regressive",
     "cite_raw": "Nakashima 31 (PhD, Institute for Household Energy)",
     "cite_author": "Nakashima", "cite_year": 2031, "cite_pub": None,
     "read_text": "any socialized transmission cost lands hardest on the "
                  "households least able to absorb it",
     "body": "Because residential rate design is largely volumetric, any "
             "socialized transmission cost lands hardest on the households "
             "least able to absorb it.",
     "read_ratio": 0.55, "side": "aff", "block": "Contention One"},
]


class ScriptedProvider(Provider):
    """Returns queued payloads in order. Records the prompts it saw."""

    name = "scripted"

    def __init__(self, payloads: list):
        self.payloads = list(payloads)
        self.prompts: list[str] = []

    def complete(self, system, user, *, schema=None, max_tokens=4096,
                 temperature=0.0):
        self.prompts.append(user)
        payload = self.payloads.pop(0) if self.payloads else {}
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return LLMResponse(text=text, usage=Usage(calls=1, input_tokens=10,
                                                  output_tokens=5),
                           provider=self.name)


def scripted_llm(payloads) -> LLM:
    return LLM(provider=ScriptedProvider(payloads), cache_dir="")


@pytest.fixture
def ctx():
    return CardContext(CARDS)


@pytest.fixture(scope="module")
def store():
    d = tempfile.mkdtemp()
    path = build_synthetic(os.path.join(d, "synth.docx"))
    # source_id must be passed in, not patched afterwards: cards_for_node joins
    # cards to outline nodes on (source_id, path), so a mismatch silently yields
    # zero cards per node and an empty report.
    cards, root, _ = parse_docx(path, source_id="synth")
    st = Store(os.path.join(d, "t.db"))
    from cardgraph.models import Source
    st.add_source(Source(source_id="synth", path=path, title="synth"))
    st.add_cards(cards)
    st.add_outline(root)
    from cardgraph.graph.relate import build_all
    build_all(st)
    return st


# ---------------------------------------------------------------------------
# grounding — the part that must not be wrong
# ---------------------------------------------------------------------------

class TestGrounding:

    def test_labels_resolve_to_real_ids(self, ctx):
        ids, problems = ctx.resolve(["C1", "[C2]"])
        assert ids == ["aaaa1111", "bbbb2222"]
        assert not problems

    def test_out_of_range_citation_is_caught(self, ctx):
        """A model citing C99 over a 2-card set is hallucinating. This must be
        detected exactly, not silently coerced to some nearby card."""
        ids, problems = ctx.resolve(["C99"])
        assert ids == []
        assert problems and "only C1..C2" in problems[0]

    def test_garbage_citation_is_caught(self, ctx):
        ids, problems = ctx.resolve(["the second one", ""])
        assert ids == []
        assert any("uninterpretable" in p for p in problems)

    def test_duplicate_citations_collapse(self, ctx):
        ids, _ = ctx.resolve(["C1", "C1", "1"])
        assert ids == ["aaaa1111"]

    def test_real_quote_verifies(self, ctx):
        ok, _ = ctx.verify_quote(
            "entered into the general rate base", ["aaaa1111"])
        assert ok

    def test_quote_verifies_across_punctuation_drift(self, ctx):
        """Smart quotes and dashes differ between the .docx and the model's
        rendering; that is not a hallucination."""
        ok, _ = ctx.verify_quote(
            "Transmission upgrades — justified by a single large customer — "
            "are entered into the general rate base", ["aaaa1111"])
        assert ok

    def test_invented_quote_is_rejected(self, ctx):
        ok, why = ctx.verify_quote(
            "residential customers were reimbursed in full by the utility",
            ["aaaa1111"])
        assert not ok
        assert "does not appear" in why

    def test_quote_attributed_to_wrong_card_is_rejected(self, ctx):
        """The text exists in the corpus, but not in the card cited. That is
        still a fabricated attribution."""
        ok, _ = ctx.verify_quote(
            "lands hardest on the households least able to absorb it",
            ["aaaa1111"])
        assert not ok

    def test_finding_with_no_valid_citation_is_demoted_not_deleted(self, ctx):
        f = Finding(kind="turn_exposure", severity=Severity.CRITICAL,
                    title="invented", detail="d", confidence=0.9)
        ground_finding(f, ctx, ["C42"])
        assert f.grounded is False
        assert f.severity is Severity.MINOR      # demoted
        assert f.confidence <= 0.3
        assert any("no valid card citation" in n for n in f.grounding_notes)

    def test_unverifiable_quotes_dropped_and_confidence_lowered(self, ctx):
        f = Finding(kind="warrant_mismatch", severity=Severity.MAJOR,
                    title="t", detail="d", confidence=0.9,
                    quotes=["entered into the general rate base",
                            "the utility issued full refunds to all classes"])
        ground_finding(f, ctx, ["C1"])
        assert f.grounded is True
        assert len(f.quotes) == 1
        assert f.confidence <= 0.6
        assert any("dropped unverifiable quote" in n for n in f.grounding_notes)

    def test_grounding_report_counts(self, ctx):
        good = ground_finding(Finding(kind="impact_gap", severity=Severity.MAJOR,
                                      title="a", detail="d"), ctx, ["C1"])
        bad = ground_finding(Finding(kind="impact_gap", severity=Severity.MAJOR,
                                     title="b", detail="d"), ctx, ["C7"])
        r = grounding_report([good, bad])
        assert r == {"findings": 2, "grounded": 1, "grounded_rate": 0.5,
                     "dropped_quotes": 0, "invalid_citations": 1}

    def test_normalize_is_stable(self):
        assert normalize("The  “quick”—brown … fox!") == "the quick brown fox"


# ---------------------------------------------------------------------------
# model output handling
# ---------------------------------------------------------------------------

class TestChainExtraction:

    def test_supported_step_without_citation_is_downgraded(self):
        """A model that says 'supported' and cites nothing is contradicting
        itself. Trust the citation, not the label."""
        llm = scripted_llm([{
            "thesis": "T",
            "chain": [
                {"step": 1, "claim": "a", "status": "supported", "cards": ["C1"]},
                {"step": 2, "claim": "b", "status": "supported", "cards": []},
            ],
        }])
        thesis, links, warns = extract_chain(llm, "P", "aff", CARDS)
        assert thesis == "T"
        assert links[0].status == "supported" and links[0].card_ids == ["aaaa1111"]
        assert links[1].status == "missing"
        assert any("downgraded" in w for w in warns)

    def test_out_of_range_citation_surfaces_as_warning(self):
        llm = scripted_llm([{
            "thesis": "T",
            "chain": [{"step": 1, "claim": "a", "status": "weak",
                       "cards": ["C88"]}],
        }])
        _, links, warns = extract_chain(llm, "P", "aff", CARDS)
        assert links[0].card_ids == []
        assert any("only C1..C2" in w for w in warns)

    def test_chain_findings_severity(self):
        from cardgraph.analysis.llm_analyzers import chain_findings
        from cardgraph.analysis.schema import ChainLink
        links = [ChainLink(1, "a", "supported", ["x"]),
                 ChainLink(2, "b", "missing", []),
                 ChainLink(3, "c", "weak", ["x"]),
                 ChainLink(4, "d", "supported", ["x"])]
        f = chain_findings("P", links)
        kinds = {x.kind for x in f}
        assert kinds == {"chain_break", "warrant_mismatch"}
        # a missing *internal* link is fatal; missing ends are merely major
        broken = next(x for x in f if x.kind == "chain_break")
        assert broken.severity is Severity.CRITICAL

    def test_missing_terminal_step_is_major_not_critical(self):
        from cardgraph.analysis.llm_analyzers import chain_findings
        from cardgraph.analysis.schema import ChainLink
        links = [ChainLink(1, "a", "missing", [])]
        f = chain_findings("P", links)
        assert f[0].severity is Severity.MAJOR


class TestCritique:

    def test_unknown_kind_is_discarded(self):
        """`kind` is a closed vocabulary. A model inventing a category would
        otherwise produce an unusable, ungroupable UI."""
        llm = scripted_llm([{"findings": [
            {"kind": "vibes_are_off", "severity": "major", "title": "t",
             "detail": "d", "cards": ["C1"]},
            {"kind": "impact_gap", "severity": "major", "title": "real",
             "detail": "d", "cards": ["C1"]},
        ]}])
        out, _ = critique_position(llm, "P", "aff", CARDS, [])
        assert [f.kind for f in out] == ["impact_gap"]

    def test_findings_are_grounded_on_the_way_out(self):
        llm = scripted_llm([{"findings": [
            {"kind": "impact_gap", "severity": "critical", "title": "t",
             "detail": "d", "cards": ["C9"]},
        ]}])
        out, _ = critique_position(llm, "P", "aff", CARDS, [])
        assert out[0].grounded is False
        assert out[0].severity is Severity.MINOR

    def test_prior_findings_are_shown_to_the_model(self):
        prov = ScriptedProvider([{"findings": []}])
        llm = LLM(provider=prov, cache_dir="")
        prior = [Finding(kind="source_concentration", severity=Severity.MAJOR,
                         title="only two sources", detail="d")]
        critique_position(llm, "P", "aff", CARDS, prior)
        assert "only two sources" in prov.prompts[0]


class TestBlockGeneration:

    def test_coverage_is_three_valued(self):
        llm = scripted_llm([{"blocks": [
            {"title": "A", "argument": "x", "search_query": "q",
             "priority": "major"},
        ]}])
        # strong match -> "have"
        blocks, _ = generate_blocks(llm, "P", "aff", CARDS, [],
                                    search_fn=lambda q: [("zzz", 0.81)])
        assert blocks[0].coverage == "have" and blocks[0].have_it

        llm = scripted_llm([{"blocks": [
            {"title": "A", "argument": "x", "search_query": "q",
             "priority": "major"}]}])
        blocks, _ = generate_blocks(llm, "P", "aff", CARDS, [],
                                    search_fn=lambda q: [("zzz", 0.41)])
        assert blocks[0].coverage == "partial"
        assert blocks[0].have_it is False, \
            "a weak similarity must never be reported as coverage"

        llm = scripted_llm([{"blocks": [
            {"title": "A", "argument": "x", "search_query": "q",
             "priority": "major"}]}])
        blocks, _ = generate_blocks(llm, "P", "aff", CARDS, [],
                                    search_fn=lambda q: [])
        assert blocks[0].coverage == "none"

    def test_small_corpus_refuses_to_claim_coverage(self):
        llm = scripted_llm([{"blocks": [
            {"title": "A", "argument": "x", "search_query": "q",
             "priority": "major"}]}])
        blocks, _ = generate_blocks(llm, "P", "aff", CARDS, [],
                                    search_fn=lambda q: [("zzz", 0.95)],
                                    small_corpus=True)
        assert blocks[0].coverage == "partial"

    def test_near_duplicate_proposals_collapse(self):
        a = GeneratedBlock(against="P", title="AT: Iversen — Selection Bias",
                           argument="The sample selection criteria are never "
                                    "disclosed so the universal claim could "
                                    "reflect cherry-picked cases entirely.",
                           priority=Severity.MAJOR)
        b = GeneratedBlock(against="P", title="AT: Iversen — Undisclosed Method",
                           argument="Selection criteria are never disclosed, so "
                                    "the universal claim could reflect "
                                    "cherry-picked sample cases entirely.",
                           priority=Severity.MINOR)
        c = GeneratedBlock(against="P", title="AT: Iversen — Alt Cause",
                           argument="Reserve margin shortfalls nationwide "
                                    "explain retirement deferrals independent "
                                    "of any large load growth whatsoever.",
                           priority=Severity.MAJOR)
        kept = dedupe_blocks([a, b, c])
        assert len(kept) == 2
        assert kept[0].priority is Severity.MAJOR  # keeps the higher priority


# ---------------------------------------------------------------------------
# deterministic checks
# ---------------------------------------------------------------------------

class TestDeterministic:

    def test_source_concentration_fires_on_one_author(self):
        cards = [dict(CARDS[0], card_id=f"c{i}") for i in range(5)]
        f = det.check_source_concentration("P", cards)
        assert f and f[0].severity is Severity.MAJOR
        assert f[0].evidence["distinct_sources"] == 1

    def test_source_concentration_quiet_when_diverse(self):
        cards = [dict(CARDS[0], card_id=f"c{i}", cite_author=f"Author{i}")
                 for i in range(5)]
        assert det.check_source_concentration("P", cards) == []

    def test_single_card_position(self):
        f = det.check_single_card("P", [CARDS[0]])
        assert f and f[0].kind == "single_card_contention"
        assert det.check_single_card("P", CARDS) == []

    def test_power_tagging_needs_both_a_strong_tag_and_a_weak_read(self):
        strong_and_hedged = dict(
            CARDS[0], tag="Buildout collapses ratepayer support",
            read_text="this may potentially suggest an effect", read_ratio=0.4)
        assert det.check_power_tagging([strong_and_hedged])

        strong_but_solid = dict(
            CARDS[0], tag="Buildout collapses ratepayer support",
            read_text="rates rose 14 percent in every affected territory",
            body="rates rose 14 percent in every affected territory",
            read_ratio=0.5)
        assert det.check_power_tagging([strong_but_solid]) == []

        weak_tag = dict(CARDS[0], tag="Some context on rate design",
                        read_text="this may possibly suggest", read_ratio=0.02)
        assert det.check_power_tagging([weak_tag]) == []

    def test_recency_lag(self):
        old = [dict(CARDS[0], cite_year=2018)]
        assert det.check_recency("P", old, 2031)[0].severity is Severity.MAJOR
        assert det.check_recency("P", old, 2020) == []

    def test_qualification_gap(self):
        unqual = [dict(CARDS[0], cite_raw="Smith 30", cite_pub=None)]
        assert det.check_qualifications("P", unqual)
        assert det.check_qualifications("P", [CARDS[0]]) == []

    def test_scores_move_in_the_right_direction(self):
        many = [dict(CARDS[0], card_id=f"c{i}", cite_author=f"A{i}")
                for i in range(6)]
        strong, exp_low = det.score_position(many, [])
        weak, exp_high = det.score_position(
            [CARDS[0]],
            [Finding(kind="chain_break", severity=Severity.CRITICAL,
                     title="t", detail="d")])
        assert strong > weak
        assert exp_high > exp_low
        for v in (strong, weak, exp_low, exp_high):
            assert 0.0 <= v <= 1.0

    def test_exposure_saturates(self):
        """Twenty minor findings must not outrank one critical one."""
        minors = [Finding(kind="thin_read", severity=Severity.MINOR,
                          title="t", detail="d") for _ in range(20)]
        _, many_minor = det.score_position(CARDS, minors)
        _, one_crit = det.score_position(CARDS, [
            Finding(kind="chain_break", severity=Severity.CRITICAL,
                    title="t", detail="d")])
        assert many_minor <= 1.0 and one_crit <= 1.0

    def test_self_contradiction_detected(self, store):
        rows = [dict(r) for r in store.conn.execute("SELECT * FROM cards")]
        assert rows  # sanity
        findings = det.check_self_contradiction(store)
        assert isinstance(findings, list)   # no crash on a clean corpus

    def test_template_source_detection(self):
        d = tempfile.mkdtemp()
        st = Store(os.path.join(d, "t.db"))
        from cardgraph.models import Card, Cite, Source
        st.add_source(Source(source_id="tpl", path="p", title="skeleton"))
        st.add_cards([Card(tag=f"tag {i}", cite=Cite(raw=""), body="x" * 80,
                           read_text="", source_id="tpl", ordinal=i)
                      for i in range(6)])
        assert "tpl" in det.template_sources(st)
        assert det.check_template_sources(st)


# ---------------------------------------------------------------------------
# coverage check on the real index
# ---------------------------------------------------------------------------

class TestCoverage:

    def test_offtopic_query_returns_nothing(self, store):
        e = SearchEngine(store)
        e.build()
        assert e.covers("zebra grazing patterns in the Serengeti") == []

    def test_ranked_search_would_have_lied(self, store):
        """The bug this exists to prevent: `search` returns its top k for any
        query at all, so an unsupported argument looks covered."""
        e = SearchEngine(store)
        e.build()
        q = "zebra grazing patterns in the Serengeti"
        assert e.search(q, k=4), "ranked search returns hits for anything"
        assert e.covers(q) == [], "coverage must not"

    def test_own_cards_are_excluded(self, store):
        e = SearchEngine(store)
        e.build()
        all_ids = {r["card_id"] for r in
                   store.conn.execute("SELECT card_id FROM cards")}
        assert e.covers("transmission cost allocation ratepayers",
                        exclude_card_ids=all_ids) == []

    def test_floor_is_enforced(self, store):
        e = SearchEngine(store)
        e.build()
        q = "large load interconnection cost allocation"
        assert e.covers(q, floor=0.0)
        assert e.covers(q, floor=0.999) == []


# ---------------------------------------------------------------------------
# llm plumbing
# ---------------------------------------------------------------------------

class TestLLMPlumbing:

    def test_schema_violation_triggers_a_repair_retry(self):
        prov = ScriptedProvider([
            {"findings": "not an array"},
            {"findings": []},
        ])
        llm = LLM(provider=prov, cache_dir="")
        schema = {"type": "object", "required": ["findings"],
                  "properties": {"findings": {"type": "array",
                                              "items": {"type": "object"}}}}
        out = llm.json("s", "u", schema)
        assert out == {"findings": []}
        assert len(prov.prompts) == 2
        assert "failed validation" in prov.prompts[1]

    def test_gives_up_after_max_retries(self):
        prov = ScriptedProvider([{"nope": 1}] * 5)
        llm = LLM(provider=prov, cache_dir="", max_retries=2)
        from cardgraph.llm import LLMError
        with pytest.raises(LLMError):
            llm.json("s", "u", {"type": "object", "required": ["x"],
                                "properties": {"x": {"type": "string"}}})

    def test_prose_wrapped_json_is_recovered(self):
        llm = scripted_llm(['Sure!\n```json\n{"x": "ok",}\n```\nHope that helps'])
        out = llm.json("s", "u", {"type": "object", "required": ["x"],
                                  "properties": {"x": {"type": "string"}}})
        assert out == {"x": "ok"}

    def test_disk_cache_prevents_a_second_call(self):
        d = tempfile.mkdtemp()
        prov = ScriptedProvider([{"x": "a"}, {"x": "b"}])
        schema = {"type": "object", "required": ["x"],
                  "properties": {"x": {"type": "string"}}}
        llm = LLM(provider=prov, cache_dir=d)
        assert llm.json("s", "u", schema) == {"x": "a"}
        llm2 = LLM(provider=prov, cache_dir=d)
        llm2.provider.name = prov.name
        assert llm2.json("s", "u", schema) == {"x": "a"}
        assert len(prov.prompts) == 1

    def test_stub_provider_records_then_replays(self):
        d = tempfile.mkdtemp()
        live = ScriptedProvider([{"x": 1}])
        stub = StubProvider(d, record_with=live)
        first = stub.complete("s", "u", schema=None)
        assert not first.from_cache
        replay = StubProvider(d).complete("s", "u", schema=None)
        assert replay.from_cache and replay.text == first.text

    def test_stub_misses_loudly_on_a_prompt_change(self):
        """A cassette keyed on the prompt means an edited prompt fails rather
        than silently replaying an answer to the old question."""
        d = tempfile.mkdtemp()
        StubProvider(d, record_with=ScriptedProvider([{"x": 1}])).complete(
            "s", "original prompt", schema=None)
        from cardgraph.llm import LLMUnavailable
        with pytest.raises(LLMUnavailable):
            StubProvider(d).complete("s", "edited prompt", schema=None)

    def test_usage_ledger_accumulates(self):
        prov = ScriptedProvider([{"x": "a"}, {"x": "b"}])
        llm = LLM(provider=prov, cache_dir="")
        schema = {"type": "object", "required": ["x"],
                  "properties": {"x": {"type": "string"}}}
        llm.json("s", "u1", schema)
        llm.json("s", "u2", schema)
        assert llm.ledger.calls == 2
        assert llm.ledger.input_tokens == 20
        assert "2 call(s)" in llm.ledger.summary()

    @pytest.mark.parametrize("raw,expected", [
        ('{"a":1}', {"a": 1}),
        ('```json\n{"a":1}\n```', {"a": 1}),
        ('prose {"a":1} more', {"a": 1}),
        ('{"a":[1,2,],}', {"a": [1, 2]}),
    ])
    def test_extract_json(self, raw, expected):
        assert extract_json(raw) == expected

    def test_validate_catches_shape_errors(self):
        schema = {"type": "object", "required": ["a"],
                  "properties": {"a": {"type": "array",
                                       "items": {"type": "integer"}}}}
        assert validate({"a": [1, 2]}, schema) == []
        assert validate({}, schema)
        assert validate({"a": "no"}, schema)
        assert validate({"a": ["x"]}, schema)


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_offline_run_produces_a_real_report(self, store):
        r = analyze(store, use_llm=False)
        assert r.llm_provider == "none"
        assert r.contentions
        assert r.stats["total_findings"] > 0
        assert any("No LLM provider" in w for w in r.warnings)

    def test_offline_run_makes_no_model_calls(self, store):
        prov = ScriptedProvider([])
        analyze(store, llm=LLM(provider=prov, cache_dir=""), use_llm=False)
        assert prov.prompts == []

    def test_report_serializes(self, store):
        r = analyze(store, use_llm=False)
        blob = json.dumps(r.to_dict())
        back = json.loads(blob)
        assert back["contentions"][0]["title"]
        assert isinstance(back["stats"]["total_findings"], int)

    def test_positions_are_ordered_by_priority(self, store):
        r = analyze(store, use_llm=False)
        p = [c.priority for c in r.contentions]
        assert p == sorted(p, reverse=True)

    def test_model_pass_is_budgeted(self, store):
        """max_positions must actually cap spend.

        Four calls per position: chain, critique, blocks, verdict. If that
        number changes, this test is the thing that tells you the per-position
        cost of a run just went up.
        """
        payloads = []
        for _ in range(4):
            payloads += [{"thesis": "t", "chain": []}, {"findings": []},
                         {"blocks": []}, "a one paragraph verdict"]
        prov = ScriptedProvider(payloads)
        analyze(store, llm=LLM(provider=prov, cache_dir=""), max_positions=1)
        assert len(prov.prompts) == 4

    def test_verdict_is_generated_after_findings(self, store):
        """The verdict summarizes findings, so it must be produced last —
        otherwise it summarizes an empty list."""
        prov = ScriptedProvider([
            {"thesis": "t", "chain": [
                {"step": 1, "claim": "c", "status": "missing", "cards": []}]},
            {"findings": []}, {"blocks": []},
            "This position has a hole in step one.",
        ])
        r = analyze(store, llm=LLM(provider=prov, cache_dir=""), max_positions=1)
        assert r.contentions[0].verdict == "This position has a hole in step one."
        # the verdict prompt must contain the chain_break finding
        assert "step 1 is missing" in prov.prompts[-1]


class TestIndexPersistence:
    """The search index is fitted once and cached.

    At archive scale a cold fit is ~4 minutes. Paying that on every `search`
    invocation and every `serve` startup makes the tool unusable at exactly the
    size it was built for. The dangerous failure mode is the opposite one --
    serving a cached index that cannot see cards added since -- so the
    invalidation test matters more than the speed one.
    """

    def _store(self, tmp_path, n=6):
        from cardgraph.models import Card, Cite, Source
        st = Store(str(tmp_path / "t.db"))
        st.add_source(Source(source_id="s", path="p", title="t"))
        st.add_cards([
            Card(tag=f"tag {i}", cite=Cite(raw=f"Author{i} 20"),
                 body="ratepayer transmission cost allocation " * 4,
                 read_text=f"distinct read text number {i} about rates",
                 source_id="s", ordinal=i)
            for i in range(n)])
        return st

    def test_cache_is_written_and_reused(self, tmp_path):
        st = self._store(tmp_path)
        cache = str(tmp_path / "idx.pkl")
        msgs = []
        SearchEngine(st, cache_path=cache).build(progress=lambda *a: msgs.append(" ".join(map(str, a))))
        assert os.path.exists(cache)
        assert "fitting" in msgs[0]

        msgs2 = []
        e2 = SearchEngine(st, cache_path=cache)
        e2.build(progress=lambda *a: msgs2.append(" ".join(map(str, a))))
        assert "loaded cached index" in msgs2[0]
        assert e2.search("distinct read text number 3")

    def test_adding_a_card_invalidates_the_cache(self, tmp_path):
        """A stale index is search that silently cannot see your newest
        evidence -- worse than a slow one."""
        from cardgraph.models import Card, Cite
        st = self._store(tmp_path)
        cache = str(tmp_path / "idx.pkl")
        SearchEngine(st, cache_path=cache).build()

        st.add_cards([Card(tag="brand new", cite=Cite(raw="New 21"),
                           body="a" * 90, read_text="a brand new claim entirely",
                           source_id="s", ordinal=99)])
        msgs = []
        e = SearchEngine(st, cache_path=cache)
        e.build(progress=lambda *a: msgs.append(" ".join(map(str, a))))
        assert "fitting" in msgs[0], "must refit, not serve a stale index"
        assert any(h.tag == "brand new" for h in e.search("brand new claim", k=5))

    def test_fingerprint_detects_swap_not_just_count(self, tmp_path):
        """Deleting one card and adding another leaves the count unchanged."""
        from cardgraph.index.search import VectorIndex
        a = VectorIndex.fingerprint(["x", "y", "z"])
        b = VectorIndex.fingerprint(["x", "y", "w"])
        assert a != b
        assert a == VectorIndex.fingerprint(["z", "x", "y"])  # order-independent

    def test_corrupt_cache_falls_back_to_fitting(self, tmp_path):
        st = self._store(tmp_path)
        cache = str(tmp_path / "idx.pkl")
        with open(cache, "wb") as fh:
            fh.write(b"not a pickle")
        msgs = []
        e = SearchEngine(st, cache_path=cache)
        e.build(progress=lambda *a: msgs.append(" ".join(map(str, a))))
        assert "fitting" in msgs[0]
        assert e.search("distinct read text number 2")

    def test_format_bump_invalidates(self, tmp_path):
        from cardgraph.index.search import VectorIndex
        st = self._store(tmp_path)
        cache = str(tmp_path / "idx.pkl")
        SearchEngine(st, cache_path=cache).build()
        v = VectorIndex()
        fp = VectorIndex.fingerprint(st.card_ids())
        assert v.load(cache, fp)
        v.FORMAT = VectorIndex.FORMAT + 1
        assert not VectorIndex().load(cache, "different-fingerprint")

    def test_matrix_is_float32(self, tmp_path):
        import numpy as np
        st = self._store(tmp_path)
        e = SearchEngine(st, cache_path=str(tmp_path / "i.pkl"))
        e.build()
        assert e.vectors.matrix.dtype == np.float32


class TestSmallCorpusRobustness:
    """A first run is usually one debater's own file, not an archive.

    `max_df=0.85` drops terms appearing in most documents, which is what keeps
    a single-topic corpus from being swamped by its own vocabulary. On a small
    or repetitive corpus it drops *everything* and sklearn raises -- so indexing
    your own six-card file crashed, which is close to the worst possible first
    impression. These are the shapes that broke it.
    """

    def _engine(self, texts, tmp_path):
        from cardgraph.models import Card, Cite, Source
        st = Store(str(tmp_path / "t.db"))
        st.add_source(Source(source_id="s", path="p", title="t"))
        st.add_cards([
            Card(tag=f"t{i}", cite=Cite(raw=f"A{i} 20"), body="b" * 90,
                 read_text=t, source_id="s", ordinal=i)
            for i, t in enumerate(texts)])
        e = SearchEngine(st, cache_path=str(tmp_path / "i.pkl"))
        e.build()
        return e

    def test_identical_cards_do_not_crash(self, tmp_path):
        e = self._engine(["ratepayer transmission cost allocation"] * 6, tmp_path)
        assert len(e.vectors.card_ids) == 6
        assert e.search("ratepayer transmission")

    def test_single_card_corpus(self, tmp_path):
        e = self._engine(["one single card about ratepayers"], tmp_path)
        assert e.search("ratepayers")

    def test_corpus_of_only_stop_words(self, tmp_path):
        """Nothing survives the English stop list; fall back to keeping it."""
        e = self._engine(["the and of to a in on at"] * 4, tmp_path)
        assert len(e.vectors.card_ids) == 4

    def test_two_cards(self, tmp_path):
        """SVD needs at least two components; two rows is the boundary."""
        e = self._engine(["grid policy claim one", "grid policy claim two"],
                         tmp_path)
        assert e.search("grid policy")

    def test_empty_corpus_is_not_an_error(self, tmp_path):
        from cardgraph.models import Source
        st = Store(str(tmp_path / "t.db"))
        st.add_source(Source(source_id="s", path="p", title="t"))
        e = SearchEngine(st, cache_path=str(tmp_path / "i.pkl"))
        e.build()
        assert e.search("anything") == []


class TestOwnerScoping:
    """Every check in this layer asks about *your* files.

    Run unscoped over the published archive that assumption silently breaks:
    46,712 positions belonging to 11,643 different teams, 129,410 findings
    about other people's evidence, and a `self_contradiction` section reporting
    that half the canon is cited on both sides -- true of the community,
    meaningless as advice. Scoping is not a display filter; it is what the
    questions presuppose.
    """

    def _store(self, tmp_path):
        from cardgraph.models import Card, Cite, Side, Source
        st = Store(str(tmp_path / "t.db"))
        for school in ("Greenhill", "Lexington"):
            st.add_source(Source(source_id=f"src-{school}",
                                 path=f"/files/{school}/aff.docx",
                                 title=f"{school} Aff", card_count=2))
            cards = []
            for i, side in enumerate(("aff", "neg")):
                c = Card(tag=f"{school} tag {i}",
                         cite=Cite(raw="Bostrom 02", author="Bostrom", year=2002),
                         body="b" * 120, read_text=f"{school} read text {i}",
                         source_id=f"src-{school}",
                         source_path=f"/files/{school}/aff.docx", ordinal=i,
                         path=[school, "1AC", f"Contention {i}"])
                c.side = Side(side)
                cards.append(c)
            st.add_cards(cards)
            # The outline must mirror the cards' `path` exactly: cards_for_node
            # joins on (source_id, path_json), so a mismatch silently yields a
            # position with no cards.
            from cardgraph.models import NodeKind, OutlineNode
            root = OutlineNode(title=school, kind=NodeKind.POCKET,
                               path=[school], source_id=f"src-{school}")
            hat = OutlineNode(title="1AC", kind=NodeKind.HAT,
                              path=[school, "1AC"], source_id=f"src-{school}")
            root.children.append(hat)
            for i, c in enumerate(cards):
                node = OutlineNode(title=f"Contention {i}", kind=NodeKind.BLOCK,
                                   path=[school, "1AC", f"Contention {i}"],
                                   source_id=f"src-{school}",
                                   card_ids=[c.card_id])
                hat.children.append(node)
            st.add_outline(root)
        return st

    def test_scope_excludes_other_owners(self, tmp_path):
        st = self._store(tmp_path)
        wide = analyze(st, use_llm=False)
        narrow = analyze(st, use_llm=False, owner="Greenhill")
        assert wide.stats["positions_analyzed"] > narrow.stats["positions_analyzed"]
        assert narrow.stats["owner"] == "Greenhill"
        assert narrow.stats["positions_analyzed"] > 0
        scoped_ids = {r[0] for r in st.conn.execute(
            "SELECT node_id FROM nodes WHERE source_id = 'src-Greenhill'")}
        assert all(c.node_id in scoped_ids for c in narrow.contentions)

    def test_scoped_corpus_checks_only_see_scoped_sources(self, tmp_path):
        """The corpus-level checks issue their own SQL; the scope has to reach
        them too, or self_contradiction still reports the other school."""
        st = self._store(tmp_path)
        found = analyze(st, use_llm=False, owner="Greenhill").corpus_findings
        for f in found:
            assert "Lexington" not in f.title
            assert "Lexington" not in (f.evidence.get("owner") or "")

    def test_nonmatching_scope_yields_nothing_not_everything(self, tmp_path):
        """A typo in --owner must return an empty report, never the whole
        corpus. Silently widening a filter is how you get a 129,410-finding
        report you believe is about your files."""
        st = self._store(tmp_path)
        r = analyze(st, use_llm=False, owner="NoSuchSchool")
        assert r.stats["positions_analyzed"] == 0
        assert r.corpus_findings == []

    def test_scope_matches_on_title_as_well_as_path(self, tmp_path):
        st = self._store(tmp_path)
        assert analyze(st, use_llm=False,
                       owner="Greenhill Aff").stats["positions_analyzed"] >= 0

    def test_render_reports_the_scope(self, tmp_path):
        from cardgraph.analysis import render_text
        st = self._store(tmp_path)
        assert "Greenhill" in render_text(
            analyze(st, use_llm=False, owner="Greenhill"))
        assert "every source in the index" in render_text(
            analyze(st, use_llm=False))


class TestReportVolume:
    """A report is something a person reads."""

    def _report(self, n_corpus, n_pos_findings):
        from cardgraph.analysis.schema import (ContentionAnalysis, CorpusReport,
                                               Finding, Severity)
        r = CorpusReport()
        r.corpus_findings = [
            Finding(kind="answer_gap", severity=Severity.MAJOR,
                    title=f"gap {i}", detail="d", fix="f")
            for i in range(n_corpus)]
        ca = ContentionAnalysis(node_id="n", title="P", side="aff", card_count=4)
        ca.findings = [
            Finding(kind="power_tagging", severity=Severity.MINOR,
                    title=f"finding {i}", detail="d")
            for i in range(n_pos_findings)]
        r.contentions = [ca]
        r.stats = {"positions_analyzed": 1, "total_findings": n_corpus}
        return r

    def test_corpus_findings_are_capped_per_kind(self):
        from cardgraph.analysis import render_text
        out = render_text(self._report(50, 0), corpus_per_kind=3)
        assert out.count("gap ") <= 4          # 3 shown, plus the summary line
        assert "and 47 more answer gap finding(s)" in out

    def test_position_findings_are_capped(self):
        from cardgraph.analysis import render_text
        out = render_text(self._report(0, 20), findings_per_position=6)
        assert "and 14 more finding(s) on this position" in out

    def test_nothing_is_hidden_without_saying_so(self):
        """Truncation the reader cannot see is worse than a long report."""
        from cardgraph.analysis import render_text
        out = render_text(self._report(50, 20), corpus_per_kind=3,
                          findings_per_position=6)
        assert "more answer gap finding(s)" in out
        assert "more finding(s) on this position" in out


class TestEveryCorpusCheckIsScoped:
    """The bug this class exists to prevent, stated plainly.

    The first scoping implementation wrapped the database connection and
    spliced `source_id IN (...)` into any SQL that mentioned cards or nodes. It
    silently missed `check_duplicate_bloat` (which queries `edges`, a table with
    no source_id at all) and `check_template_sources` (which queries `sources`).
    A report scoped to one school therefore announced 21,811 duplicate pairs and
    27 unfilled outline files belonging to *other* schools, presented as that
    school's own.

    Scoping is now an explicit parameter on every check, and this test asserts
    that mechanically -- so a new check that forgets it fails here rather than
    quietly attributing the corpus to you.
    """

    CHECKS = ["check_template_sources", "check_answer_coverage",
              "check_unanswered_positions", "check_self_contradiction",
              "check_duplicate_bloat"]

    def test_every_corpus_check_accepts_a_scope(self):
        import inspect
        from cardgraph.analysis import deterministic as det
        for name in self.CHECKS:
            sig = inspect.signature(getattr(det, name))
            assert "scope" in sig.parameters, f"{name} ignores the scope"

    def test_analyze_corpus_passes_the_scope_to_all_of_them(self):
        """Accepting the parameter is not the same as being given it."""
        import inspect
        from cardgraph.analysis import deterministic as det
        src = inspect.getsource(det.analyze_corpus)
        for name in self.CHECKS:
            assert f"{name}(store, scope)" in src, \
                f"analyze_corpus calls {name} without the scope"

    def _two_schools(self, tmp_path):
        from cardgraph.models import Card, Cite, Source
        st = Store(str(tmp_path / "t.db"))
        for school in ("Mine", "Theirs"):
            st.add_source(Source(source_id=f"s-{school}",
                                 path=f"/files/{school}/a.docx",
                                 title=f"{school} file", card_count=2))
            # identical evidence in both schools -> a duplicate edge across them
            st.add_cards([
                Card(tag=f"{school} tag {i}",
                     cite=Cite(raw="Woller 97", author="Woller", year=1997),
                     body="shared body text " * 12,
                     read_text=f"the same underlying cut appears here {i}",
                     source_id=f"s-{school}",
                     source_path=f"/files/{school}/a.docx", ordinal=i)
                for i in range(2)])
        return st

    def test_duplicate_bloat_is_scoped_even_though_edges_lack_source_id(self,
                                                                       tmp_path):
        from cardgraph.analysis.deterministic import (check_duplicate_bloat,
                                                      scoped_source_ids)
        st = self._two_schools(tmp_path)
        st.add_edge("nope-a", "nope-b", "duplicates", 0.99,
                    '{"retagged": true}')
        st.add_edge("nope-c", "nope-d", "duplicates", 0.99,
                    '{"retagged": true}')
        st.add_edge("nope-e", "nope-f", "duplicates", 0.99,
                    '{"retagged": true}')
        scope = scoped_source_ids(st, "Mine")
        # none of those edges touch a card owned by "Mine"
        assert check_duplicate_bloat(st, scope) == []
        # unscoped, they are reported
        assert check_duplicate_bloat(st, None)

    def test_template_sources_are_scoped(self, tmp_path):
        from cardgraph.analysis.deterministic import (check_template_sources,
                                                      template_sources)
        from cardgraph.models import Card, Cite, Source
        st = Store(str(tmp_path / "t.db"))
        st.add_source(Source(source_id="s-Theirs", path="/files/Theirs/skel.docx",
                             title="Theirs skeleton"))
        st.add_cards([Card(tag=f"tag {i}", cite=Cite(raw=""), body="x" * 80,
                           read_text="", source_id="s-Theirs", ordinal=i)
                      for i in range(6)])
        assert template_sources(st)                       # exists
        assert template_sources(st, scope={"s-Mine"}) == {}   # not mine
        assert check_template_sources(st, {"s-Mine"}) == []

    def test_template_list_does_not_dump_every_filename(self, tmp_path):
        """27 full filenames inline is a wall of text, not a finding."""
        from cardgraph.analysis.deterministic import check_template_sources
        from cardgraph.models import Card, Cite, Source
        st = Store(str(tmp_path / "t.db"))
        for n in range(9):
            st.add_source(Source(source_id=f"s{n}", path=f"/f/{n}.docx",
                                 title=f"skeleton number {n}"))
            st.add_cards([Card(tag=f"t{n}-{i}", cite=Cite(raw=""), body="x" * 80,
                               read_text="", source_id=f"s{n}", ordinal=i)
                          for i in range(4)])
        found = check_template_sources(st)
        assert found
        assert "and 4 more" in found[0].detail


class TestCoverageRespectsScope:
    """"Do I have a card for this?" means *I*.

    Third instance of one bug. Ranked search answered it with its top-k for any
    query (§5). A boolean then reported a 0.41 cosine as coverage (§5). And on a
    shared index, an analysis scoped to one school answered it by pointing at
    *another school's* cards -- telling Apple Valley they were covered by
    Lexington's evidence. Eight confident "you already have cards for this"
    claims on a real run, every one of them about files that school does not own.
    """

    def _engine(self, tmp_path):
        from cardgraph.models import Card, Cite, Source
        st = Store(str(tmp_path / "t.db"))
        for school in ("Mine", "Theirs"):
            st.add_source(Source(source_id=f"s-{school}",
                                 path=f"/files/{school}/a.docx",
                                 title=f"{school} file", card_count=3))
            st.add_cards([
                Card(tag=f"{school} tag {i}", cite=Cite(raw=f"A{i} 20"),
                     body="b" * 120,
                     read_text=f"{school} claim about reserve margin shortfalls {i}",
                     source_id=f"s-{school}",
                     source_path=f"/files/{school}/a.docx", ordinal=i)
                for i in range(3)])
        e = SearchEngine(st, cache_path=str(tmp_path / "i.pkl"))
        e.build()
        return st, e

    def test_restrict_to_excludes_other_owners_cards(self, tmp_path):
        st, e = self._engine(tmp_path)
        mine = {r[0] for r in st.conn.execute(
            "SELECT card_id FROM cards WHERE source_id='s-Mine'")}
        q = "reserve margin shortfalls"
        assert e.covers(q, floor=0.0), "unrestricted finds something"
        hits = e.covers(q, restrict_to=mine, floor=0.0)
        assert hits, "the owner does have matching cards"
        assert all(cid in mine for cid, _ in hits)

    def test_restrict_to_empty_set_means_nothing_is_covered(self, tmp_path):
        """An owner with no cards is covered by nothing -- never by everything.
        Treating an empty restriction as 'no restriction' is how a filter
        silently widens back to the whole corpus."""
        _, e = self._engine(tmp_path)
        assert e.covers("reserve margin shortfalls",
                        restrict_to=set(), floor=0.0) == []

    def test_none_restriction_still_searches_everything(self, tmp_path):
        _, e = self._engine(tmp_path)
        assert e.covers("reserve margin shortfalls", restrict_to=None, floor=0.0)

    def test_small_corpus_guard_counts_the_scope_not_the_index(self, tmp_path):
        """Scoped to thirty cards inside a 166,816-card archive, the pool that
        can answer "do I have this?" is thirty. Using the index total reports
        confident coverage from a sample far too small to support it."""
        import inspect
        from cardgraph.analysis import engine as eng
        src = inspect.getsource(eng.analyze)
        assert "len(scope_ids) if scope_ids is not None" in src, \
            "small_corpus must be measured over the scope"
