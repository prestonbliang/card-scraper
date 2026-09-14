"""Tests for the parts that are easy to get silently wrong.

The failure mode this suite exists to catch is not a crash. It is a pipeline
that runs clean, reports a card count, and has read no read text at all --
because the style resolver missed how a file marks its underlining. That looks
like success in every log line. So `test_read_text_coverage` asserts on
coverage, not on cards parsed.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cardgraph.graph.relate import build_all, normalize_title, similarity
from cardgraph.index.search import SearchEngine, _fts_escape
from cardgraph.index.store import Store
from cardgraph.models import Card, Cite, Source, answers_target, infer_side
from cardgraph.parse.cite import looks_like_cite, parse_cite
from cardgraph.parse.docx_card import ELISION, parse_docx
from seed.generate_synthetic import build as build_synthetic


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpus")
    return build_synthetic(str(d / "synth.docx"))


@pytest.fixture(scope="module")
def parsed(corpus):
    return parse_docx(corpus)


# --- parsing --------------------------------------------------------------

def test_cards_parsed(parsed):
    cards, _, report = parsed
    assert len(cards) == 7
    assert report.cards == 7


def test_read_text_coverage(parsed):
    """The invariant that actually matters."""
    _, _, report = parsed
    assert report.read_text_coverage == 1.0
    assert not report.warnings


def test_read_text_is_a_subset_not_the_whole_body(parsed):
    cards, _, _ = parsed
    for c in cards:
        assert c.read_text, f"no read text on {c.tag!r}"
        assert 0.05 < c.read_ratio < 0.95, (
            f"{c.tag!r} read_ratio={c.read_ratio} — a card whose read text is "
            f"the entire body means the mark resolver fell back to 'everything'")


def test_non_contiguous_reads_get_an_elision_marker(parsed):
    """Underlined spans are non-contiguous; concatenating them welds words
    together into sentences the author never wrote."""
    cards, _, _ = parsed
    assert any(ELISION.strip() in c.read_text for c in cards)
    for c in cards:
        assert "baseresidential" not in c.read_text.replace(" ", "")


def test_emphasis_is_narrower_than_read(parsed):
    cards, _, _ = parsed
    withemph = [c for c in cards if c.emphasis_text]
    assert withemph
    for c in withemph:
        assert len(c.emphasis_text) < len(c.read_text)


def test_outline_hierarchy(parsed):
    _, root, _ = parsed
    titles = []

    def walk(n):
        titles.append(n.title)
        for ch in n.children:
            walk(ch)

    walk(root)
    assert "Grid Cost DA" in titles
    assert "AT: Ratepayer Harm" in titles
    assert "Contention One -- Ratepayer Harm" in titles


# --- cites ----------------------------------------------------------------

@pytest.mark.parametrize("raw,author,year", [
    ('Almeida 31 (Renata Almeida, Sr. Fellow, "X," 3/2/31)', "Almeida", 2031),
    ("Okonkwo 30 (T. Okonkwo, Prof. of Regulatory Economics)", "Okonkwo", 2030),
    ("Bergström 31 (A. Bergström, Assoc., 2/22/31)", "Bergström", 2031),
])
def test_parse_cite(raw, author, year):
    c = parse_cite(raw)
    assert c.author and c.author.startswith(author.split()[0])
    assert c.year == year


def test_two_digit_year_disambiguation():
    assert parse_cite("Smith 98 (...)").year == 1998
    assert parse_cite("Smith 08 (...)").year == 2008


def test_looks_like_cite_rejects_prose():
    prose = ("Across the six planning regions we surveyed in 2029, large-load "
             "interconnection agreements have shifted upgrade costs onto the "
             "requesting customer rather than the general ratepayer base, a "
             "reversal of prior practice that we document at length below.")
    assert not looks_like_cite(prose)


# --- graph ----------------------------------------------------------------

@pytest.mark.parametrize("heading,target", [
    ("AT: Ratepayer Harm", "Ratepayer Harm"),
    ("A/T: Grid Cost DA", "Grid Cost DA"),
    ("Answers to: Fossil Lock-In", "Fossil Lock-In"),
    ("2AC AT: Politics", "Politics"),
])
def test_answers_target(heading, target):
    assert answers_target(heading) == target


def test_answers_target_ignores_normal_blocks():
    assert answers_target("Uniqueness") is None
    assert answers_target("Contention One") is None


def test_similarity_tolerates_naming_drift():
    assert similarity("Ratepayer Harm", "AT: Ratepayer Harm") == 1.0
    assert normalize_title("2AC AT: Rate Payer  DA") == "rate payer da"
    assert similarity("Ratepayer Harm", "Fossil Lock-In") == 0.0
    # long titles are where the exact-match shortcut earns its keep
    assert similarity("AT: Cost Allocation Reform Solves Ratepayer Impacts",
                      "Cost Allocation Reform Solves Ratepayer Impacts") == 1.0


# --- ids ------------------------------------------------------------------

def _card(**kw):
    base = dict(tag="t", cite=Cite(raw="Smith 24"), body="b" * 100,
                read_text="the read portion", source_id="s", ordinal=0)
    base.update(kw)
    return Card(**base)


def test_same_evidence_retagged_collapses_to_one_id():
    a = _card(tag="Buildout raises rates")
    b = _card(tag="Rate impacts are large")
    assert a.card_id == b.card_id


def test_empty_read_text_cards_do_not_collapse():
    """Regression: 33 outline placeholders once hashed to a single row."""
    a = _card(tag="tag one", read_text="", cite=Cite(raw=""), ordinal=0)
    b = _card(tag="tag two", read_text="", cite=Cite(raw=""), ordinal=1)
    assert a.card_id != b.card_id


def test_infer_side():
    assert infer_side(["Grid Cost DA", "1NC"]).value == "neg"
    assert infer_side(["Ratepayer Advantage", "1AC"]).value == "aff"


# --- search ---------------------------------------------------------------

def test_fts_escape_neutralizes_syntax():
    assert _fts_escape('rate* "exact" (x)') == '"rate" "exact" "x"'
    assert _fts_escape("a AND b") == '"a" AND "b"'


@pytest.fixture(scope="module")
def engine(parsed, corpus):
    cards, root, _ = parsed
    db = os.path.join(tempfile.mkdtemp(), "t.db")
    store = Store(db)
    store.add_cards(cards)
    store.add_source(Source(
        source_id=cards[0].source_id,
        path="/corpus/Greenhill/synthetic.docx",
        title="Greenhill 2026 synthetic",
        origin="local",
        license="test fixture",
        url="https://example.invalid/greenhill",
        card_count=len(cards),
    ))
    store.add_outline(root)
    build_all(store)
    e = SearchEngine(store)
    e.build()
    return e, store


def test_lexical_query_finds_author(engine):
    e, _ = engine
    hits = e.search("Okonkwo", k=5)
    assert hits and "Okonkwo" in hits[0].cite_raw


def test_semantic_query_without_shared_vocabulary(engine):
    """No content word here appears in the target card's read text."""
    e, _ = engine
    hits = e.search("households end up paying for wires they do not use", k=3)
    assert hits
    assert "households" in hits[0].tag.lower() or "household" in hits[0].read_text.lower()


def test_source_filter_limits_lexical_and_vector_results(engine):
    e, _ = engine
    assert e.search("Okonkwo", k=5, source="Greenhill")
    assert not e.search("Okonkwo", k=5, source="another-school")


def test_answer_edges_built(engine):
    _, store = engine
    n = store.conn.execute(
        "SELECT COUNT(*) FROM edges WHERE kind='answers'").fetchone()[0]
    assert n >= 2


def test_flags_catch_the_hedged_body_pattern(engine):
    _, store = engine
    rows = store.conn.execute(
        "SELECT warrant_flags FROM cards WHERE warrant_flags IS NOT NULL").fetchall()
    assert rows
