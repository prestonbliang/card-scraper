"""Tests for the local, explainable smart-search layer."""

from __future__ import annotations

from cardgraph.index.search import SearchEngine, parse_smart_request, smart_query_variants
from cardgraph.index.store import Store
from cardgraph.models import Card, Cite, Source


def test_smart_request_extracts_side_and_year_without_guessing_topic():
    text, filters = parse_smart_request(
        "find negative cards from 2024 about nuclear power"
    )

    assert text == "nuclear power"
    assert filters == {"side": "neg", "year_min": 2024}

    text, filters = parse_smart_request("show affirmative cards between 2020 and 2022 on costs")
    assert text == "costs"
    assert filters == {"side": "aff", "year_min": 2020, "year_max": 2022}


def test_smart_query_variants_preserve_original_and_strip_request_framing():
    variants = smart_query_variants("Please find cards about household costs")

    assert variants[0] == "Please find cards about household costs"
    assert "household costs" in variants
    assert any("ratepayer" in variant for variant in variants)
    assert all(variant.strip() for variant in variants)


def test_unknown_terms_do_not_return_arbitrary_zero_similarity_cards(tmp_path):
    store = Store(str(tmp_path / "cards.db"))
    store.add_source(Source(source_id="fixture", path="fixture.docx", title="Fixture"))
    store.add_cards([Card(
        tag="Grid reform", cite=Cite(raw="Example 2024", year=2024),
        body="Grid reform improves allocation. " * 8,
        read_text="Grid reform improves allocation.", source_id="fixture",
        source_path="fixture.docx", ordinal=0,
    )])
    engine = SearchEngine(store, cache_path=str(tmp_path / "index.pkl"))
    engine.build()

    assert engine.search("banana unicorn", k=5) == []
    store.close()


def test_match_badges_distinguish_exact_and_semantic_retrieval(tmp_path):
    store = Store(str(tmp_path / "cards.db"))
    store.add_source(Source(source_id="fixture", path="fixture.docx", title="Fixture"))
    store.add_cards([
        Card(
            tag="Household costs", cite=Cite(raw="Example 2024", year=2024),
            body="Household costs rise with the rate burden. " * 8,
            read_text="Household costs rise with the rate burden.", source_id="fixture",
            source_path="fixture.docx", ordinal=0,
        ),
        Card(
            tag="Residential burden", cite=Cite(raw="Example 2023", year=2023),
            body="Residential burden increases when rates rise. " * 8,
            read_text="Residential burden increases when rates rise.", source_id="fixture",
            source_path="fixture.docx", ordinal=1,
        ),
    ])
    engine = SearchEngine(store, cache_path=str(tmp_path / "index.pkl"))
    engine.build()

    exact = engine.search("household costs", k=2)
    assert exact[0].match_type == "exact"
    assert exact[0].confidence == "high"
    assert exact[0].to_dict()["confidence"] == "high"

    exploratory = engine.search("residential burden", k=2)
    assert exploratory[0].match_type == "exact"
    assert exploratory[0].confidence in {"high", "medium"}
    store.close()


def test_smart_query_variants_are_bounded_and_empty_safe():
    assert smart_query_variants("   ") == []
    assert len(smart_query_variants("show cards about bans, emissions, and nuclear", limit=2)) == 2
