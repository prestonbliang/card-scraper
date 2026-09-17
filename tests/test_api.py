"""HTTP-level regression tests for the local Card Scraper API."""

from __future__ import annotations

from fastapi.testclient import TestClient

from cardgraph.api.main import create_app
from cardgraph.index.store import Store
from cardgraph.models import Card, Cite, Source


def _app(tmp_path):
    db = str(tmp_path / "cards.db")
    store = Store(db)
    store.add_source(Source(
        source_id="public", path="/corpus/public.docx", title="Public release",
        origin="online", license="fixture terms",
        url="https://openev.debatecoaches.org/releases/public.docx",
        card_count=1,
    ))
    store.add_cards([Card(
        tag="Grid costs hurt households",
        cite=Cite(raw="Example 2026", author="Example", year=2026),
        body="Grid costs hurt households. " * 8,
        read_text="Grid costs hurt households.",
        source_id="public", source_path="/corpus/public.docx", ordinal=0,
    )])
    store.close()
    return create_app(db)


def test_api_validates_search_inputs_and_year_ranges(tmp_path):
    with TestClient(_app(tmp_path)) as client:
        assert client.get("/api/search?q=grid&k=0").status_code == 422
        assert client.get("/api/search?q=grid&side=maybe").status_code == 422
        assert client.get("/api/search?q=grid&year_min=2027&year_max=2020").status_code == 422
        assert client.get("/api/tree?limit=0").status_code == 422


def test_api_search_and_card_preserve_provenance(tmp_path):
    with TestClient(_app(tmp_path)) as client:
        result = client.get("/api/search", params={"q": "grid costs"})
        assert result.status_code == 200
        hit = result.json()["hits"][0]
        assert hit["source_title"] == "Public release"
        assert hit["source_origin"] == "online"
        assert hit["source_url"].startswith("https://openev")
        assert any(reason.startswith("exact:") for reason in hit["match_reasons"])
        assert any("lexical rank" in reason for reason in hit["match_reasons"])
        assert hit["confidence"] in {"high", "medium", "exploratory"}
        assert hit["match_type"] in {"exact", "hybrid", "lexical", "semantic"}

        detail = client.get(f"/api/card/{hit['card_id']}")
        assert detail.status_code == 200
        assert detail.json()["source_license"] == "fixture terms"

        catalog = client.get("/api/catalog")
        assert catalog.status_code == 200
        assert any(s["id"] == "debate-central" for s in catalog.json()["sources"])


def test_api_smart_search_returns_transparent_variants_and_provenance(tmp_path):
    with TestClient(_app(tmp_path)) as client:
        result = client.get("/api/smart-search", params={
            "q": "Please find cards about household costs",
        })
        assert result.status_code == 200
        payload = result.json()
        assert payload["interpreted_query"] == "household costs"
        assert any("household" in variant.lower() for variant in payload["variants"])
        assert payload["hits"][0]["source_title"] == "Public release"
        assert payload["hits"][0]["matched_queries"]
        assert payload["hits"][0]["match_reasons"]
        assert payload["hits"][0]["confidence"] in {"medium", "exploratory"}
        assert payload["hits"][0]["match_type"] == "semantic"

        overridden = client.get("/api/smart-search", params={
            "q": "negative cards about grid", "side": "aff",
        }).json()
        assert overridden["interpreted_filters"]["side"] == "aff"

        assert client.get(
            "/api/smart-search?q=grid&year_min=2027&year_max=2020"
        ).status_code == 422
