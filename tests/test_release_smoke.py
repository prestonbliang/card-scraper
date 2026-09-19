"""Release smoke test for the public application boundary."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from cardgraph.api.main import create_app
from cardgraph.index.store import Store
from cardgraph.models import Card, Cite, Source


def test_local_application_shell_searches_and_reports_source_health(tmp_path):
    db = tmp_path / "release-smoke.db"
    store = Store(str(db))
    store.add_source(Source(
        source_id="smoke-source", path="smoke.docx", title="Smoke release",
        origin="online", license="fixture", url="https://example.test/smoke.docx",
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        card_count=1,
    ))
    store.add_cards([Card(
        tag="Clean energy reduces grid costs",
        cite=Cite(raw="Smoke Author 2026", author="Smoke Author", year=2026),
        body="Clean energy reduces grid costs. " * 6,
        read_text="Clean energy reduces grid costs.",
        source_id="smoke-source", source_path="smoke.docx", ordinal=0,
    )])
    store.close()

    with TestClient(create_app(str(db))) as client:
        shell = client.get("/")
        assert shell.status_code == 200
        assert "Card Scraper" in shell.text

        result = client.get("/api/search", params={"q": "grid costs"})
        assert result.status_code == 200
        assert result.json()["count"] == 1
        assert result.json()["hits"][0]["source_freshness"] == "fresh"

        source = client.get("/api/sources").json()["sources"][0]
        assert source["health"]["state"] == "fresh"
        assert client.get("/api/stats").json()["cards"] == 1
