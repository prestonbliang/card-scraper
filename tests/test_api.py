"""HTTP-level regression tests for the local Card Scraper API."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from cardgraph.api.main import create_app
from cardgraph.index.store import Store
from cardgraph.ingest.base import Acquired
from cardgraph.models import Card, Cite, NodeKind, OutlineNode, Source


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


def test_restart_marks_interrupted_ingest_jobs_failed(tmp_path):
    db = tmp_path / "restart.db"
    store = Store(str(db))
    store.save_ingest_job({
        "job_id": "stuck", "url": "https://example.net/cases.zip",
        "license": "fixture", "state": "importing", "processed": 1,
        "total": 3, "imported": 0, "cards_added": 0, "skipped": 0,
        "failed": 0, "details": [], "error": None, "created_at": time.time(),
    })
    store.close()

    with TestClient(create_app(str(db))) as client:
        jobs = client.get("/api/ingest").json()["jobs"]
        assert jobs[0]["job_id"] == "stuck"
        assert jobs[0]["state"] == "failed"
        assert jobs[0]["error"] == "application stopped before this import completed"


def test_api_validates_search_inputs_and_year_ranges(tmp_path):
    with TestClient(_app(tmp_path)) as client:
        response = client.post("/api/ingest", json={"url": "https://example.net/cases.zip"})
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        for _ in range(20):
            status = client.get(f"/api/ingest/{job_id}").json()
            if status["state"] == "failed":
                break
            time.sleep(0.01)
        assert status["state"] == "failed"
        assert "allowlist" in status["error"]
        assert client.post("/api/ingest", json={"url": "https://openev.debatecoaches.org/", "limit": 0}).status_code == 422
        assert client.get("/api/search?q=grid&k=0").status_code == 422
        assert client.get("/api/search?q=grid&side=maybe").status_code == 422
        assert client.get("/api/search?q=grid&year_min=2027&year_max=2020").status_code == 422
        assert client.get("/api/tree?limit=0").status_code == 422

    with TestClient(_app(tmp_path)) as restarted:
        persisted = restarted.get(f"/api/ingest/{job_id}")
        assert persisted.status_code == 200
        assert persisted.json()["state"] == "failed"


def test_browser_ingest_uses_the_same_store_and_returns_counts(tmp_path, monkeypatch):
    import cardgraph.api.main as api_main

    source = Source(
        source_id="public-import", path=str(tmp_path / "public.docx"),
        title="Public import", origin="online", license="fixture terms",
        url="https://openev.debatecoaches.org/public.docx",
    )
    card = Card(
        tag="Imported household costs",
        cite=Cite(raw="Example 2026", author="Example", year=2026),
        body="Imported household costs. " * 8,
        read_text="Imported household costs.", source_id=source.source_id,
        source_path=source.path, ordinal=0,
    )

    class FakeAdapter:
        def __init__(self, **_kwargs):
            pass

        def acquire(self, limit=None):
            assert limit == 25
            return [Acquired(path=source.path, source=source)]

    monkeypatch.setattr(api_main, "OnlineEvidenceAdapter", FakeAdapter)
    monkeypatch.setattr(
        api_main, "parse_any",
        lambda path, source_id: ([card], OutlineNode(title="Public", kind=NodeKind.POCKET, source_id=source_id), object()),
    )
    with TestClient(_app(tmp_path)) as client:
        response = client.post("/api/ingest", json={"url": source.url})
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        for _ in range(50):
            payload = client.get(f"/api/ingest/{job_id}").json()
            if payload["state"] == "completed":
                break
            time.sleep(0.01)
        assert payload["state"] == "completed"
        assert payload["imported"] == 1
        assert payload["cards_added"] == 1
        assert payload["stats"]["cards"] == 2

        refreshed = Card(
            tag="Imported transmission burden",
            cite=Cite(raw="Example 2027", author="Example", year=2027),
            body="Imported transmission burden. " * 8,
            read_text="Imported transmission burden.", source_id=source.source_id,
            source_path=source.path, ordinal=0,
        )
        monkeypatch.setattr(
            api_main, "parse_any",
            lambda path, source_id: ([refreshed], OutlineNode(
                title="Public", kind=NodeKind.POCKET, source_id=source_id), object()),
        )
        refresh = client.post("/api/sources/public-import/refresh")
        assert refresh.status_code == 202
        refresh_id = refresh.json()["job_id"]
        for _ in range(50):
            refresh_status = client.get(f"/api/ingest/{refresh_id}").json()
            if refresh_status["state"] == "completed":
                break
            time.sleep(0.01)
        assert refresh_status["state"] == "completed"
        assert client.get("/api/search?q=transmission+burden").json()["count"] == 1
        managed = {item["source_id"]: item for item in client.get("/api/sources").json()["sources"]}
        assert managed["public-import"]["card_count"] == 1


def test_source_management_removes_only_that_source_and_lists_job_history(tmp_path):
    with TestClient(_app(tmp_path)) as client:
        sources = client.get("/api/sources").json()["sources"]
        assert sources[0]["source_id"] == "public"
        history = client.get("/api/ingest").json()
        assert history["jobs"] == []
        removed = client.delete("/api/sources/public")
        assert removed.status_code == 200
        assert removed.json()["stats"]["cards"] == 0
        assert client.delete("/api/sources/public").status_code == 404


def test_source_health_marks_stale_and_failed_refresh_preserves_cards(tmp_path, monkeypatch):
    import cardgraph.api.main as api_main

    db = tmp_path / "health.db"
    seed = Store(str(db))
    old = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat(timespec="seconds")
    seed.add_source(Source(
        source_id="stale", path="/corpus/stale.docx", title="Stale release",
        origin="online", license="fixture", fetched_at=old,
        url="https://openev.debatecoaches.org/stale.docx", card_count=1,
    ))
    seed.add_cards([Card(
        tag="Known good card", cite=Cite(raw="Example 2026"),
        body="Known good card. " * 8, read_text="Known good card.",
        source_id="stale", source_path="/corpus/stale.docx", ordinal=0,
    )])
    seed.close()

    class BrokenAdapter:
        def __init__(self, **_kwargs):
            pass

        def acquire(self, limit=None):
            return [Acquired(path=str(tmp_path / "stale.docx"), source=Source(
                source_id="stale", path=str(tmp_path / "stale.docx"),
                title="Stale release", origin="online", license="fixture",
                fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                url="https://openev.debatecoaches.org/stale.docx"))]

    monkeypatch.setattr(api_main, "OnlineEvidenceAdapter", BrokenAdapter)
    monkeypatch.setattr(api_main, "parse_any", lambda *_args: (_ for _ in ()).throw(RuntimeError("bad document")))
    with TestClient(api_main.create_app(str(db))) as client:
        listed = client.get("/api/sources").json()["sources"][0]
        assert listed["health"]["state"] == "stale"
        queued = client.post("/api/sources/stale/refresh")
        assert queued.status_code == 202
        for _ in range(50):
            status = client.get(f"/api/ingest/{queued.json()['job_id']}").json()
            if status["state"] == "failed":
                break
            time.sleep(0.01)
        assert status["state"] == "failed"
        health = client.get("/api/sources").json()["sources"][0]["health"]
        assert health["state"] == "refresh-failed"
        assert client.get("/api/search?q=known+good").json()["count"] == 1


def test_empty_refresh_preserves_last_known_good_source(tmp_path, monkeypatch):
    import cardgraph.api.main as api_main

    db = tmp_path / "empty-refresh.db"
    seed = Store(str(db))
    seed.add_source(Source(
        source_id="release", path="release.docx", title="Release",
        origin="online", license="fixture", url="https://openev.debatecoaches.org/release/",
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), card_count=1,
    ))
    seed.add_cards([Card(
        tag="Known good", cite=Cite(raw="Example 2026"),
        body="Known good evidence. " * 8, read_text="Known good evidence.",
        source_id="release", source_path="release.docx", ordinal=0,
    )])
    seed.close()

    class EmptyAdapter:
        def __init__(self, **_kwargs):
            pass

        def acquire(self, limit=None):
            return []

    monkeypatch.setattr(api_main, "OnlineEvidenceAdapter", EmptyAdapter)
    with TestClient(api_main.create_app(str(db))) as client:
        queued = client.post("/api/sources/release/refresh")
        for _ in range(50):
            status = client.get(f"/api/ingest/{queued.json()['job_id']}").json()
            if status["state"] == "failed":
                break
            time.sleep(0.01)
        assert status["state"] == "failed"
        assert status["error"] == "refresh failed; last-known-good evidence was retained"
        assert client.get("/api/search?q=known+good").json()["count"] == 1
        health = client.get("/api/sources").json()["sources"][0]["health"]
        assert health["state"] == "refresh-failed"


def test_multi_file_refresh_is_atomic(tmp_path, monkeypatch):
    import cardgraph.api.main as api_main

    db = tmp_path / "atomic.db"
    seed = Store(str(db))
    seed.add_source(Source(
        source_id="release-old", path="old.docx", title="Release old",
        origin="online", license="fixture", url="https://openev.debatecoaches.org/release/",
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), card_count=1,
    ))
    seed.add_cards([Card(
        tag="Old evidence", cite=Cite(raw="Old 2025"), body="Old evidence. " * 8,
        read_text="Old evidence.", source_id="release-old", source_path="old.docx", ordinal=0,
    )])
    seed.close()

    first_source = Source(
        source_id="release-new-one", path="one.docx", title="Release one",
        origin="online", license="fixture", url="https://openev.debatecoaches.org/one.docx",
    )
    second_source = Source(
        source_id="release-new-two", path="two.docx", title="Release two",
        origin="online", license="fixture", url="https://openev.debatecoaches.org/two.docx",
    )

    class MultiAdapter:
        def __init__(self, **_kwargs):
            pass

        def acquire(self, limit=None):
            return [Acquired(path="one.docx", source=first_source),
                    Acquired(path="two.docx", source=second_source)]

    monkeypatch.setattr(api_main, "OnlineEvidenceAdapter", MultiAdapter)
    calls = {"count": 0, "fail_second": True}
    first_card = Card(tag="New first", cite=Cite(raw="New 2026"),
                      body="New first. " * 8, read_text="New first.",
                      source_id=first_source.source_id, source_path="one.docx", ordinal=0)
    second_card = Card(tag="New second", cite=Cite(raw="New 2026"),
                       body="New second. " * 8, read_text="New second.",
                       source_id=second_source.source_id, source_path="two.docx", ordinal=0)

    def parse(path, source_id):
        calls["count"] += 1
        if calls["fail_second"] and calls["count"] == 2:
            raise RuntimeError("second file is corrupt")
        card = first_card if source_id == first_source.source_id else second_card
        return [card], OutlineNode(title=source_id, kind=NodeKind.POCKET, source_id=source_id), object()

    monkeypatch.setattr(api_main, "parse_any", parse)
    with TestClient(api_main.create_app(str(db))) as client:
        queued = client.post("/api/sources/release-old/refresh")
        for _ in range(50):
            status = client.get(f"/api/ingest/{queued.json()['job_id']}").json()
            if status["state"] == "failed":
                break
            time.sleep(0.01)
        assert status["state"] == "failed"
        assert client.get("/api/search?q=old+evidence").json()["count"] == 1
        assert client.get("/api/search?q=new+first").json()["count"] == 0

        calls["count"] = 0
        calls["fail_second"] = False
        success = client.post("/api/sources/release-old/refresh")
        for _ in range(50):
            status = client.get(f"/api/ingest/{success.json()['job_id']}").json()
            if status["state"] == "completed":
                break
            time.sleep(0.01)
        assert status["state"] == "completed"
        assert client.get("/api/search?q=old+evidence").json()["count"] == 0
        assert client.get("/api/search?q=new+first").json()["count"] == 1
        assert client.get("/api/search?q=new+second").json()["count"] == 1


def test_refresh_after_failure_does_not_coalesce_onto_dead_job(tmp_path, monkeypatch):
    """A retry requested right after a failed refresh must start a new job.

    The failed worker lingers in the coalescing map while it finishes durable
    bookkeeping; queuing behind that terminal job would report its old failure
    instead of running the retry (CI-only timing made this visible).
    """
    import cardgraph.api.main as api_main

    db = tmp_path / "retry.db"
    seed = Store(str(db))
    seed.add_source(Source(
        source_id="release-old", path="old.docx", title="Release old",
        origin="online", license="fixture", url="https://openev.debatecoaches.org/release/",
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), card_count=1,
    ))
    seed.add_cards([Card(
        tag="Old evidence", cite=Cite(raw="Old 2025"), body="Old evidence. " * 8,
        read_text="Old evidence.", source_id="release-old", source_path="old.docx", ordinal=0,
    )])
    seed.close()

    retry_source = Source(
        source_id="release-retry", path="retry.docx", title="Release retry",
        origin="online", license="fixture", url="https://openev.debatecoaches.org/retry.docx",
    )

    class RetryAdapter:
        def __init__(self, **_kwargs):
            pass

        def acquire(self, limit=None):
            time.sleep(0.002)  # widen the terminal-job window like CI disks do
            return [Acquired(path="retry.docx", source=retry_source)]

    monkeypatch.setattr(api_main, "OnlineEvidenceAdapter", RetryAdapter)
    calls = {"count": 0, "fail_first": True}
    retry_card = Card(tag="New retry", cite=Cite(raw="Retry 2026"),
                      body="New retry. " * 8, read_text="New retry.",
                      source_id=retry_source.source_id, source_path="retry.docx", ordinal=0)

    def parse(path, source_id):
        calls["count"] += 1
        if calls["fail_first"] and calls["count"] == 1:
            raise RuntimeError("first file is corrupt")
        return [retry_card], OutlineNode(
            title=source_id, kind=NodeKind.POCKET, source_id=source_id), object()

    monkeypatch.setattr(api_main, "parse_any", parse)
    with TestClient(api_main.create_app(str(db))) as client:
        first = client.post("/api/sources/release-old/refresh").json()
        for _ in range(100):
            status = client.get(f"/api/ingest/{first['job_id']}").json()
            if status["state"] in {"failed", "completed"}:
                break
            time.sleep(0.01)
        assert status["state"] == "failed"

        calls["fail_first"] = False
        retry = client.post("/api/sources/release-old/refresh").json()
        assert retry["job_id"] != first["job_id"], (
            "retry was coalesced onto the already-failed job")
        for _ in range(100):
            status = client.get(f"/api/ingest/{retry['job_id']}").json()
            if status["state"] in {"failed", "completed"}:
                break
            time.sleep(0.01)
        assert status["state"] == "completed", status
        assert client.get("/api/search?q=new+retry").json()["count"] == 1


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
        assert hit["evidence_status"] == "traceable"
        assert hit["read_ratio"] > 0

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
        assert payload["hits"][0]["evidence_status"] == "traceable"

        overridden = client.get("/api/smart-search", params={
            "q": "negative cards about grid", "side": "aff",
        }).json()
        assert overridden["interpreted_filters"]["side"] == "aff"

        assert client.get(
            "/api/smart-search?q=grid&year_min=2027&year_max=2020"
        ).status_code == 422


def test_health_reports_ready_index_and_corrupt_analysis_is_recoverable(tmp_path):
    with TestClient(_app(tmp_path)) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["search_index"] == "ready"
        assert health.json()["stats"]["cards"] == 1

        (tmp_path / "analysis.json").write_text("{not-json", encoding="utf-8")
        report = client.get("/api/analysis")
        assert report.status_code == 200
        assert report.json()["available"] is False
        assert "unreadable" in report.json()["error"]


def test_unexpected_worker_failure_is_reported_instead_of_sticking(tmp_path, monkeypatch):
    import cardgraph.api.main as api_main

    source = Source(
        source_id="replacement", path="replacement.docx", title="Replacement",
        origin="online", license="fixture", url="https://openev.debatecoaches.org/replacement/",
    )

    class Adapter:
        def __init__(self, **_kwargs):
            pass

        def acquire(self, limit=None):
            return [Acquired(path=source.path, source=source)]

    card = Card(
        tag="Replacement card", cite=Cite(raw="Example 2027"),
        body="Replacement card. " * 8, read_text="Replacement card.",
        source_id=source.source_id, source_path=source.path, ordinal=0,
    )
    monkeypatch.setattr(api_main, "OnlineEvidenceAdapter", Adapter)
    monkeypatch.setattr(
        api_main, "parse_any",
        lambda *_args: ([card], OutlineNode(
            title="Replacement", kind=NodeKind.POCKET, source_id=source.source_id), object()),
    )
    original_replace = Store.replace_source
    monkeypatch.setattr(
        Store, "replace_source",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("disk transaction failed")),
    )
    with TestClient(_app(tmp_path)) as client:
        queued = client.post("/api/sources/public/refresh")
        job_id = queued.json()["job_id"]
        for _ in range(50):
            status = client.get(f"/api/ingest/{job_id}").json()
            if status["state"] == "failed":
                break
            time.sleep(0.01)
        assert status["state"] == "failed"
        assert status["error"] == "import worker failed: disk transaction failed"
        persisted = client.get(f"/api/ingest/{job_id}").json()
        assert persisted["state"] == "failed"
    monkeypatch.setattr(Store, "replace_source", original_replace)
