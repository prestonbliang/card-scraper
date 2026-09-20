"""Portable workspace bundle behavior and safety regressions."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cardgraph.api.main import create_app
from cardgraph.cli import main as cli_main
from cardgraph.index.store import Store
from cardgraph.models import Card, Cite, Source
from cardgraph.workspace import (WorkspaceBundleError, export_bundle, inspect_bundle,
                                  restore_bundle)


def _seed(db: Path, source_file: Path | None = None) -> None:
    store = Store(str(db))
    path = str(source_file) if source_file else "missing.docx"
    store.add_source(Source(
        source_id="fixture", path=path, title="Fixture release", origin="local",
        license="fixture terms", url="https://example.test/fixture", card_count=1,
    ))
    store.add_cards([Card(
        tag="Households pay grid costs", cite=Cite(raw="Fixture Author 2026"),
        body="Households pay grid costs. " * 8,
        read_text="Households pay grid costs.", source_id="fixture",
        source_path=path, ordinal=0,
    )])
    store.close()


def test_bundle_round_trip_relocates_corpus_and_browser_state(tmp_path):
    db = tmp_path / "source.db"
    source = tmp_path / "corpus" / "fixture.docx"
    source.parent.mkdir()
    source.write_bytes(b"source fixture bytes")
    _seed(db, source)

    bundle = tmp_path / "workspace.zip"
    manifest = export_bundle(
        str(db), str(bundle),
        browser_state={
            "pinned": {"card-1": {"card_id": "card-1", "note": "compare"}},
            "pinned_order": ["card-1"],
            "recent_searches": ["grid costs"],
        },
    )
    assert manifest["includes_corpus"] is True
    assert bundle.exists()
    inspection = inspect_bundle(str(bundle))
    assert inspection.stats["cards"] == 1
    assert inspection.dropped_pins == 1
    assert inspection.browser_state["pinned"] == {}
    assert inspection.browser_state["recent_searches"] == ["grid costs"]

    restored = restore_bundle(str(tmp_path / "restored.db"), str(bundle))
    assert restored.imported_files == 1
    assert restored.browser_state["recent_searches"] == ["grid costs"]
    assert restored.stats["cards"] == 1
    restored_store = Store(str(tmp_path / "restored.db"))
    imported_path = restored_store.conn.execute(
        "SELECT path FROM sources WHERE source_id='fixture'"
    ).fetchone()[0]
    assert Path(imported_path).read_bytes() == b"source fixture bytes"
    assert restored_store.conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0] == 1
    restored_store.close()


def test_bundle_without_corpus_is_explicit_and_restores_index(tmp_path):
    db = tmp_path / "source.db"
    _seed(db)
    bundle = tmp_path / "metadata-only.zip"
    manifest = export_bundle(str(db), str(bundle), include_corpus=False)
    assert manifest["sources"][0]["included"] is False
    restored = restore_bundle(str(tmp_path / "restored.db"), str(bundle))
    assert restored.imported_files == 0
    assert restored.stats["cards"] == 1


def test_inspection_does_not_modify_an_existing_database(tmp_path):
    db = tmp_path / "source.db"
    _seed(db)
    bundle = tmp_path / "workspace.zip"
    export_bundle(str(db), str(bundle), include_corpus=False)
    before = Store(str(db)).stats()
    inspection = inspect_bundle(str(bundle))
    after_store = Store(str(db))
    assert inspection.stats == {"sources": 1, "cards": 1, "nodes": 0, "edges": 0}
    assert after_store.stats() == before
    after_store.close()


def test_tampered_bundle_is_rejected_before_live_database_changes(tmp_path):
    db = tmp_path / "source.db"
    _seed(db)
    bundle = tmp_path / "workspace.zip"
    export_bundle(str(db), str(bundle), include_corpus=False)
    with zipfile.ZipFile(bundle) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members["manifest.json"])
    manifest["database"]["sha256"] = "0" * 64
    members["manifest.json"] = json.dumps(manifest).encode()
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)

    with pytest.raises(WorkspaceBundleError, match="integrity check failed"):
        restore_bundle(str(db), str(bad))
    store = Store(str(db))
    assert store.stats()["cards"] == 1
    store.close()


def test_workspace_api_exports_and_restores_browser_state(tmp_path):
    source_db = tmp_path / "source.db"
    _seed(source_db)
    target_db = tmp_path / "target.db"
    with TestClient(create_app(str(source_db))) as source_client:
        response = source_client.post("/api/workspace/export", json={
            "include_corpus": False,
            "browser_state": {
                "pinned": {"card-1": {"card_id": "card-1"}},
                "pinned_order": ["card-1"],
                "recent_searches": ["households"],
            },
        })
        assert response.status_code == 200
        bundle = response.content
        assert response.headers["content-type"].startswith("application/zip")

    with TestClient(create_app(str(target_db))) as target_client:
        inspected = target_client.post(
            "/api/workspace/inspect", content=bundle,
            headers={"content-type": "application/zip"},
        )
        assert inspected.status_code == 200
        assert inspected.json()["stats"]["cards"] == 1
        assert inspected.json()["dropped_pins"] == 1
        response = target_client.post(
            "/api/workspace/restore", content=bundle,
            headers={"content-type": "application/zip"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["stats"]["cards"] == 1
        assert payload["browser_state"]["recent_searches"] == ["households"]
        assert payload["dropped_pins"] == 1
        assert target_client.get("/api/search?q=households").json()["count"] == 1


def test_workspace_cli_exports_and_imports_with_explicit_db(tmp_path, capsys):
    source_db = tmp_path / "source.db"
    _seed(source_db)
    bundle = tmp_path / "cli.zip"
    assert cli_main([
        "--db", str(source_db), "workspace", "export", str(bundle), "--no-corpus",
    ]) == 0
    assert "exported" in capsys.readouterr().out

    target_db = tmp_path / "target.db"
    assert cli_main([
        "--db", str(target_db), "workspace", "import", str(bundle),
    ]) == 0
    assert "restored" in capsys.readouterr().out
    store = Store(str(target_db))
    assert store.stats()["cards"] == 1
    store.close()


def test_workspace_api_rejects_invalid_upload(tmp_path):
    db = tmp_path / "source.db"
    _seed(db)
    with TestClient(create_app(str(db))) as client:
        response = client.post(
            "/api/workspace/restore", content=b"not a zip",
            headers={"content-type": "application/zip"},
        )
        assert response.status_code == 400
        assert "could not open bundle" in response.json()["detail"]
        assert client.get("/api/stats").json()["cards"] == 1
