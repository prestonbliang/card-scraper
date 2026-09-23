"""Regression tests for research packet export."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from cardgraph.api.main import create_app
from cardgraph.index.store import Store
from cardgraph.models import Card, Cite, Source, Side
from cardgraph.packet import (PacketError, bibtex_bibliography, build_packet,
                              collect_cards, csv_sheet, markdown_brief,
                              provenance_manifest, write_packet)


def _seed(tmp_path, *, fetched_at="2026-09-01T00:00:00+00:00", origin="online"):
    db = str(tmp_path / "packet.db")
    store = Store(db)
    store.add_source(Source(
        source_id="pub", path="/corpus/p.docx", title="Public release",
        origin=origin, license="fixture terms",
        url="https://ev.example/p.docx", card_count=3, fetched_at=fetched_at,
    ))
    aff = Card(tag="Grid costs hurt households",
               cite=Cite(raw="Nguyen 2025", author="Nguyen", year=2025),
               body="Grid costs hurt households. " * 8,
               read_text="Grid costs hurt households.",
               source_id="pub", ordinal=0, side=Side.AFF)
    neg = Card(tag="Renewables now cheaper",
               cite=Cite(raw="Ortiz 2026", author="Ortiz", year=2026),
               body="Renewables undercut coal on price. " * 8,
               read_text="Renewables undercut coal.",
               source_id="pub", ordinal=1, side=Side.NEG)
    bare = Card(tag="Placeholder without metadata",
                cite=Cite(raw=""), body="unmarked body", read_text="",
                source_id="pub", ordinal=2)
    store.add_cards([aff, neg, bare])
    return store, aff, neg, bare


def test_packet_by_ids_groups_sides_and_records_provenance(tmp_path):
    store, aff, neg, _bare = _seed(tmp_path)
    try:
        filename, payload, manifest = build_packet(
            store, card_ids=[aff.card_id, neg.card_id])
        assert filename.startswith("card-scraper-packet-") and filename.endswith(".zip")
        bundle = zipfile.ZipFile(io.BytesIO(payload))
        assert set(bundle.namelist()) == {
            "README.md", "bibliography.bib", "cards.csv", "manifest.json"}
        assert manifest["card_count"] == 2
        assert manifest["cards"] == [aff.card_id, neg.card_id]
        assert manifest["sources"]["pub"]["url"] == "https://ev.example/p.docx"
        assert manifest["sources"]["pub"]["freshness"] in {"fresh", "stale"}
        assert manifest["evidence_status_counts"] == {"traceable": 2}

        brief = bundle.read("README.md").decode()
        assert "## Affirmative" in brief and "## Negative" in brief
        assert "Nguyen 2025" in brief and "https://ev.example/p.docx" in brief
        assert "retrieval metadata is not a factual verdict" in brief

        bibliography = bundle.read("bibliography.bib").decode()
        assert "@misc{nguyen2025," in bibliography
        assert "@misc{ortiz2026," in bibliography

        rows = bundle.read("cards.csv").decode().strip().splitlines()
        assert len(rows) == 3
        assert rows[0].startswith("card_id,side,tag,")
    finally:
        store.close()


def test_packet_by_query_reuses_search_filters(tmp_path):
    store, aff, _neg, _bare = _seed(tmp_path)
    try:
        cards = collect_cards(store, query="grid costs", k=5)
        assert [c.card_id for c in cards] == [aff.card_id]
        assert cards[0].evidence_status == "traceable"
        assert cards[0].source_freshness in {"fresh", "stale"}

        filename, payload, manifest = build_packet(store, query="grid costs", k=5)
        assert manifest["card_count"] == 1
    finally:
        store.close()


def test_packet_rejects_missing_ids_and_conflicting_input(tmp_path):
    store, aff, _neg, _bare = _seed(tmp_path)
    try:
        with pytest.raises(PacketError):
            build_packet(store, card_ids=["nonexistent"])
        with pytest.raises(PacketError):
            collect_cards(store, card_ids=[aff.card_id], query="grid")
        with pytest.raises(PacketError):
            collect_cards(store, query="   ")
        with pytest.raises(PacketError):
            collect_cards(store, card_ids=[" "])
    finally:
        store.close()


def test_packet_bibtex_disambiguates_and_flags_missing_metadata(tmp_path):
    cards = [
        type("C", (), {"cite_author": "Nguyen", "cite_year": 2025,
                       "cite_pub": None, "cite_url": None, "source_url": None,
                       "cite_raw": "Nguyen 2025", "tag": "One",
                       "source_title": "S", "source_id": "s"})(),
        type("C", (), {"cite_author": "Nguyen", "cite_year": 2025,
                       "cite_pub": None, "cite_url": None, "source_url": None,
                       "cite_raw": "Nguyen 2025", "tag": "Two",
                       "source_title": "S", "source_id": "s"})(),
        type("C", (), {"cite_author": None, "cite_year": None,
                       "cite_pub": None, "cite_url": None, "source_url": None,
                       "cite_raw": "", "tag": "Mystery",
                       "source_title": "S", "source_id": "s"})(),
    ]
    text = bibtex_bibliography(cards)  # type: ignore[arg-type]
    assert "@misc{nguyen2025," in text
    assert "@misc{nguyen2025a," in text
    assert "@misc{unknownnodate," in text
    # every field line except the last ends with a comma
    for entry in text.split("@misc{")[1:]:
        body = entry.split("}")[0]
        field_lines = [line for line in body.splitlines()
                       if line.strip() and "=" in line]
        assert all(line.rstrip().endswith(",")
                   for line in field_lines[:-1])
        assert not field_lines[-1].rstrip().endswith(",")


def test_packet_marks_stale_and_needs_review_cards(tmp_path):
    old = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    store, _aff, _neg, bare = _seed(tmp_path, fetched_at=old)
    try:
        cards = collect_cards(store, card_ids=[bare.card_id])
        assert cards[0].evidence_status == "needs-review"
        assert cards[0].source_freshness == "stale"
        manifest = provenance_manifest(cards)
        assert manifest["evidence_status_counts"] == {"needs-review": 1}
    finally:
        store.close()


def test_packet_write_is_atomic_and_reports_manifest(tmp_path):
    store, aff, _neg, _bare = _seed(tmp_path)
    try:
        out = tmp_path / "out" / "packet.zip"
        filename, manifest = write_packet(
            store, str(out), card_ids=[aff.card_id])
        assert out.exists()
        with zipfile.ZipFile(out) as bundle:
            assert set(bundle.namelist()) == {
                "README.md", "bibliography.bib", "cards.csv", "manifest.json"}
        assert manifest["card_count"] == 1
        assert json.loads(zipfile.ZipFile(out).read("manifest.json"))["card_count"] == 1
        assert filename in str(filename)
    finally:
        store.close()


def test_packet_markdown_and_csv_handles_unclassified(tmp_path):
    store, _aff, _neg, bare = _seed(tmp_path)
    try:
        cards = collect_cards(store, card_ids=[bare.card_id])
        brief = markdown_brief(cards)
        assert "## Unclassified" in brief
        assert "no read text was parsed" in brief
        sheet = csv_sheet(cards)
        assert sheet.splitlines()[1].startswith(f"{bare.card_id},unknown,")
    finally:
        store.close()


def test_api_packet_by_ids_and_query(tmp_path):
    db = str(tmp_path / "api.db")
    store, aff, neg, _bare = _seed(tmp_path)
    store.close()
    (tmp_path / "unused.db").write_bytes(b"")
    import os
    os.replace(str(tmp_path / "packet.db"), db)

    with TestClient(create_app(db)) as client:
        response = client.post("/api/packet",
                               json={"card_ids": [aff.card_id, neg.card_id]})
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"
        assert "attachment" in response.headers.get("content-disposition", "")
        bundle = zipfile.ZipFile(io.BytesIO(response.content))
        assert json.loads(bundle.read("manifest.json"))["card_count"] == 2

        response = client.post("/api/packet",
                               json={"query": "grid costs", "k": 3})
        assert response.status_code == 200
        assert json.loads(
            zipfile.ZipFile(io.BytesIO(response.content)).read("manifest.json")
        )["card_count"] == 1

        missing = client.post("/api/packet", json={"card_ids": ["nope"]})
        assert missing.status_code == 400
        assert "not in the index" in missing.json()["detail"]

        conflict = client.post(
            "/api/packet", json={"card_ids": [aff.card_id], "query": "x"})
        assert conflict.status_code == 422

        years = client.post(
            "/api/packet",
            json={"query": "grid", "year_min": 2026, "year_max": 2020})
        assert years.status_code == 422
