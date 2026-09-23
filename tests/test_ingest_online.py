"""Tests for reviewed source discovery and conservative PDF ingestion."""

from __future__ import annotations

import io
import sys
import types
import zipfile

import pytest

from cardgraph.ingest.base import HttpIndexAdapter
from cardgraph.ingest.fetch import FetchResult, Fetcher
from cardgraph.ingest.policy import AccessPolicy, AccessRefused, source_catalog


class FakeFetcher:
    def __init__(self, payloads: dict[str, bytes | str]):
        self.payloads = payloads

    def get(self, url: str) -> FetchResult:
        payload = self.payloads[url]
        if isinstance(payload, bytes):
            return FetchResult(url=url, status=200, content=payload)
        return FetchResult(url=url, status=200, text=payload)

    def download(self, url: str, dest: str) -> str:
        from pathlib import Path

        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            payload = self.payloads[url]
            fh.write(payload if isinstance(payload, bytes) else payload.encode())
        return dest


def adapter(tmp_path, index_url="https://openev.debatecoaches.org/releases/"):
    return HttpIndexAdapter(
        index_url=index_url,
        license="public-test-fixture",
        workdir=str(tmp_path),
        policy=AccessPolicy(respect_robots=False),
    )


def test_catalog_distinguishes_public_attributed_and_gated_sources():
    catalog = {source["id"]: source for source in source_catalog()}
    assert catalog["debate-central"]["access"] == "attributed"
    assert catalog["debateus"]["access"] == "attributed"
    assert catalog["opencaselist"]["access"] == "gated"
    assert "Not fetched" in catalog["opencaselist"]["note"]


def test_discover_resolves_supported_links_and_ignores_navigation(tmp_path):
    index = """<html><body>
      <a href="files/affirmative.docx?season=2026">Aff</a>
      <a href="files/negative.zip#download">Neg</a>
      <a href="files/topic.pdf">PDF</a>
      <a href="about.html">About</a>
      <a href="https://example.net/private.docx">wrong host is still a link</a>
    </body></html>"""
    a = adapter(tmp_path)
    a.fetcher = FakeFetcher({a.index_url: index})

    assert a.discover() == [
        "https://openev.debatecoaches.org/releases/files/affirmative.docx?season=2026",
        "https://openev.debatecoaches.org/releases/files/negative.zip#download",
        "https://openev.debatecoaches.org/releases/files/topic.pdf",
        "https://openev.debatecoaches.org/releases/about.html",
    ]


def test_direct_file_url_is_discoverable_without_fetching_index(tmp_path):
    url = "https://openev.debatecoaches.org/files/cases.zip?download=1"
    a = adapter(tmp_path, url)
    a.fetcher = FakeFetcher({})
    assert a.discover() == [url]


def test_discover_refuses_unknown_direct_file_host(tmp_path):
    a = adapter(tmp_path, "https://example.net/cases.zip")
    a.fetcher = FakeFetcher({})

    with pytest.raises(AccessRefused):
        a.discover()


def test_binary_download_replaces_destination_atomically(tmp_path):
    class FixtureFetcher(Fetcher):
        def get(self, url):
            return FetchResult(url=url, status=200, content=b"new complete payload")

    destination = tmp_path / "release.docx"
    destination.write_bytes(b"old known-good payload")
    fetcher = FixtureFetcher(AccessPolicy(respect_robots=False))
    assert fetcher.download("https://openev.debatecoaches.org/release.docx", str(destination)) == str(destination)
    assert destination.read_bytes() == b"new complete payload"
    assert not list(tmp_path.glob(".card-scraper-download-*"))


def test_zip_extraction_skips_traversal_and_non_debate_files(tmp_path):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("cases/affirmative.htm", b"<html>aff</html>")
        zf.writestr("cases/negative.docx", b"docx bytes")
        zf.writestr("cases/topic.pdf", b"pdf bytes")
        zf.writestr("notes.txt", b"not a card")
        zf.writestr("../escaped.docx", b"must not be written")
        zf.writestr("/absolute.docx", b"must not be written")

    url = "https://openev.debatecoaches.org/releases/cases.zip"
    a = adapter(tmp_path)
    a.fetcher = FakeFetcher({url: archive.getvalue()})
    acquired = a._acquire_zip(url)

    assert [item.source.title for item in acquired] == ["affirmative", "negative", "topic"]
    assert all(item.source.origin == "http-index" for item in acquired)
    assert all(item.source.license == "public-test-fixture" for item in acquired)
    assert not (tmp_path / "escaped.docx").exists()
    assert not (tmp_path / "absolute.docx").exists()


def test_zip_limit_applies_to_members(tmp_path):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("one.docx", b"1")
        zf.writestr("two.docx", b"2")

    url = "https://openev.debatecoaches.org/releases/cases.zip"
    a = adapter(tmp_path)
    a.fetcher = FakeFetcher({url: archive.getvalue()})
    acquired = a._acquire_zip(url, limit=1)

    assert len(acquired) == 1
    assert acquired[0].source.title == "one"


def test_pdf_parser_is_page_level_and_transparent(tmp_path, monkeypatch):
    class FakePage:
        def extract_text(self):
            return (
                "Smith 26 (Jane Smith, Researcher)\n"
                "The policy changes incentives for hospitals and reduces costs "
                "for households across the country."
            )

    fake_pypdf = types.SimpleNamespace(PdfReader=lambda path: types.SimpleNamespace(
        pages=[FakePage()]))
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    from cardgraph.parse.pdf_card import parse_pdf

    path = tmp_path / "public.pdf"
    path.write_bytes(b"fixture")
    cards, root, report = parse_pdf(str(path), "source")

    assert len(cards) == 1
    assert cards[0].read_text == ""
    assert cards[0].disclosed_only
    assert cards[0].path == ["Page 1"]
    assert root.children[0].card_ids == [cards[0].card_id]
    assert "page level" in report.warnings[0]
