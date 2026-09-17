"""Regression checks for the single-file search interface contract."""

from pathlib import Path


WEB = Path(__file__).parents[1] / "web" / "index.html"


def test_search_state_is_shareable_and_restorable():
    html = WEB.read_text(encoding="utf-8")

    for marker in (
        'id="copyLink"',
        "function syncUrl",
        "function restoreUrlState",
        'window.addEventListener("popstate"',
        'params.set("q", q)',
        'params.set("mode", $("#searchMode").value)',
        'params.set("side", $("#side").value)',
        'params.set("source", $("#source").value.trim())',
        'params.set("block", $("#block").value.trim())',
    ):
        assert marker in html


def test_search_url_does_not_reintroduce_manual_year_controls():
    html = WEB.read_text(encoding="utf-8")
    assert 'id="yearMin"' not in html
    assert 'id="yearMax"' not in html
