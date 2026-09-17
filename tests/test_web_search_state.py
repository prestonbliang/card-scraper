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
        'params.set("card", card)',
        'params.set("node", node)',
        'function detailShare(kind, id)',
        'showCard(params.get("card"), {updateUrl: false})',
        'showNode(params.get("node"), {updateUrl: false})',
        'const RECENT_SEARCHES_KEY',
        'function rememberSearch(query)',
        'function renderRecentSearches(target)',
        'Press / to focus · Esc to clear',
        'const PINNED_KEY',
        'function pinButton(c)',
        'function showBoard()',
        'function exportBoard()',
        'id="mBoard"',
        'const PINNED_ORDER_KEY',
        'function orderedPinned()',
        'function boardCardEl(card)',
        'function reorderPinned(draggedId, targetId)',
        'Private note for this card',
        'function highlightedRead(text, reasons)',
        'el("mark", "search-hit", part)',
        'evidence-badge',
        'Traceable evidence',
    ):
        assert marker in html


def test_recent_searches_and_keyboard_shortcuts_are_local_only():
    html = WEB.read_text(encoding="utf-8")
    assert "localStorage.getItem(RECENT_SEARCHES_KEY)" in html
    assert "localStorage.setItem(RECENT_SEARCHES_KEY" in html
    assert "document.addEventListener(\"keydown\"" in html
    assert "e.key === \"/\"" in html
    assert "e.key === \"Escape\"" in html


def test_research_board_is_local_and_exportable():
    html = WEB.read_text(encoding="utf-8")
    assert "localStorage.setItem(PINNED_KEY" in html
    assert "card-scraper-research-board.md" in html
    assert "Export cited brief" in html
    assert "Affirmative" in html and "Negative" in html


def test_detail_urls_have_copy_link_controls():
    html = WEB.read_text(encoding="utf-8")
    assert 'detailShare("card", id)' in html
    assert 'detailShare("node", id)' in html
    assert 'title = `Copy a link to this ${kind}`' in html


def test_search_url_does_not_reintroduce_manual_year_controls():
    html = WEB.read_text(encoding="utf-8")
    assert 'id="yearMin"' not in html
    assert 'id="yearMax"' not in html
