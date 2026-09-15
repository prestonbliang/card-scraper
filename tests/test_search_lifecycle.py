"""Regression tests for search freshness and source-owned lifecycle behavior."""

from __future__ import annotations

from cardgraph.index.search import SearchEngine
from cardgraph.index.store import Store
from cardgraph.models import Card, Cite, OutlineNode, Source, NodeKind


def _card(source_id: str, ordinal: int, text: str, *, author: str = "Author",
          year: int = 2026, side: str = "aff", block: str = "Position") -> Card:
    card = Card(
        tag=f"{text} tag",
        cite=Cite(raw=f"{author} {str(year)[-2:]}", author=author, year=year),
        body=(text + " ") * 8,
        read_text=(text + " ") * 3,
        source_id=source_id,
        source_path=f"/files/{source_id}/case.docx",
        ordinal=ordinal,
        path=[block],
    )
    from cardgraph.models import Side
    card.side = Side(side)
    return card


def _store(tmp_path, cards: list[Card]) -> Store:
    store = Store(str(tmp_path / "cards.db"))
    source_ids = {card.source_id for card in cards}
    for source_id in source_ids:
        store.add_source(Source(
            source_id=source_id,
            path=f"/files/{source_id}/case.docx",
            title=f"{source_id} case",
            origin="local",
            card_count=sum(c.source_id == source_id for c in cards),
        ))
    store.add_cards(cards)
    return store


def test_live_engine_refreshes_after_cards_are_added(tmp_path):
    store = _store(tmp_path, [_card("mine", 0, "old claim")])
    engine = SearchEngine(store, cache_path=str(tmp_path / "index.pkl"))
    engine.build()
    assert not any(h.tag.startswith("new claim") for h in
                   engine.search("new claim", k=5))

    store.add_cards([_card("mine", 1, "new claim")])
    hits = engine.search("new claim", k=5)
    assert any(h.tag.startswith("new claim") for h in hits)


def test_metadata_filters_are_applied_before_ranking(tmp_path):
    cards = [_card("mine", i, f"shared debate vocabulary {i}", author="Other")
             for i in range(80)]
    cards.append(_card("mine", 80, "shared debate vocabulary target",
                       author="Target", year=2024, side="neg", block="AT: Target"))
    store = _store(tmp_path, cards)
    engine = SearchEngine(store, cache_path=str(tmp_path / "index.pkl"))
    engine.build()

    hits = engine.search(
        "shared debate vocabulary", k=1, author="Target", year_max=2024,
        side="neg", block="target",
    )
    assert len(hits) == 1
    assert hits[0].cite_author == "Target"
    assert hits[0].side == "neg"


def test_remove_source_removes_cards_fts_nodes_and_edges(tmp_path):
    store = _store(tmp_path, [_card("mine", 0, "removable claim")])
    store.add_source(Source(
        source_id="other", path="/files/other/case.docx", title="other",
        card_count=1,
    ))
    other = _card("other", 0, "other claim")
    store.add_cards([other])
    mine_node = OutlineNode(
        title="Mine", kind=NodeKind.BLOCK, path=["Mine"], source_id="mine",
    )
    store.add_outline(mine_node)
    store.add_edge(mine_node.node_id, other.card_id, "duplicates", 0.9)
    store.remove_source("mine")

    assert store.conn.execute(
        "SELECT 1 FROM cards WHERE source_id='mine'").fetchone() is None
    assert store.conn.execute(
        "SELECT 1 FROM fts_map WHERE card_id=?", (_card("mine", 0, "removable claim").card_id,)
    ).fetchone() is None
    assert store.conn.execute(
        "SELECT 1 FROM edges WHERE src_node_id=?", (mine_node.node_id,)
    ).fetchone() is None


def test_local_adapter_zero_limit_means_zero_files(tmp_path):
    from cardgraph.ingest.base import LocalDirAdapter

    (tmp_path / "one.docx").write_bytes(b"x")
    assert LocalDirAdapter(str(tmp_path)).acquire(limit=0) == []
