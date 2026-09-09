"""SQLite storage + full-text index.

One file, no server. A season of camp files is a few hundred thousand cards,
which SQLite handles without complaint, and a single-file database is something
you can hand to a teammate.

The FTS index is built over **read_text**, not body. See models.Card for why:
the body is context the debater never says, and indexing it means a search for
"ratepayer transfer" happily returns a card whose read portion is about
something else entirely. Body is stored and searchable via a separate column so
you can still find the context when you want it, but it does not drive ranking.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager

from ..models import Card, Cite, OutlineNode, Side, Source

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS sources (
    source_id   TEXT PRIMARY KEY,
    path        TEXT,
    title       TEXT,
    origin      TEXT,
    license     TEXT,
    fetched_at  TEXT,
    url         TEXT,
    card_count  INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS cards (
    card_id        TEXT PRIMARY KEY,
    tag            TEXT NOT NULL,
    cite_raw       TEXT,
    cite_author    TEXT,
    cite_year      INTEGER,
    cite_pub       TEXT,
    cite_url       TEXT,
    body           TEXT,
    read_text      TEXT,
    emphasis_text  TEXT,
    read_ratio     REAL,
    side           TEXT,
    path_json      TEXT,
    block          TEXT,
    pocket         TEXT,
    source_id      TEXT,
    source_path    TEXT,
    ordinal        INTEGER,
    warrant_flags  TEXT,
    disclosed_only INTEGER DEFAULT 0,
    round_context  TEXT
);

CREATE INDEX IF NOT EXISTS idx_cards_source ON cards(source_id);
CREATE INDEX IF NOT EXISTS idx_cards_block  ON cards(block);
CREATE INDEX IF NOT EXISTS idx_cards_author ON cards(cite_author);
CREATE INDEX IF NOT EXISTS idx_cards_year   ON cards(cite_year);
CREATE INDEX IF NOT EXISTS idx_cards_side   ON cards(side);

CREATE TABLE IF NOT EXISTS nodes (
    node_id    TEXT PRIMARY KEY,
    title      TEXT,
    kind       TEXT,
    path_json  TEXT,
    parent_id  TEXT,
    source_id  TEXT,
    card_count INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_nodes_title  ON nodes(title);

-- Argument graph: "AT: Ratepayer Harm" --answers--> "Ratepayer Harm"
CREATE TABLE IF NOT EXISTS edges (
    src_node_id TEXT,
    dst_node_id TEXT,
    kind        TEXT,          -- answers | supports | duplicates
    confidence  REAL,
    evidence    TEXT,
    PRIMARY KEY (src_node_id, dst_node_id, kind)
);

CREATE VIRTUAL TABLE IF NOT EXISTS cards_fts USING fts5(
    tag,
    read_text,
    cite_raw,
    body,
    content=''
);

-- rowid <-> card_id mapping for the contentless FTS table
CREATE TABLE IF NOT EXISTS fts_map (
    rowid   INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id TEXT UNIQUE
);
"""


class Store:
    def __init__(self, path: str = "data/cardgraph.db"):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    @contextmanager
    def tx(self):
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # -- writes ------------------------------------------------------------

    def add_source(self, src: Source) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT OR REPLACE INTO sources
                   (source_id, path, title, origin, license, fetched_at, url, card_count)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (src.source_id, src.path, src.title, src.origin, src.license,
                 src.fetched_at, src.url, src.card_count),
            )

    def add_cards(self, cards: list[Card]) -> int:
        added = 0
        with self.tx() as c:
            for card in cards:
                cur = c.execute(
                    """INSERT OR IGNORE INTO cards
                       (card_id, tag, cite_raw, cite_author, cite_year, cite_pub,
                        cite_url, body, read_text, emphasis_text, read_ratio, side,
                        path_json, block, pocket, source_id, source_path, ordinal,
                        warrant_flags, disclosed_only, round_context)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (card.card_id, card.tag, card.cite.raw, card.cite.author,
                     card.cite.year, card.cite.publication, card.cite.url,
                     card.body, card.read_text, card.emphasis_text,
                     card.read_ratio, card.side.value, json.dumps(card.path),
                     card.block, card.pocket, card.source_id, card.source_path,
                     card.ordinal, json.dumps(card.warrant_flags),
                     1 if card.disclosed_only else 0, card.round_context),
                )
                if cur.rowcount:
                    added += 1
                    m = c.execute(
                        "INSERT OR IGNORE INTO fts_map (card_id) VALUES (?)",
                        (card.card_id,),
                    )
                    rowid = m.lastrowid
                    if rowid:
                        c.execute(
                            "INSERT INTO cards_fts (rowid, tag, read_text, cite_raw, body)"
                            " VALUES (?,?,?,?,?)",
                            (rowid, card.tag, card.read_text, card.cite.raw, card.body),
                        )
        return added

    def add_outline(self, root: OutlineNode, parent_id: str | None = None) -> None:
        with self.tx() as c:
            self._add_outline(c, root, parent_id)

    def _add_outline(self, c, node: OutlineNode, parent_id: str | None) -> None:
        c.execute(
            """INSERT OR REPLACE INTO nodes
               (node_id, title, kind, path_json, parent_id, source_id, card_count)
               VALUES (?,?,?,?,?,?,?)""",
            (node.node_id, node.title, node.kind.value, json.dumps(node.path),
             parent_id, node.source_id, len(node.card_ids)),
        )
        for child in node.children:
            self._add_outline(c, child, node.node_id)

    def add_edge(self, src: str, dst: str, kind: str, confidence: float,
                 evidence: str = "") -> None:
        with self.tx() as c:
            c.execute(
                """INSERT OR REPLACE INTO edges
                   (src_node_id, dst_node_id, kind, confidence, evidence)
                   VALUES (?,?,?,?,?)""",
                (src, dst, kind, confidence, evidence),
            )

    # -- reads -------------------------------------------------------------

    def stats(self) -> dict:
        c = self.conn
        q = lambda s: c.execute(s).fetchone()[0]  # noqa: E731
        return {
            "sources": q("SELECT COUNT(*) FROM sources"),
            "cards": q("SELECT COUNT(*) FROM cards"),
            "nodes": q("SELECT COUNT(*) FROM nodes"),
            "edges": q("SELECT COUNT(*) FROM edges"),
            "cards_with_read_text": q(
                "SELECT COUNT(*) FROM cards WHERE read_text != ''"),
            "authors": q("SELECT COUNT(DISTINCT cite_author) FROM cards "
                         "WHERE cite_author IS NOT NULL"),
        }

    def card(self, card_id: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM cards WHERE card_id=?",
                              (card_id,)).fetchone()
        return dict(r) if r else None

    def cards_for_node(self, node_id: str) -> list[dict]:
        n = self.conn.execute("SELECT * FROM nodes WHERE node_id=?",
                              (node_id,)).fetchone()
        if not n:
            return []
        path = json.loads(n["path_json"])
        rows = self.conn.execute(
            "SELECT * FROM cards WHERE source_id=? AND path_json=? ORDER BY ordinal",
            (n["source_id"], json.dumps(path)),
        ).fetchall()
        return [dict(r) for r in rows]

    def tree(self, parent_id: str | None = None) -> list[dict]:
        if parent_id is None:
            rows = self.conn.execute(
                "SELECT * FROM nodes WHERE parent_id IS NULL ORDER BY title"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM nodes WHERE parent_id=? ORDER BY title",
                (parent_id,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["path"] = json.loads(d.pop("path_json") or "[]")
            d["has_children"] = bool(self.conn.execute(
                "SELECT 1 FROM nodes WHERE parent_id=? LIMIT 1", (d["node_id"],)
            ).fetchone())
            out.append(d)
        return out

    def edges_for(self, node_id: str) -> dict[str, list[dict]]:
        out_ = self.conn.execute(
            """SELECT e.*, n.title AS dst_title FROM edges e
               LEFT JOIN nodes n ON n.node_id = e.dst_node_id
               WHERE e.src_node_id=?""", (node_id,)).fetchall()
        in_ = self.conn.execute(
            """SELECT e.*, n.title AS src_title FROM edges e
               LEFT JOIN nodes n ON n.node_id = e.src_node_id
               WHERE e.dst_node_id=?""", (node_id,)).fetchall()
        return {"outgoing": [dict(r) for r in out_],
                "incoming": [dict(r) for r in in_]}

    def all_cards(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM cards")]

    def close(self) -> None:
        self.conn.close()


def row_to_card(r: dict) -> Card:
    return Card(
        tag=r["tag"],
        cite=Cite(raw=r["cite_raw"] or "", author=r["cite_author"],
                  year=r["cite_year"], publication=r["cite_pub"], url=r["cite_url"]),
        body=r["body"] or "",
        read_text=r["read_text"] or "",
        emphasis_text=r["emphasis_text"] or "",
        source_id=r["source_id"] or "",
        source_path=r["source_path"] or "",
        ordinal=r["ordinal"] or 0,
        path=json.loads(r["path_json"] or "[]"),
        side=Side(r["side"] or "unknown"),
        card_id=r["card_id"],
        disclosed_only=bool(r["disclosed_only"] if "disclosed_only" in r.keys() else 0),
        round_context=(r["round_context"] if "round_context" in r.keys() else "") or "",
    )
