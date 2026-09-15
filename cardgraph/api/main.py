"""FastAPI service.

Local-first by default: binds to 127.0.0.1, no auth, no telemetry. If you ever
expose this beyond localhost, put auth in front of it and re-read the license
of every source in your index first -- "I built a search engine over camp files"
and "I republished camp files" are different acts and only one of them is
clearly fine.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ..graph.relate import build_all
from ..index.search import SearchEngine
from ..index.store import Store
from ..ingest.policy import source_catalog

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "web")
if not os.path.exists(os.path.join(WEB_DIR, "index.html")):
    WEB_DIR = os.path.join(os.sys.prefix, "share", "cardgraph")


def create_app(db_path: str = "data/cardgraph.db") -> FastAPI:
    store = Store(db_path)
    engine = SearchEngine(store)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        engine.build()
        try:
            yield
        finally:
            # SQLite keeps WAL/shm handles open on Windows. Closing the
            # app-owned store makes embedded TestClient use and clean temp
            # directories deterministic instead of leaking a connection until
            # process exit.
            store.close()

    app = FastAPI(title="Card Scraper", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
        allow_methods=["GET"], allow_headers=["*"],
    )

    @app.get("/api/stats")
    def stats() -> dict:
        return store.stats()

    @app.get("/api/catalog")
    def catalog() -> dict:
        """List reviewed source options and their access boundaries."""
        return {"sources": source_catalog()}

    @app.get("/api/sources")
    def sources(limit: int = Query(100, ge=1, le=500),
                origin: str | None = None) -> dict:
        """List searchable online/local evidence sources and their provenance."""
        where = []
        params: list[object] = []
        if origin:
            where.append("origin = ?")
            params.append(origin)
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        rows = store.conn.execute(
            "SELECT source_id, title, origin, license, fetched_at, url, card_count "
            f"FROM sources{clause} ORDER BY fetched_at DESC, title LIMIT ?",
            (*params, limit),
        ).fetchall()
        return {"sources": [dict(r) for r in rows]}

    @app.get("/api/search")
    def search(
        q: str = Query(..., min_length=1, max_length=500),
        k: int = Query(25, ge=1, le=100),
        side: str | None = Query(
            None, pattern="^(aff|neg|both|unknown)$",
        ),
        author: str | None = Query(None, max_length=200),
        year_min: int | None = Query(None, ge=1900, le=2200),
        year_max: int | None = Query(None, ge=1900, le=2200),
        block: str | None = Query(None, max_length=200),
        min_read_ratio: float | None = Query(None, ge=0.0, le=1.0),
        source: str | None = Query(None, max_length=300),
    ) -> dict:
        if year_min is not None and year_max is not None and year_min > year_max:
            raise HTTPException(422, "year_min must not exceed year_max")
        hits = engine.search(q, k=k, side=side, author=author, year_min=year_min,
                             year_max=year_max, block=block,
                             min_read_ratio=min_read_ratio, source=source)
        return {"query": q, "count": len(hits), "hits": [h.to_dict() for h in hits]}

    @app.get("/api/card/{card_id}")
    def card(card_id: str) -> dict:
        row = store.card(card_id)
        if not row:
            raise HTTPException(404, "no such card")
        source = store.conn.execute(
            "SELECT title AS source_title, origin AS source_origin, "
            "license AS source_license, url AS source_url "
            "FROM sources WHERE source_id=?", (row.get("source_id"),)
        ).fetchone()
        if source:
            row.update(dict(source))
        row["path"] = json.loads(row.pop("path_json") or "[]")
        row["warrant_flags"] = json.loads(row.get("warrant_flags") or "[]")
        row["similar"] = [h.to_dict() for h in engine.similar(card_id, k=6)]
        dupes = store.conn.execute(
            """SELECT dst_node_id AS other, confidence, evidence FROM edges
               WHERE kind='duplicates' AND src_node_id=?
               UNION
               SELECT src_node_id AS other, confidence, evidence FROM edges
               WHERE kind='duplicates' AND dst_node_id=?""",
            (card_id, card_id)).fetchall()
        row["duplicates"] = [dict(d) for d in dupes]
        return row

    @app.get("/api/tree")
    def tree(parent: str | None = Query(None, max_length=100),
             limit: int = Query(200, ge=1, le=500),
             offset: int = Query(0, ge=0),
             q: str | None = Query(None, max_length=200)) -> dict:
        return store.tree(parent, limit=limit, offset=offset, q=q)

    @app.get("/api/node/{node_id}")
    def node(node_id: str) -> dict:
        rows = store.cards_for_node(node_id)
        for r in rows:
            r["path"] = json.loads(r.pop("path_json") or "[]")
            r["warrant_flags"] = json.loads(r.get("warrant_flags") or "[]")
        meta = store.conn.execute("SELECT * FROM nodes WHERE node_id=?",
                                  (node_id,)).fetchone()
        if not meta:
            raise HTTPException(404, "no such node")
        m = dict(meta)
        m["path"] = json.loads(m.pop("path_json") or "[]")
        return {"node": m, "cards": rows, "edges": store.edges_for(node_id),
                "children": store.tree(node_id)["nodes"]}

    # ---- analysis --------------------------------------------------------
    # A model-backed run costs money and takes minutes, so the report is
    # persisted and served from disk. The UI never triggers one implicitly;
    # ?refresh=1 is the only way to spend anything, and it is a POST.
    analysis_path = os.path.join(os.path.dirname(db_path) or ".",
                                 "analysis.json")

    @app.get("/api/analysis")
    def get_analysis() -> dict:
        if not os.path.exists(analysis_path):
            return {"available": False,
                    "hint": "run `cardgraph analyze --json data/analysis.json`, "
                            "or POST /api/analysis/run"}
        with open(analysis_path) as fh:
            payload = json.load(fh)
        payload["available"] = True
        payload["generated_at"] = os.path.getmtime(analysis_path)
        return payload

    @app.post("/api/analysis/run")
    def run_analysis(top: int = Query(6, ge=1, le=50),
                     llm: bool = True,
                     blocks: bool = True) -> dict:
        from ..analysis import analyze
        report = analyze(store, max_positions=top, use_llm=llm,
                         generate_blocks=blocks, engine=engine)
        payload = report.to_dict()
        with open(analysis_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        payload["available"] = True
        return payload

    @app.get("/api/analysis/kinds")
    def analysis_kinds() -> dict:
        from ..analysis import KINDS
        return {"kinds": KINDS}

    @app.post("/api/graph/rebuild")
    def rebuild() -> dict:
        result = build_all(store)
        engine.build()
        return result

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(os.path.join(WEB_DIR, "index.html"))

    return app


app = create_app()
