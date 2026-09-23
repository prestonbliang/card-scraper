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
import tempfile
import threading
from datetime import datetime, timezone
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.background import BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..graph.relate import build_all
from ..index.search import SearchEngine, parse_smart_request
from ..index.store import Store
from ..ingest.base import OnlineEvidenceAdapter
from ..ingest.policy import AccessRefused, source_catalog
from ..parse.docx_card import UnsupportedFormat, parse_any
from ..packet import PacketError, build_packet
from ..workspace import (MAX_TOTAL_BYTES, WorkspaceBundleError, bundle_summary,
                         export_bundle, inspect_bundle, restore_bundle)

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "web")
if not os.path.exists(os.path.join(WEB_DIR, "index.html")):
    WEB_DIR = os.path.join(os.sys.prefix, "share", "cardgraph")


class IngestRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=25, ge=1, le=100)
    license: str = Field(
        default="verify source terms before use", max_length=300,
    )
    replace_source_id: str | None = Field(default=None, max_length=100)


class WorkspaceExportRequest(BaseModel):
    browser_state: dict | None = None
    include_corpus: bool = True


class PacketRequest(BaseModel):
    card_ids: list[str] = Field(default_factory=list, max_length=500)
    query: str | None = Field(default=None, max_length=500)
    k: int = Field(default=10, ge=1, le=500)
    side: str | None = Field(default=None, pattern="^(aff|neg|both|unknown)$")
    author: str | None = Field(default=None, max_length=200)
    year_min: int | None = Field(default=None, ge=1900, le=2200)
    year_max: int | None = Field(default=None, ge=1900, le=2200)
    block: str | None = Field(default=None, max_length=200)
    min_read_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str | None = Field(default=None, max_length=300)
    mode: str = Field(default="balanced", pattern="^(strict|balanced|explore)$")


def create_app(db_path: str = "data/cardgraph.db") -> FastAPI:
    store = Store(db_path)
    store.recover_interrupted_jobs()
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

    app = FastAPI(title="Card Scraper", version="0.2.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
        allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict:
        """Return a lightweight readiness signal for local launch scripts."""
        try:
            stats = store.stats()
            database = "ok"
        except Exception:  # noqa: BLE001
            stats = None
            database = "error"
        return {
            "status": "ok" if database == "ok" else "error",
            "database": database,
            "search_index": "ready" if engine._built else "starting",
            "stats": stats,
        }

    @app.get("/api/stats")
    def stats() -> dict:
        return store.stats()

    @app.post("/api/workspace/export")
    def workspace_export(request: WorkspaceExportRequest,
                         background_tasks: BackgroundTasks) -> FileResponse:
        """Download a verified ZIP snapshot without exposing its temp path."""
        try:
            fd, path = tempfile.mkstemp(prefix="card-scraper-", suffix=".zip")
            os.close(fd)
            export_bundle(db_path, path, browser_state=request.browser_state,
                          include_corpus=request.include_corpus)
        except WorkspaceBundleError as exc:
            if "path" in locals() and os.path.exists(path):
                os.remove(path)
            raise HTTPException(400, str(exc)) from exc
        background_tasks.add_task(os.remove, path)
        return FileResponse(path, media_type="application/zip",
                            filename="card-scraper-workspace.zip")

    @app.post("/api/packet")
    def packet(request: PacketRequest, background_tasks: BackgroundTasks) -> FileResponse:
        """Download a research packet ZIP for pinned cards or a search query."""
        if request.year_min is not None and request.year_max is not None \
                and request.year_min > request.year_max:
            raise HTTPException(422, "year_min must not exceed year_max")
        if request.card_ids and request.query:
            raise HTTPException(422, "pass card_ids or a query, not both")
        kwargs = {"query": request.query} if request.query else \
            {"card_ids": request.card_ids}
        for option in ("side", "author", "block", "source", "mode"):
            value = getattr(request, option)
            if value:
                kwargs[option] = value
        if request.year_min is not None:
            kwargs["year_min"] = request.year_min
        if request.year_max is not None:
            kwargs["year_max"] = request.year_max
        if request.min_read_ratio is not None:
            kwargs["min_read_ratio"] = request.min_read_ratio
        if request.query:
            kwargs["k"] = request.k
        try:
            filename, payload, _manifest = build_packet(store, **kwargs)
        except PacketError as exc:
            raise HTTPException(400, str(exc)) from exc
        fd, path = tempfile.mkstemp(prefix="card-scraper-packet-", suffix=".zip")
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        background_tasks.add_task(os.remove, path)
        return FileResponse(path, media_type="application/zip", filename=filename)

    @app.post("/api/workspace/inspect")
    async def workspace_inspect(request: Request) -> dict:
        """Preflight a bundle without touching the current workspace."""
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_TOTAL_BYTES:
                    raise HTTPException(413, "workspace bundle is too large")
            except ValueError as exc:
                raise HTTPException(400, "invalid content length") from exc
        fd, path = tempfile.mkstemp(prefix="card-scraper-inspect-", suffix=".zip")
        os.close(fd)
        size = 0
        try:
            with open(path, "wb") as target:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_TOTAL_BYTES:
                        raise HTTPException(413, "workspace bundle is too large")
                    target.write(chunk)
            result = inspect_bundle(path)
            return {"bundle": bundle_summary(result.manifest),
                    "browser_state": result.browser_state, "stats": result.stats,
                    "dropped_pins": result.dropped_pins}
        except WorkspaceBundleError as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    @app.post("/api/workspace/restore")
    async def workspace_restore(request: Request) -> dict:
        """Validate a ZIP upload completely before replacing this local index."""
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                declared_size = int(content_length)
            except ValueError as exc:
                raise HTTPException(400, "invalid content length") from exc
            if declared_size < 0 or declared_size > MAX_TOTAL_BYTES:
                raise HTTPException(413, "workspace bundle is too large")
        fd, path = tempfile.mkstemp(prefix="card-scraper-upload-", suffix=".zip")
        os.close(fd)
        size = 0
        try:
            with open(path, "wb") as target:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_TOTAL_BYTES:
                        raise HTTPException(413, "workspace bundle is too large")
                    target.write(chunk)
            result = restore_bundle(db_path, path, connection=store.conn)
            engine.build(use_cache=False)
            return {
                "restored": bundle_summary(result.manifest),
                "browser_state": result.browser_state,
                "stats": result.stats,
                "dropped_pins": result.dropped_pins,
            }
        except WorkspaceBundleError as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    @app.get("/api/catalog")
    def catalog() -> dict:
        """List reviewed source options and their access boundaries."""
        return {"sources": source_catalog()}

    # Browser imports run outside the request thread so a slow public index
    # cannot freeze the local application. Each job owns a separate SQLite
    # connection; the search engine sees the new cards through card_revision().
    jobs: dict[str, dict] = {}
    active_refreshes: dict[str, str] = {}
    jobs_lock = threading.Lock()

    def _public_job(job_id: str, request: IngestRequest) -> None:
        with jobs_lock:
            job = jobs[job_id]
            job["state"] = "discovering"
        worker_store = Store(db_path)
        worker_store.save_ingest_job(job)
        try:
            workdir = os.path.join(os.path.dirname(db_path) or ".", "corpus")
            adapter = OnlineEvidenceAdapter(
                index_url=request.url, license=request.license, workdir=workdir,
            )
            try:
                acquired = adapter.acquire(limit=request.limit)
            except AccessRefused as exc:
                error = str(exc)
                if job.get("replace_source_id"):
                    worker_store.mark_source_refresh_failed(job["replace_source_id"], error)
                with jobs_lock:
                    job.update(state="failed", error=error)
                worker_store.save_ingest_job(job)
                return
            except Exception as exc:  # noqa: BLE001
                error = f"source discovery failed: {exc}"
                if job.get("replace_source_id"):
                    worker_store.mark_source_refresh_failed(job["replace_source_id"], error)
                with jobs_lock:
                    job.update(state="failed", error=error)
                worker_store.save_ingest_job(job)
                return

            with jobs_lock:
                job.update(state="importing", total=len(acquired))
            worker_store.save_ingest_job(job)

            if job.get("replace_source_id") and not acquired:
                error = "refresh discovered no supported files"
                worker_store.mark_source_refresh_failed(
                    job["replace_source_id"], error)
                with jobs_lock:
                    job["failed"] += 1
                    job["details"].append(error)
                    job["state"] = "failed"
                    job["error"] = "refresh failed; last-known-good evidence was retained"
                worker_store.save_ingest_job(job)
                return

            if job.get("replace_source_id"):
                # Stage every replacement before touching the live index. A
                # source can expand to several files; parsing one successfully
                # and failing on the next must not create a mixed release.
                staged = []
                for index, item in enumerate(acquired, 1):
                    with jobs_lock:
                        if job["cancel"].is_set():
                            job["state"] = "cancelled"
                            worker_store.save_ingest_job(job)
                            return
                    try:
                        cards, root, _report = parse_any(item.path, item.source.source_id)
                    except Exception as exc:  # noqa: BLE001
                        error = f"{os.path.basename(item.path)}: {exc}"
                        with jobs_lock:
                            job["failed"] += 1
                            job["details"].append(error)
                            job["state"] = "failed"
                            job["error"] = "refresh failed; last-known-good evidence was retained"
                        worker_store.mark_source_refresh_failed(
                            job["replace_source_id"], error)
                        worker_store.save_ingest_job(job)
                        return
                    item.source.card_count = len(cards)
                    staged.append((item.source, cards, root))
                    with jobs_lock:
                        job["processed"] = index
                    worker_store.save_ingest_job(job)

                added = worker_store.replace_source(job["replace_source_id"], staged)
                with jobs_lock:
                    job["imported"] = len(staged)
                    job["cards_added"] = added
                    job["state"] = "completed"
                    job["stats"] = worker_store.stats()
                worker_store.clear_source_refresh_failure(job["replace_source_id"])
                worker_store.save_ingest_job(job)
                return

            replaced = False
            for index, item in enumerate(acquired, 1):
                with jobs_lock:
                    if job["cancel"].is_set():
                        job["state"] = "cancelled"
                        worker_store.save_ingest_job(job)
                        return
                try:
                    cards, root, _report = parse_any(item.path, item.source.source_id)
                except UnsupportedFormat:
                    with jobs_lock:
                        job["skipped"] += 1
                except Exception as exc:  # noqa: BLE001
                    with jobs_lock:
                        job["failed"] += 1
                        job["details"].append(f"{os.path.basename(item.path)}: {exc}")
                    if job.get("replace_source_id"):
                        worker_store.mark_source_refresh_failed(
                            job["replace_source_id"], str(exc))
                else:
                    item.source.card_count = len(cards)
                    if job.get("replace_source_id") and not replaced:
                        # The new file is fully parsed before this point. Only
                        # now remove the old evidence, so a failed download or
                        # parse never destroys the last known-good source.
                        worker_store.remove_source(job["replace_source_id"])
                        replaced = True
                    existing = worker_store.conn.execute(
                        "SELECT 1 FROM sources WHERE source_id=?",
                        (item.source.source_id,),
                    ).fetchone()
                    if not existing:
                        worker_store.add_source(item.source)
                        added = worker_store.add_cards(cards)
                        worker_store.add_outline(root)
                        with jobs_lock:
                            job["imported"] += 1
                            job["cards_added"] += added
                with jobs_lock:
                    job["processed"] = index
                worker_store.save_ingest_job(job)
            with jobs_lock:
                if job["state"] != "cancelled":
                    if job.get("replace_source_id") and job["failed"]:
                        job["state"] = "failed"
                        job["error"] = "refresh failed; last-known-good evidence was retained"
                    else:
                        job["state"] = "completed"
                        if job.get("replace_source_id"):
                            worker_store.clear_source_refresh_failure(job["replace_source_id"])
                    job["stats"] = worker_store.stats()
            worker_store.save_ingest_job(job)
        finally:
            worker_store.close()

    def public_job(job_id: str, request: IngestRequest) -> None:
        """Convert unexpected worker failures into a durable failed job.

        The normal discovery/parser paths report their own errors, but storage,
        transaction, or third-party failures can still escape those branches.
        A daemon thread has no caller to receive such an exception; without this
        guard the UI would show an import stuck in ``importing`` until restart.
        """
        try:
            _public_job(job_id, request)
        except Exception as exc:  # noqa: BLE001
            error = f"import worker failed: {exc}"
            with jobs_lock:
                job = jobs.get(job_id)
                if job:
                    job.update(state="failed", error=error)
            try:
                recovery_store = Store(db_path)
                try:
                    if job:
                        recovery_store.save_ingest_job(job)
                finally:
                    recovery_store.close()
            except Exception:
                # The original failure is already reflected in memory; avoid
                # hiding it behind a second storage exception during recovery.
                pass
        finally:
            with jobs_lock:
                if jobs.get(job_id, {}).get("replace_source_id"):
                    source_id = jobs[job_id]["replace_source_id"]
                    if active_refreshes.get(source_id) == job_id:
                        del active_refreshes[source_id]

    def public_job_view(job: dict) -> dict:
        return {key: value for key, value in job.items() if key != "cancel"}

    def queue_ingest(request: IngestRequest) -> dict:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "job_id": job_id, "url": request.url, "license": request.license,
            "replace_source_id": request.replace_source_id,
            "state": "queued", "processed": 0, "total": None,
            "imported": 0, "cards_added": 0, "skipped": 0, "failed": 0,
            "details": [], "error": None, "stats": None,
            "created_at": time.time(),
            "cancel": threading.Event(),
        }
        with jobs_lock:
            if request.replace_source_id:
                existing_id = active_refreshes.get(request.replace_source_id)
                existing = jobs.get(existing_id) if existing_id else None
                # Coalesce only onto a job that is still running. A failed or
                # cancelled refresh lingers in this map until its worker thread
                # finishes durable bookkeeping; queuing behind that corpse would
                # make the new request report the dead job's terminal state.
                if existing and existing.get("state") in {
                        "queued", "discovering", "importing", "cancelling"}:
                    return public_job_view(existing)
                active_refreshes[request.replace_source_id] = job_id
            jobs[job_id] = job
        store.save_ingest_job(job)
        threading.Thread(
            target=public_job, args=(job_id, request), daemon=True,
        ).start()
        return public_job_view(job)

    @app.post("/api/ingest", status_code=202)
    def ingest_source(request: IngestRequest) -> dict:
        """Queue a capped public import and return a pollable job."""
        return queue_ingest(request)

    @app.get("/api/ingest")
    def ingest_history(limit: int = Query(50, ge=1, le=200)) -> dict:
        return {"jobs": store.ingest_jobs(limit)}

    @app.get("/api/ingest/{job_id}")
    def ingest_status(job_id: str) -> dict:
        with jobs_lock:
            job = jobs.get(job_id)
            if job:
                return public_job_view(job)
        job = store.ingest_job(job_id)
        if not job:
            raise HTTPException(404, "no such ingest job")
        return job

    @app.delete("/api/ingest/{job_id}")
    def cancel_ingest(job_id: str) -> dict:
        with jobs_lock:
            job = jobs.get(job_id)
            if not job:
                raise HTTPException(404, "no such ingest job")
            if job["state"] in {"queued", "discovering", "importing"}:
                job["cancel"].set()
                job["state"] = "cancelling"
                store.save_ingest_job(job)
            return public_job_view(job)

    @app.post("/api/sources/{source_id}/refresh", status_code=202)
    def refresh_source(source_id: str) -> dict:
        row = store.conn.execute(
            "SELECT url, license FROM sources WHERE source_id=?", (source_id,)
        ).fetchone()
        if not row or not row["url"]:
            raise HTTPException(404, "source has no refreshable public URL")
        return queue_ingest(IngestRequest(
            url=row["url"], license=row["license"] or "verify source terms before use",
            replace_source_id=source_id,
        ))

    @app.delete("/api/sources/{source_id}")
    def remove_source(source_id: str) -> dict:
        if not store.conn.execute(
                "SELECT 1 FROM sources WHERE source_id=?", (source_id,)
        ).fetchone():
            raise HTTPException(404, "no such source")
        store.remove_source(source_id)
        return {"removed": source_id, "stats": store.stats()}

    def source_health(row: dict) -> dict:
        """Return an explainable freshness state for a source record."""
        if row.get("last_refresh_failed_at"):
            return {"state": "refresh-failed", "age_days": None,
                    "error": row.get("last_refresh_error")}
        fetched = row.get("fetched_at")
        if not fetched:
            return {"state": "unknown", "age_days": None, "error": None}
        try:
            stamp = datetime.fromisoformat(fetched.replace("Z", "+00:00"))
            age = max(0, (datetime.now(timezone.utc) - stamp).days)
        except (TypeError, ValueError):
            return {"state": "unknown", "age_days": None, "error": None}
        threshold = 90 if row.get("origin") in {"opendebateevidence", "dataset"} else 30
        return {"state": "stale" if age >= threshold else "fresh",
                "age_days": age, "error": None, "threshold_days": threshold}

    def with_source_health(row: dict) -> dict:
        item = dict(row)
        item["health"] = source_health(item)
        return item

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
            "SELECT source_id, title, origin, license, fetched_at, url, card_count, "
            "last_refresh_failed_at, last_refresh_error "
            f"FROM sources{clause} ORDER BY fetched_at DESC, title LIMIT ?",
            (*params, limit),
        ).fetchall()
        return {"sources": [with_source_health(r) for r in rows]}

    @app.post("/api/sources/refresh-stale", status_code=202)
    def refresh_stale_sources() -> dict:
        rows = store.conn.execute(
            "SELECT source_id, url, license, origin, fetched_at, "
            "last_refresh_failed_at, last_refresh_error FROM sources "
            "WHERE url IS NOT NULL AND url != ''"
        ).fetchall()
        queued = []
        for row in rows:
            if source_health(dict(row))["state"] != "stale":
                continue
            queued.append(queue_ingest(IngestRequest(
                url=row["url"], license=row["license"] or "verify source terms before use",
                replace_source_id=row["source_id"],
            )))
        return {"queued": queued, "count": len(queued)}

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
        mode: str = Query("balanced", pattern="^(strict|balanced|explore)$"),
    ) -> dict:
        if year_min is not None and year_max is not None and year_min > year_max:
            raise HTTPException(422, "year_min must not exceed year_max")
        hits = engine.search(q, k=k, side=side, author=author, year_min=year_min,
                             year_max=year_max, block=block,
                             min_read_ratio=min_read_ratio, source=source,
                             mode=mode)
        return {"query": q, "count": len(hits), "hits": [h.to_dict() for h in hits]}

    @app.get("/api/smart-search")
    def smart_search(
        q: str = Query(..., min_length=1, max_length=500),
        k: int = Query(25, ge=1, le=100),
        side: str | None = Query(None, pattern="^(aff|neg|both|unknown)$"),
        author: str | None = Query(None, max_length=200),
        year_min: int | None = Query(None, ge=1900, le=2200),
        year_max: int | None = Query(None, ge=1900, le=2200),
        block: str | None = Query(None, max_length=200),
        min_read_ratio: float | None = Query(None, ge=0.0, le=1.0),
        source: str | None = Query(None, max_length=300),
        mode: str = Query("balanced", pattern="^(strict|balanced|explore)$"),
    ) -> dict:
        """Expand conversational debate requests without generating evidence.

        Every returned hit is still retrieved from the local index; ``variants``
        makes the transparent query expansion visible to the caller.
        """
        if year_min is not None and year_max is not None and year_min > year_max:
            raise HTTPException(422, "year_min must not exceed year_max")
        interpreted_query, inferred = parse_smart_request(q)
        explicit = {
            key: value for key, value in {
                "side": side, "author": author, "year_min": year_min,
                "year_max": year_max, "block": block,
                "min_read_ratio": min_read_ratio, "source": source,
            }.items() if value is not None
        }
        interpreted = {**inferred, **explicit}
        hits, variants = engine.smart_search(
            q, k=k, side=side, author=author, year_min=year_min,
            year_max=year_max, block=block, min_read_ratio=min_read_ratio,
            source=source, mode=mode,
        )
        return {
            "query": q, "interpreted_query": interpreted_query,
            "interpreted_filters": interpreted, "variants": variants,
            "count": len(hits), "hits": [h.to_dict() for h in hits],
        }

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
        try:
            with open(analysis_path, encoding="utf-8") as fh:
                payload = json.load(fh)
            if not isinstance(payload, dict):
                raise ValueError("analysis report must be a JSON object")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return {"available": False,
                    "error": f"saved analysis report is unreadable: {exc}",
                    "hint": "run POST /api/analysis/run to replace it"}
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
        temporary = None
        try:
            fd, temporary = tempfile.mkstemp(
                prefix=".card-scraper-analysis-", suffix=".json",
                dir=os.path.dirname(analysis_path) or ".",
            )
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(temporary, analysis_path)
            temporary = None
        finally:
            if temporary and os.path.exists(temporary):
                os.remove(temporary)
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
