"""Portable Card Scraper workspace bundles.

A bundle is a ZIP with a SQLite snapshot, the source files that can be found,
and a small browser-state export.  The manifest and SHA-256 hashes make a
restore verifiable before it touches the live index.  Restore never deletes an
existing corpus directory: imported files live under a new ``corpus/imported``
folder and the database is swapped only after every member has been validated.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import shutil
import sqlite3
import stat
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

BUNDLE_FORMAT = "card-scraper-workspace"
BUNDLE_VERSION = 1
DATABASE_MEMBER = "cardgraph.db"
STATE_MEMBER = "browser-state.json"
MAX_MEMBERS = 200_000
MAX_MEMBER_BYTES = 8 * 1024 * 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_STATE_BYTES = 32 * 1024 * 1024


class WorkspaceBundleError(ValueError):
    """Raised when a workspace bundle is invalid or cannot be restored safely."""


@dataclass
class WorkspaceRestore:
    manifest: dict[str, Any]
    browser_state: dict[str, Any] | None
    imported_files: int
    stats: dict[str, int]
    dropped_pins: int = 0


@dataclass
class WorkspaceInspection:
    manifest: dict[str, Any]
    browser_state: dict[str, Any] | None
    stats: dict[str, int]
    dropped_pins: int = 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    try:
        data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise WorkspaceBundleError(f"browser state is not JSON-safe: {exc}") from exc
    if len(data) > MAX_STATE_BYTES:
        raise WorkspaceBundleError("browser state is too large to include in a bundle")
    return data


def _normalize_browser_state(state: dict[str, Any] | None) -> dict[str, Any] | None:
    if state is None:
        return None
    if not isinstance(state, dict):
        raise WorkspaceBundleError("browser state must be an object")
    pinned = state.get("pinned", {})
    order = state.get("pinned_order", [])
    recent = state.get("recent_searches", [])
    if not isinstance(pinned, dict) or not isinstance(order, list) \
            or not isinstance(recent, list):
        raise WorkspaceBundleError("browser state has an invalid board or search history")
    if len(pinned) > 10_000 or len(order) > 10_000 or len(recent) > 100:
        raise WorkspaceBundleError("browser state contains too many items")
    clean_order = [item for item in order if isinstance(item, str) and item in pinned]
    clean_recent = [item.strip() for item in recent
                    if isinstance(item, str) and item.strip()][:100]
    normalized = {
        "pinned": pinned,
        "pinned_order": clean_order,
        "recent_searches": clean_recent,
    }
    _json_bytes(normalized)
    return normalized


def _safe_member(name: str) -> bool:
    """Return whether a ZIP member can be addressed below its extraction root.

    ZIP names are POSIX-looking even on Windows. Rejecting anything that needs
    normalization is intentional: ``corpus/../escape`` must not become a valid
    member merely because ``normpath`` can simplify it.
    """
    if not name or "\x00" in name or "\\" in name or ":" in name:
        return False
    normalized = posixpath.normpath(name)
    if normalized != name or normalized in {"", ".", ".."}:
        return False
    return not name.startswith("/") and not name.startswith("../") \
        and not PurePosixPath(name).drive \
        and all(part not in {"", ".", ".."} for part in name.split("/"))


def _safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return value[:120] or "source"


def _resolve_source_path(raw: str, db_path: str) -> str | None:
    if not raw:
        return None
    # Relative source paths are conventionally relative to the database, not
    # to whichever directory happens to launch the CLI. Check that location
    # first, then retain the historical cwd fallback for older databases.
    candidates = [raw] if os.path.isabs(raw) else [
        os.path.join(os.path.dirname(db_path), raw),
        os.path.abspath(raw),
    ]
    seen: set[str] = set()
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        # Check the link before realpath: resolving first would make
        # os.path.islink() false and silently include a file outside the
        # workspace through a symlink.
        if os.path.islink(candidate):
            continue
        resolved = os.path.realpath(candidate)
        if os.path.isfile(resolved) and not os.path.islink(candidate):
            return resolved
    return None


def _snapshot_database(db_path: str, destination: str) -> None:
    if not os.path.isfile(db_path):
        raise WorkspaceBundleError(f"database does not exist: {db_path}")
    source = sqlite3.connect(db_path)
    target = sqlite3.connect(destination)
    try:
        source.backup(target)
        target.commit()
    except sqlite3.Error as exc:
        raise WorkspaceBundleError(f"could not snapshot database: {exc}") from exc
    finally:
        target.close()
        source.close()


def _database_stats(conn: sqlite3.Connection) -> dict[str, int]:
    def count(table: str) -> int:
        try:
            return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        except sqlite3.Error:
            return 0
    return {
        "sources": count("sources"), "cards": count("cards"),
        "nodes": count("nodes"), "edges": count("edges"),
    }


def _source_manifest(snapshot: str, db_path: str, include_corpus: bool) -> list[dict[str, Any]]:
    conn = sqlite3.connect(snapshot)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT source_id, path FROM sources ORDER BY source_id"
        ).fetchall()
    except sqlite3.Error as exc:
        raise WorkspaceBundleError(f"database has no readable sources table: {exc}") from exc
    finally:
        conn.close()

    entries: list[dict[str, Any]] = []
    by_path: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw = row["path"] or ""
        resolved = _resolve_source_path(raw, db_path) if include_corpus else None
        entry: dict[str, Any] = {
            "source_id": row["source_id"],
            # Preserve portability without putting the exporting machine's
            # full local path (often including a username) in the bundle.
            "original_name": os.path.basename(raw) if raw else "",
            "included": bool(resolved), "archive_path": None,
            "size": 0, "sha256": None,
        }
        if not include_corpus:
            entry["reason"] = "corpus excluded by export option"
        elif not resolved:
            entry["reason"] = "source file was not found or was a symlink"
        else:
            if resolved in by_path:
                entry.update({key: by_path[resolved][key] for key in (
                    "archive_path", "size", "sha256")})
            else:
                archive_path = (
                    f"corpus/{_safe_filename(str(row['source_id']))}-"
                    f"{_safe_filename(os.path.basename(resolved))}"
                )
                size = os.path.getsize(resolved)
                digest = _hash_file(resolved)
                entry.update({"archive_path": archive_path, "size": size,
                              "sha256": digest})
                entry["_local_path"] = resolved
                by_path[resolved] = entry
        entries.append(entry)
    return entries


def export_bundle(db_path: str, bundle_path: str, *,
                  browser_state: dict[str, Any] | None = None,
                  include_corpus: bool = True) -> dict[str, Any]:
    """Create a validated, atomic workspace bundle and return its manifest."""
    db_path = os.path.abspath(db_path)
    bundle_path = os.path.abspath(bundle_path)
    output_dir = os.path.dirname(bundle_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    normalized_state = _normalize_browser_state(browser_state)
    staging = tempfile.mkdtemp(prefix=".card-scraper-export-", dir=output_dir)
    temporary_output = os.path.join(staging, "workspace.zip")
    snapshot = os.path.join(staging, DATABASE_MEMBER)
    try:
        _snapshot_database(db_path, snapshot)
        sources = _source_manifest(snapshot, db_path, include_corpus)
        # `_local_path` is an export-time detail only. Never serialize it into
        # the portable manifest, both for privacy and because it is meaningless
        # on the receiving machine.
        source_paths = {
            item["archive_path"]: item["_local_path"]
            for item in sources if item.get("included") and item.get("_local_path")
        }
        for item in sources:
            item.pop("_local_path", None)
        extras: list[dict[str, Any]] = []
        analysis = os.path.join(os.path.dirname(db_path), "analysis.json")
        if os.path.isfile(analysis) and not os.path.islink(analysis):
            extras.append({"path": "analysis.json", "size": os.path.getsize(analysis),
                           "sha256": _hash_file(analysis), "local_path": analysis})
        database = {"path": DATABASE_MEMBER, "size": os.path.getsize(snapshot),
                    "sha256": _hash_file(snapshot)}
        state_meta = None
        state_bytes = None
        if normalized_state is not None:
            state_bytes = _json_bytes(normalized_state)
            state_meta = {"path": STATE_MEMBER, "size": len(state_bytes),
                          "sha256": _hash_bytes(state_bytes)}
        manifest: dict[str, Any] = {
            "format": BUNDLE_FORMAT, "version": BUNDLE_VERSION,
            "created_at": _now(), "database": database,
            "sources": sources, "extra_files": [
                {key: value for key, value in item.items() if key != "local_path"}
                for item in extras
            ],
            "browser_state": state_meta,
            "includes_corpus": include_corpus,
        }

        with zipfile.ZipFile(temporary_output, "w", zipfile.ZIP_DEFLATED,
                             allowZip64=True, compresslevel=6) as archive:
            archive.write(snapshot, DATABASE_MEMBER)
            written: set[str] = {DATABASE_MEMBER}
            for item in sources:
                if not item["included"] or item["archive_path"] in written:
                    continue
                resolved = source_paths.get(item["archive_path"])
                if not resolved:
                    raise WorkspaceBundleError(
                        f"source disappeared during export: {item.get('original_name', 'unknown')}"
                    )
                before = _hash_file(resolved)
                if before != item["sha256"]:
                    raise WorkspaceBundleError(
                        f"source changed during export: {item.get('original_name', 'unknown')}"
                    )
                archive.write(resolved, item["archive_path"])
                after = _hash_file(resolved)
                if after != before:
                    raise WorkspaceBundleError(
                        f"source changed during export: {item.get('original_name', 'unknown')}"
                    )
                written.add(item["archive_path"])
            for item in extras:
                archive.write(item["local_path"], item["path"])
            if state_bytes is not None:
                archive.writestr(STATE_MEMBER, state_bytes)
            archive.writestr("manifest.json", json.dumps(
                manifest, ensure_ascii=False, sort_keys=True, indent=2,
            ).encode("utf-8"))
        os.replace(temporary_output, bundle_path)
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _file_metadata(value: Any, label: str) -> tuple[int, str]:
    if not isinstance(value, dict):
        raise WorkspaceBundleError(f"manifest has no metadata for {label}")
    try:
        size = int(value["size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkspaceBundleError(f"manifest has an invalid size for {label}") from exc
    digest = value.get("sha256")
    if size < 0 or size > MAX_MEMBER_BYTES or not isinstance(digest, str) \
            or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise WorkspaceBundleError(f"manifest has invalid integrity data for {label}")
    return size, digest


def _read_manifest(archive: zipfile.ZipFile) -> dict[str, Any]:
    infos = archive.infolist()
    if len(infos) > MAX_MEMBERS:
        raise WorkspaceBundleError("bundle contains too many files")
    names: set[str] = set()
    total = 0
    for info in infos:
        if info.filename in names:
            raise WorkspaceBundleError(f"duplicate ZIP member: {info.filename}")
        names.add(info.filename)
        if not _safe_member(info.filename):
            raise WorkspaceBundleError(f"unsafe ZIP member: {info.filename}")
        mode = (info.external_attr >> 16) & 0xFFFF
        if stat.S_ISLNK(mode):
            raise WorkspaceBundleError(f"symlink ZIP member is not allowed: {info.filename}")
        if info.file_size > MAX_MEMBER_BYTES:
            raise WorkspaceBundleError(f"ZIP member is too large: {info.filename}")
        total += info.file_size
    if total > MAX_TOTAL_BYTES:
        raise WorkspaceBundleError("bundle expands beyond the safety limit")
    try:
        raw = archive.read("manifest.json")
    except KeyError as exc:
        raise WorkspaceBundleError("bundle is missing manifest.json") from exc
    if len(raw) > MAX_MANIFEST_BYTES:
        raise WorkspaceBundleError("manifest is too large")
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WorkspaceBundleError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != BUNDLE_FORMAT \
            or manifest.get("version") != BUNDLE_VERSION:
        raise WorkspaceBundleError("unsupported Card Scraper workspace bundle")
    database = manifest.get("database")
    if not isinstance(database, dict) or database.get("path") != DATABASE_MEMBER:
        raise WorkspaceBundleError("manifest has no supported database snapshot")
    _file_metadata(database, DATABASE_MEMBER)
    if not isinstance(manifest.get("sources"), list):
        raise WorkspaceBundleError("manifest sources must be a list")
    if not isinstance(manifest.get("extra_files", []), list):
        raise WorkspaceBundleError("manifest extra_files must be a list")
    if not isinstance(manifest.get("includes_corpus"), bool):
        raise WorkspaceBundleError("manifest has no corpus inclusion flag")
    state = manifest.get("browser_state")
    if state is not None:
        if not isinstance(state, dict) or state.get("path") != STATE_MEMBER:
            raise WorkspaceBundleError("manifest has an invalid browser-state entry")
        _file_metadata(state, STATE_MEMBER)
    for item in manifest["sources"]:
        if not isinstance(item, dict) or not isinstance(item.get("source_id"), str) \
                or not item["source_id"]:
            raise WorkspaceBundleError("manifest has an invalid source entry")
        if not isinstance(item.get("included"), bool):
            raise WorkspaceBundleError("manifest source has no inclusion flag")
        if item["included"]:
            _file_metadata(item, str(item["source_id"]))
        elif item.get("archive_path") is not None:
            raise WorkspaceBundleError("excluded source unexpectedly has an archive path")
    for item in manifest.get("extra_files", []):
        if not isinstance(item, dict) or not _safe_member(item.get("path", "")):
            raise WorkspaceBundleError("manifest has an invalid extra file")
        _file_metadata(item, str(item["path"]))
    return manifest


def _manifest_members(manifest: dict[str, Any]) -> set[str]:
    members = {"manifest.json", DATABASE_MEMBER}
    reserved = {"manifest.json", DATABASE_MEMBER, STATE_MEMBER}
    state = manifest.get("browser_state")
    if state:
        if not isinstance(state, dict) or state.get("path") != STATE_MEMBER:
            raise WorkspaceBundleError("manifest has an invalid browser-state entry")
        members.add(STATE_MEMBER)
    source_paths: dict[str, tuple[int, str]] = {}
    for item in manifest.get("sources", []):
        if not isinstance(item, dict) or not isinstance(item.get("source_id"), str):
            raise WorkspaceBundleError("manifest has an invalid source entry")
        archive_path = item.get("archive_path")
        if item.get("included"):
            if not isinstance(archive_path, str) or not archive_path.startswith("corpus/"):
                raise WorkspaceBundleError("included source has no safe archive path")
            if not _safe_member(archive_path):
                raise WorkspaceBundleError("included source path is unsafe")
            metadata = _file_metadata(item, str(item["source_id"]))
            if archive_path in reserved:
                raise WorkspaceBundleError(f"source uses a reserved member: {archive_path}")
            previous = source_paths.setdefault(archive_path, metadata)
            if previous != metadata:
                raise WorkspaceBundleError(
                    f"source archive path has conflicting hashes: {archive_path}"
                )
            members.add(archive_path)
    for item in manifest.get("extra_files", []):
        if not isinstance(item, dict) or not _safe_member(item.get("path", "")):
            raise WorkspaceBundleError("manifest has an invalid extra file")
        path = item["path"]
        if path in reserved or path in members:
            raise WorkspaceBundleError(f"duplicate or reserved bundle member: {path}")
        _file_metadata(item, path)
        members.add(path)
    return members


def _extract_verified(archive: zipfile.ZipFile, member: str, destination: str,
                      expected_size: int, expected_sha256: str) -> None:
    try:
        info = archive.getinfo(member)
    except KeyError as exc:
        raise WorkspaceBundleError(f"bundle is missing {member}") from exc
    if info.file_size != expected_size:
        raise WorkspaceBundleError(f"size mismatch for {member}")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    digest = hashlib.sha256()
    try:
        with archive.open(info, "r") as source, open(destination, "wb") as target:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                target.write(chunk)
                digest.update(chunk)
    except (OSError, zipfile.BadZipFile) as exc:
        raise WorkspaceBundleError(f"could not extract {member}: {exc}") from exc
    if digest.hexdigest() != expected_sha256:
        raise WorkspaceBundleError(f"integrity check failed for {member}")


def _validate_staged_database(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    result = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if result != "ok":
        conn.close()
        raise WorkspaceBundleError(f"database integrity check failed: {result}")
    required = {"sources", "cards", "nodes", "edges", "cards_fts", "fts_map"}
    found = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE name IN (%s)" %
        ",".join("?" * len(required)), tuple(required)
    )}
    if not required.issubset(found):
        conn.close()
        missing = ", ".join(sorted(required - found))
        raise WorkspaceBundleError(f"database is missing tables: {missing}")
    return conn


def _filter_browser_state(state: dict[str, Any] | None,
                          conn: sqlite3.Connection) -> tuple[dict[str, Any] | None, int]:
    """Drop board pins whose cards are not present in the restored index."""
    state = _normalize_browser_state(state)
    if state is None:
        return None, 0
    card_ids = {row[0] for row in conn.execute("SELECT card_id FROM cards")}
    pinned = state["pinned"]
    retained = {card_id: card for card_id, card in pinned.items()
                if card_id in card_ids}
    dropped = len(pinned) - len(retained)
    return {
        "pinned": retained,
        "pinned_order": [card_id for card_id in state["pinned_order"]
                         if card_id in retained],
        "recent_searches": state["recent_searches"],
    }, dropped


def inspect_bundle(bundle_path: str) -> WorkspaceInspection:
    """Verify every bundle member without changing a database or corpus.

    This is the preflight used by the UI and CLI. Restore repeats validation
    because inspection and restore are separate operations and the file may be
    replaced between them.
    """
    bundle_path = os.path.abspath(bundle_path)
    staging = tempfile.mkdtemp(prefix=".card-scraper-inspect-")
    try:
        try:
            archive = zipfile.ZipFile(bundle_path, "r")
        except (OSError, zipfile.BadZipFile) as exc:
            raise WorkspaceBundleError(f"could not open bundle: {exc}") from exc
        with archive:
            manifest = _read_manifest(archive)
            allowed = _manifest_members(manifest)
            actual = {info.filename for info in archive.infolist() if not info.is_dir()}
            if not actual.issubset(allowed):
                extras = ", ".join(sorted(actual - allowed)[:3])
                raise WorkspaceBundleError(f"bundle has unlisted files: {extras}")
            database = manifest["database"]
            staged_db = os.path.join(staging, DATABASE_MEMBER)
            _extract_verified(archive, DATABASE_MEMBER, staged_db,
                              int(database["size"]), str(database["sha256"]))
            for item in manifest["sources"]:
                if not item.get("included"):
                    continue
                destination = os.path.join(staging, item["archive_path"].replace("/", os.sep))
                _extract_verified(archive, item["archive_path"], destination,
                                  int(item["size"]), str(item["sha256"]))
            browser_state = None
            state = manifest.get("browser_state")
            if state:
                state_path = os.path.join(staging, STATE_MEMBER)
                _extract_verified(archive, STATE_MEMBER, state_path,
                                  int(state["size"]), str(state["sha256"]))
                try:
                    with open(state_path, encoding="utf-8") as fh:
                        browser_state = _normalize_browser_state(json.load(fh))
                except (OSError, json.JSONDecodeError) as exc:
                    raise WorkspaceBundleError(f"invalid browser state: {exc}") from exc
            for item in manifest.get("extra_files", []):
                destination = os.path.join(staging, item["path"].replace("/", os.sep))
                _extract_verified(archive, item["path"], destination,
                                  int(item["size"]), str(item["sha256"]))
        staged = _validate_staged_database(staged_db)
        try:
            for item in manifest["sources"]:
                if item.get("included") and not staged.execute(
                        "SELECT 1 FROM sources WHERE source_id=?",
                        (item["source_id"],)).fetchone():
                    raise WorkspaceBundleError(
                        f"manifest source is missing from database: {item['source_id']}"
                    )
            browser_state, dropped_pins = _filter_browser_state(browser_state, staged)
            stats = _database_stats(staged)
        finally:
            staged.close()
        return WorkspaceInspection(manifest=manifest, browser_state=browser_state,
                                   stats=stats, dropped_pins=dropped_pins)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def restore_bundle(db_path: str, bundle_path: str, *,
                   connection: sqlite3.Connection | None = None) -> WorkspaceRestore:
    """Validate and restore a bundle into ``db_path``.

    ``connection`` is used by the running API so the open Store remains valid.
    The database snapshot is backed up into that connection only after all ZIP
    members and the staged SQLite database pass integrity checks.
    """
    db_path = os.path.abspath(db_path)
    bundle_path = os.path.abspath(bundle_path)
    parent = os.path.dirname(db_path) or "."
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=".card-scraper-restore-", dir=parent)
    final_corpus: str | None = None
    owned_connection = connection is None
    target = connection or sqlite3.connect(db_path, check_same_thread=False)
    target.row_factory = sqlite3.Row
    replaced = False
    previous_db = os.path.join(staging, "previous.db")
    previous_analysis = os.path.join(staging, "previous-analysis.json")
    analysis_target = os.path.join(parent, "analysis.json")
    analysis_replaced = False
    browser_state: dict[str, Any] | None = None
    try:
        try:
            archive = zipfile.ZipFile(bundle_path, "r")
        except (OSError, zipfile.BadZipFile) as exc:
            raise WorkspaceBundleError(f"could not open bundle: {exc}") from exc
        with archive:
            manifest = _read_manifest(archive)
            allowed = _manifest_members(manifest)
            actual = {info.filename for info in archive.infolist()
                      if not info.is_dir()}
            if not actual.issubset(allowed):
                extras = ", ".join(sorted(actual - allowed)[:3])
                raise WorkspaceBundleError(f"bundle has unlisted files: {extras}")
            database = manifest["database"]
            staged_db = os.path.join(staging, DATABASE_MEMBER)
            _extract_verified(archive, DATABASE_MEMBER, staged_db,
                              int(database["size"]), str(database["sha256"]))

            staged_entries: dict[str, str] = {}
            for item in manifest["sources"]:
                if not item.get("included"):
                    continue
                archive_path = item["archive_path"]
                if archive_path in staged_entries:
                    continue
                destination = os.path.join(staging, archive_path.replace("/", os.sep))
                _extract_verified(archive, archive_path, destination,
                                  int(item["size"]), str(item["sha256"]))
                staged_entries[archive_path] = destination

            state_meta = manifest.get("browser_state")
            if state_meta:
                state_path = os.path.join(staging, STATE_MEMBER)
                _extract_verified(archive, STATE_MEMBER, state_path,
                                  int(state_meta["size"]), str(state_meta["sha256"]))
                try:
                    with open(state_path, encoding="utf-8") as fh:
                        browser_state = _normalize_browser_state(json.load(fh))
                except (OSError, json.JSONDecodeError) as exc:
                    raise WorkspaceBundleError(f"invalid browser state: {exc}") from exc

            for item in manifest.get("extra_files", []):
                destination = os.path.join(staging, item["path"].replace("/", os.sep))
                _extract_verified(archive, item["path"], destination,
                                  int(item["size"]), str(item["sha256"]))

        staged = _validate_staged_database(staged_db)
        dropped_pins = 0
        try:
            bundle_id = uuid.uuid4().hex[:12]
            included = [item for item in manifest["sources"] if item.get("included")]
            if included:
                final_corpus = os.path.join(parent, "corpus", "imported", bundle_id)
                os.makedirs(os.path.dirname(final_corpus), exist_ok=True)
                stage_corpus = os.path.join(staging, "corpus")
                os.replace(stage_corpus, final_corpus)
                for item in included:
                    source_id = item["source_id"]
                    archive_path = item["archive_path"]
                    imported_path = os.path.join(final_corpus,
                                                 archive_path.removeprefix("corpus/")
                                                 .replace("/", os.sep))
                    exists = staged.execute(
                        "SELECT 1 FROM sources WHERE source_id=?", (source_id,)
                    ).fetchone()
                    if not exists:
                        raise WorkspaceBundleError(
                            f"manifest source is missing from database: {source_id}"
                        )
                    staged.execute("UPDATE sources SET path=? WHERE source_id=?",
                                   (imported_path, source_id))
                    staged.execute("UPDATE cards SET source_path=? WHERE source_id=?",
                                   (imported_path, source_id))
                staged.commit()
            if staged.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise WorkspaceBundleError("rewritten database failed integrity check")
            browser_state, dropped_pins = _filter_browser_state(browser_state, staged)
        finally:
            staged.close()

        # Keep a rollback snapshot for the small window after the live backup.
        previous = sqlite3.connect(previous_db)
        try:
            target.backup(previous)
            previous.commit()
        finally:
            previous.close()

        source = sqlite3.connect(staged_db)
        try:
            source.backup(target)
            target.commit()
        finally:
            source.close()
        replaced = True

        analysis_item = next((item for item in manifest.get("extra_files", [])
                              if item["path"] == "analysis.json"), None)
        if analysis_item:
            if os.path.isfile(analysis_target):
                shutil.copy2(analysis_target, previous_analysis)
            os.replace(os.path.join(staging, "analysis.json"), analysis_target)
            analysis_replaced = True

        return WorkspaceRestore(
            manifest=manifest, browser_state=browser_state,
            imported_files=len(included), stats=_database_stats(target),
            dropped_pins=dropped_pins,
        )
    except Exception:
        if replaced:
            try:
                old = sqlite3.connect(previous_db)
                try:
                    old.backup(target)
                    target.commit()
                finally:
                    old.close()
            except sqlite3.Error:
                # Preserve the original exception; the target database has
                # already passed the SQLite backup boundary.
                pass
        if analysis_replaced:
            try:
                if os.path.isfile(previous_analysis):
                    os.replace(previous_analysis, analysis_target)
                elif os.path.isfile(analysis_target):
                    os.remove(analysis_target)
            except OSError:
                pass
        if final_corpus:
            shutil.rmtree(final_corpus, ignore_errors=True)
        raise
    finally:
        if owned_connection:
            target.close()
        shutil.rmtree(staging, ignore_errors=True)


def bundle_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return safe, concise metadata for CLI/API responses."""
    included = sum(1 for item in manifest.get("sources", []) if item.get("included"))
    return {
        "format": manifest.get("format"), "version": manifest.get("version"),
        "created_at": manifest.get("created_at"),
        "sources": len(manifest.get("sources", [])),
        "source_files": included,
        "includes_corpus": bool(manifest.get("includes_corpus")),
        "has_browser_state": bool(manifest.get("browser_state")),
    }
