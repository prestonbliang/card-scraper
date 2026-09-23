"""cardgraph command line.

    python -m cardgraph.cli ingest local ./my-files
    python -m cardgraph.cli ingest git https://github.com/ashtarcommunications/caselist-archive
    python -m cardgraph.cli ingest online https://openev.debatecoaches.org/
    python -m cardgraph.cli ingest-opendebate --query "data center" --limit 2000
    python -m cardgraph.cli graph
    python -m cardgraph.cli analyze --top 5
    python -m cardgraph.cli analyze --owner Greenhill --top 5
    python -m cardgraph.cli analyze --no-llm --json report.json
    python -m cardgraph.cli search "households subsidize industrial load"
    python -m cardgraph.cli packet --query "grid expansion permits" -k 12 --out packet.zip
    python -m cardgraph.cli packet --ids <card_id> [<card_id> ...]
    python -m cardgraph.cli stats
    python -m cardgraph.cli policy
    python -m cardgraph.cli workspace export data/card-scraper-workspace.zip
    python -m cardgraph.cli workspace import data/card-scraper-workspace.zip
    python -m cardgraph.cli serve
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from .graph.relate import build_all
from .index.search import SearchEngine, parse_smart_request
from .index.store import Store
from .ingest.base import (CaselistArchiveAdapter, GitRepoAdapter,
                          LocalDirAdapter, OnlineEvidenceAdapter,
                          OpenEvidenceAdapter)
from .ingest.policy import (AccessPolicy, AccessRefused, explain_allowlist,
                            source_catalog)
from .parse.docx_card import UnsupportedFormat, parse_any
from .packet import PacketError
from .workspace import (WorkspaceBundleError, bundle_summary, export_bundle,
                         inspect_bundle, restore_bundle)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _year(value: str) -> int:
    parsed = int(value)
    if not 1900 <= parsed <= 2200:
        raise argparse.ArgumentTypeError("must be between 1900 and 2200")
    return parsed


def _ratio(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def cmd_ingest(args) -> int:
    store = Store(args.db)
    policy = AccessPolicy()

    if args.kind == "local":
        adapter = LocalDirAdapter(args.target, workdir=args.workdir, policy=policy)
    elif args.kind == "git":
        adapter = GitRepoAdapter(args.target, workdir=args.workdir, policy=policy)
    elif args.kind == "caselist-archive":
        adapter = CaselistArchiveAdapter(workdir=args.workdir, policy=policy)
    elif args.kind == "openev":
        adapter = OpenEvidenceAdapter(
            index_url=args.target or "https://openev.debatecoaches.org/",
            stealth=args.stealth, workdir=args.workdir, policy=policy,
        )
    elif args.kind == "online":
        adapter = OnlineEvidenceAdapter(index_url=args.target,
                                        license=args.license,
                                        stealth=args.stealth,
                                        workdir=args.workdir, policy=policy)
    else:
        print(f"unknown source kind: {args.kind}", file=sys.stderr)
        return 2

    try:
        acquired = adapter.acquire(limit=args.limit)
    except AccessRefused as exc:
        print("\nREFUSED BY POLICY\n", exc, file=sys.stderr)
        return 3

    if not acquired:
        print("nothing to ingest (no .docx or .htm found)")
        return 0

    # Resume by default. A full archive ingest takes tens of minutes; without
    # this, one interruption at file 14,000 means starting over. Sources are
    # keyed on the absolute path, so re-running only picks up what is new.
    done: set[str] = set()
    if not args.force:
        done = {r[0] for r in store.conn.execute(
            "SELECT source_id FROM sources WHERE card_count > 0")}
        skipped = sum(1 for a in acquired if a.source.source_id in done)
        if skipped:
            print(f"resuming: {skipped} of {len(acquired)} files already "
                  f"ingested (--force to redo)")
            acquired = [a for a in acquired
                        if a.source.source_id not in done]
            if not acquired:
                print("nothing new to ingest")
                return 0

    total_cards = 0
    low_coverage: list[str] = []
    failures = 0
    skipped_fmt = 0
    skip_reason = ""
    unmarked = 0
    started = time.monotonic()
    every = max(1, min(200, len(acquired) // 20 or 1))
    verbose = len(acquired) <= 50

    for i, item in enumerate(acquired, 1):
        try:
            cards, root, report = parse_any(item.path, item.source.source_id)
        except UnsupportedFormat as exc:
            # Not a failure: a format we do not claim to handle.
            skipped_fmt += 1
            skip_reason = skip_reason or exc.reason
            continue
        except Exception as exc:  # noqa: BLE001
            failures += 1
            if verbose:
                print(f"  ! parse failed {os.path.basename(item.path)}: {exc}")
            continue
        item.source.card_count = len(cards)
        # Local files are versioned by size/mtime. If an existing path changed,
        # remove its previous cards before inserting the successful replacement;
        # otherwise stale evidence remains searchable forever.
        old_ids = [r[0] for r in store.conn.execute(
            "SELECT source_id FROM sources WHERE path=? AND source_id != ?",
            (item.source.path, item.source.source_id),
        )]
        if args.force and store.conn.execute(
                "SELECT 1 FROM sources WHERE source_id=?", (item.source.source_id,)
        ).fetchone():
            old_ids.append(item.source.source_id)
        for old_id in old_ids:
            store.remove_source(old_id)
        store.add_source(item.source)
        added = store.add_cards(cards)
        store.add_outline(root)
        total_cards += added
        if report.unmarked_source:
            unmarked += 1

        if verbose:
            print(f"  {report.summary()}  (+{added} new)")
            for w in report.warnings:
                print(f"      warning: {w}")
        elif i % every == 0 or i == len(acquired):
            # A long ingest with no output is indistinguishable from a hang.
            elapsed = time.monotonic() - started
            rate = i / elapsed if elapsed else 0
            eta = (len(acquired) - i) / rate if rate else 0
            print(f"  [{i}/{len(acquired)}] {total_cards} cards · "
                  f"{rate:.1f} files/s · eta {eta / 60:.1f} min"
                  + (f" · {failures} failed" if failures else "")
                  + (f" · {skipped_fmt} skipped" if skipped_fmt else ""),
                  flush=True)

        if report.cards and report.read_text_coverage < 0.5 \
                and not report.unmarked_source:
            low_coverage.append(item.path)

    elapsed = time.monotonic() - started
    parsed = len(acquired) - failures - skipped_fmt
    print(f"\ningested {parsed} of {len(acquired)} files in "
          f"{elapsed / 60:.1f} min, {total_cards} new cards"
          + (f", {failures} failed" if failures else ""))
    if skipped_fmt:
        print(f"{skipped_fmt} file(s) skipped as an unsupported format -- "
              f"{skip_reason}")
    if unmarked:
        print(f"{unmarked} file(s) contained no highlighting at all -- normal "
              f"for open-source disclosure uploads; their cards fall back to "
              f"full-body search.")
    if low_coverage:
        print(f"\n{len(low_coverage)} file(s) had poor read-text coverage. "
              f"Open one and check how it marks the read portion before you "
              f"trust search results from them:")
        for p in low_coverage[:5]:
            print(f"  - {p}")
    return 0


def cmd_analyze(args) -> int:
    from .analysis import analyze, render_text

    store = Store(args.db)
    if store.stats()["cards"] == 0:
        print("no cards indexed -- run `ingest` first", file=sys.stderr)
        return 1
    report = analyze(
        store, max_positions=args.top, min_cards=args.min_cards,
        use_llm=not args.no_llm, generate_blocks=not args.no_blocks,
        coverage_floor=args.coverage_floor, owner=args.owner,
        progress=(lambda *a: print("..", *a, flush=True)) if args.verbose else None,
    )
    if args.json:
        import json as _json
        with open(args.json, "w") as fh:
            _json.dump(report.to_dict(), fh, indent=2)
        print(f"wrote {args.json}")
    print(render_text(report, max_positions=args.top, verbose=args.verbose))
    return 0


def cmd_ingest_opendebate(args) -> int:
    from .ingest.opendebate import DEDUPED, FULL, DatasetFilter, ingest

    store = Store(args.db)
    filt = DatasetFilter(
        query=args.query, year_min=args.year_min, year_max=args.year_max,
        event=args.event, side=args.side,
        min_duplicate_count=args.min_duplicates,
    )
    result = ingest(store, dataset=(FULL if args.full else DEDUPED), filt=filt,
                    limit=args.limit,
                    progress=lambda *a: print("..", *a, flush=True))
    for k, v in result.items():
        print(f"{k}: {v}")
    return 0


def cmd_graph(args) -> int:
    store = Store(args.db)
    result = build_all(store)
    for k, v in result.items():
        print(f"{k}: {v}")
    return 0


def cmd_search(args) -> int:
    if args.year_min is not None and args.year_max is not None \
            and args.year_min > args.year_max:
        print("year-min must not exceed year-max", file=sys.stderr)
        return 2
    store = Store(args.db)
    engine = SearchEngine(store)
    engine.build()
    if args.smart:
        hits, variants = engine.smart_search(
            args.query, k=args.k, side=args.side, author=args.author,
            year_min=args.year_min, year_max=args.year_max, block=args.block,
            min_read_ratio=args.min_read_ratio, source=args.source,
            mode=args.mode,
        )
        interpreted_query, inferred = parse_smart_request(args.query)
        print("smart query: " + interpreted_query)
        if inferred:
            print("smart filters: " + ", ".join(f"{k}={v}" for k, v in inferred.items()))
        print("smart queries: " + " | ".join(variants))
    else:
        hits = engine.search(
            args.query, k=args.k, side=args.side, author=args.author,
            year_min=args.year_min, year_max=args.year_max, block=args.block,
            min_read_ratio=args.min_read_ratio, source=args.source,
            mode=args.mode,
        )
    if not hits:
        print("no results")
        return 0
    for i, h in enumerate(hits, 1):
        print(f"\n{i}. [{h.side}] {h.tag}")
        print(f"   cite : {h.cite_raw[:110]}")
        print(f"   source: {h.source_title or h.source_origin or h.source_id}")
        if h.source_url:
            print(f"   url   : {h.source_url}")
        print(f"   read : {h.read_text[:220]}")
        print(f"   match: {h.confidence} · {h.match_type}  (lex #{h.lexical_rank}, vec #{h.vector_rank})")
        print(f"   evidence: {h.evidence_status} · read coverage {h.read_ratio:.0%}")
        print(f"   score: {h.score:.4f}")
    return 0


def cmd_packet(args) -> int:
    from .packet import PacketError, write_packet

    if args.year_min is not None and args.year_max is not None \
            and args.year_min > args.year_max:
        print("year-min must not exceed year-max", file=sys.stderr)
        return 2
    if args.ids and args.query:
        print("pass either --ids or a query, not both", file=sys.stderr)
        return 2
    store = Store(args.db)
    try:
        kwargs = {"query": args.query} if args.query else {"card_ids": args.ids}
        for option in ("side", "author", "block", "source"):
            value = getattr(args, option, None)
            if value:
                kwargs[option] = value
        if args.year_min is not None:
            kwargs["year_min"] = args.year_min
        if args.year_max is not None:
            kwargs["year_max"] = args.year_max
        if args.min_read_ratio is not None:
            kwargs["min_read_ratio"] = args.min_read_ratio
        if args.query:
            kwargs["k"] = args.k
            kwargs["smart"] = args.smart
            kwargs["mode"] = args.mode
        filename, manifest = write_packet(store, args.out, **kwargs)
    except PacketError as exc:
        print(f"packet error: {exc}", file=sys.stderr)
        return 2
    finally:
        store.close()
    print(f"wrote {args.out} ({filename})")
    print(f"  cards: {manifest['card_count']}")
    for name, count in manifest["evidence_status_counts"].items():
        print(f"  {name}: {count}")
    for source_id, source in manifest["sources"].items():
        fresh = source["freshness"] + (" (refresh failed)" if source["refresh_failed"] else "")
        print(f"  source {source_id}: {len(source['cards'])} card(s), {fresh}")
    return 0


def cmd_stats(args) -> int:
    store = Store(args.db)
    for k, v in store.stats().items():
        print(f"{k:24} {v}")
    return 0


def cmd_policy(args) -> int:
    print(explain_allowlist())
    return 0


def _read_browser_state(path: str | None) -> dict | None:
    if not path:
        return None
    import json
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkspaceBundleError(f"could not read browser state: {exc}") from exc


def cmd_workspace_export(args) -> int:
    manifest = export_bundle(
        args.db, args.bundle, browser_state=_read_browser_state(args.state),
        include_corpus=not args.no_corpus,
    )
    print(f"exported {args.bundle}")
    for key, value in bundle_summary(manifest).items():
        print(f"{key}: {value}")
    return 0


def cmd_workspace_inspect(args) -> int:
    inspection = inspect_bundle(args.bundle)
    print(f"validated {args.bundle}")
    for key, value in bundle_summary(inspection.manifest).items():
        print(f"{key}: {value}")
    print(f"stats: {inspection.stats}")
    if inspection.dropped_pins:
        print(f"warning: {inspection.dropped_pins} board pin(s) refer to cards not in this bundle")
    return 0


def cmd_workspace_import(args) -> int:
    restored = restore_bundle(args.db, args.bundle)
    print(f"restored {args.bundle}")
    for key, value in bundle_summary(restored.manifest).items():
        print(f"{key}: {value}")
    print(f"imported files: {restored.imported_files}")
    print(f"stats: {restored.stats}")
    if restored.dropped_pins:
        print(f"warning: dropped {restored.dropped_pins} board pin(s) not present in the restored index")
    if restored.browser_state is not None:
        print("browser state: included (API clients can restore it locally)")
    return 0


def cmd_catalog(args) -> int:
    """Print reviewed public and gated source profiles."""
    for source in source_catalog():
        print(f"{source['name']} [{source['access']}]\n  {source['url']}\n"
              f"  {source['kind']}\n  {source['note']}\n"
              f"  command: {source['command']}\n")
    return 0


def cmd_serve(args) -> int:
    import uvicorn
    from .api.main import create_app

    uvicorn.run(create_app(args.db), host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cardgraph")
    p.add_argument("--db", default="data/cardgraph.db")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("ingest")
    i.add_argument("kind", choices=["local", "git", "caselist-archive", "openev",
                                     "online"])
    i.add_argument("target", nargs="?", default="")
    i.add_argument("--workdir", default="data/corpus")
    i.add_argument("--limit", type=_positive_int, default=None)
    i.add_argument("--license", default="verify source terms before use",
                   help="attribution/license note recorded for online sources")
    i.add_argument("--stealth", action="store_true",
                   help="use Scrapling's browser path for a public site that permits access")
    i.add_argument("--force", action="store_true",
                   help="re-ingest files already in the database")
    i.set_defaults(func=cmd_ingest)

    od = sub.add_parser("ingest-opendebate",
                        help="stream the published OpenDebateEvidence dataset")
    od.add_argument("--query", help="substring filter over tag/text")
    od.add_argument("--year-min", type=_year, dest="year_min")
    od.add_argument("--year-max", type=_year, dest="year_max")
    od.add_argument("--event", choices=["ld", "cx"])
    od.add_argument("--side", choices=["A", "N"])
    od.add_argument("--min-duplicates", type=int, dest="min_duplicates",
                    help="keep only cards many teams read (quality proxy)")
    od.add_argument("--limit", type=_positive_int, default=5000)
    od.add_argument("--full", action="store_true",
                    help="use the full dataset instead of the deduplicated one")
    od.set_defaults(func=cmd_ingest_opendebate)

    an = sub.add_parser("analyze", help="find weaknesses in your positions")
    an.add_argument("--top", type=_positive_int, default=8,
                    help="how many positions get the (paid) model pass")
    an.add_argument("--min-cards", type=_positive_int, default=1, dest="min_cards")
    an.add_argument("--no-llm", action="store_true",
                    help="deterministic checks only; fully offline")
    an.add_argument("--no-blocks", action="store_true",
                    help="skip answer-block generation")
    an.add_argument("--coverage-floor", type=_ratio, default=0.35,
                    dest="coverage_floor")
    an.add_argument("--owner", metavar="PATTERN",
                    help="restrict to sources whose path or title matches. "
                         "Essential on a shared corpus: every check asks about "
                         "'your files', and unscoped over the published archive "
                         "that means 46,712 positions belonging to 11,643 teams.")
    an.add_argument("--json", help="also write the full report to this path")
    an.add_argument("-v", "--verbose", action="store_true")
    an.set_defaults(func=cmd_analyze)

    g = sub.add_parser("graph")
    g.set_defaults(func=cmd_graph)

    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("-k", type=_positive_int, default=10)
    s.add_argument("--side", choices=["aff", "neg", "both", "unknown"])
    s.add_argument("--author")
    s.add_argument("--year-min", type=_year, dest="year_min")
    s.add_argument("--year-max", type=_year, dest="year_max")
    s.add_argument("--block", help="filter by block or contention name")
    s.add_argument("--source", help="filter by source id, title, origin, or path")
    s.add_argument("--min-read-ratio", type=_ratio, dest="min_read_ratio")
    s.add_argument("--smart", action="store_true",
                   help="expand conversational wording into transparent local queries")
    s.add_argument("--mode", choices=["strict", "balanced", "explore"],
                   default="balanced", help="retrieval precision mode")
    s.set_defaults(func=cmd_search)

    st = sub.add_parser("stats")
    st.set_defaults(func=cmd_stats)

    pk = sub.add_parser("packet", help="export selected cards as a shareable research packet (ZIP)")
    pk.add_argument("--ids", nargs="+", help="card ids (from search results or the board)")
    pk.add_argument("--query", help="use the top-k search results as the packet contents")
    pk.add_argument("-k", type=_positive_int, default=10)
    pk.add_argument("--out", default="card-scraper-packet.zip", help="destination .zip path")
    pk.add_argument("--side", choices=["aff", "neg", "both", "unknown"])
    pk.add_argument("--author")
    pk.add_argument("--year-min", type=_year, dest="year_min")
    pk.add_argument("--year-max", type=_year, dest="year_max")
    pk.add_argument("--block")
    pk.add_argument("--source")
    pk.add_argument("--min-read-ratio", type=_ratio, dest="min_read_ratio")
    pk.add_argument("--smart", action="store_true")
    pk.add_argument("--mode", choices=["strict", "balanced", "explore"], default="balanced")
    pk.set_defaults(func=cmd_packet)

    po = sub.add_parser("policy")
    po.set_defaults(func=cmd_policy)

    ws = sub.add_parser("workspace", help="portable, integrity-checked workspace bundles")
    ws_sub = ws.add_subparsers(dest="workspace_cmd", required=True)
    wse = ws_sub.add_parser("export", help="export the database, corpus, and optional browser state")
    wse.add_argument("bundle", help="destination .zip path")
    wse.add_argument("--state", help="JSON file exported from the browser workspace")
    wse.add_argument("--no-corpus", action="store_true",
                     help="export the index and metadata without source files")
    wse.set_defaults(func=cmd_workspace_export)
    wsi = ws_sub.add_parser("import", help="validate and restore a workspace bundle")
    wsi.add_argument("bundle", help="source .zip path")
    wsi.set_defaults(func=cmd_workspace_import)
    wsv = ws_sub.add_parser("inspect", help="preflight a bundle without changing the index")
    wsv.add_argument("bundle", help="source .zip path")
    wsv.set_defaults(func=cmd_workspace_inspect)

    ca = sub.add_parser("catalog", help="show reviewed evidence sources and access notes")
    ca.set_defaults(func=cmd_catalog)

    sv = sub.add_parser("serve")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=_positive_int, default=8000)
    sv.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except WorkspaceBundleError as exc:
        print(f"workspace error: {exc}", file=sys.stderr)
        return 2
    except PacketError as exc:
        print(f"packet error: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        # piping into `head` is normal usage, not an error
        try:
            sys.stdout.close()
        finally:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
