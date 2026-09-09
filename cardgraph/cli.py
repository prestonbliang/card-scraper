"""cardgraph command line.

    python -m cardgraph.cli ingest local ./my-files
    python -m cardgraph.cli ingest git https://github.com/ashtarcommunications/caselist-archive
    python -m cardgraph.cli ingest openev https://openev.debatecoaches.org/
    python -m cardgraph.cli ingest opendebate --query "data center" --limit 2000
    python -m cardgraph.cli graph
    python -m cardgraph.cli analyze --top 5
    python -m cardgraph.cli analyze --no-llm --json report.json
    python -m cardgraph.cli search "households subsidize industrial load"
    python -m cardgraph.cli stats
    python -m cardgraph.cli policy
    python -m cardgraph.cli serve
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from .graph.relate import build_all
from .index.search import SearchEngine
from .index.store import Store
from .ingest.base import (CaselistArchiveAdapter, GitRepoAdapter,
                          LocalDirAdapter, OpenEvidenceAdapter)
from .ingest.policy import AccessPolicy, AccessRefused, explain_allowlist
from .parse.docx_card import UnsupportedFormat, parse_any


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
        adapter = OpenEvidenceAdapter(index_url=args.target, workdir=args.workdir,
                                      policy=policy)
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
        coverage_floor=args.coverage_floor,
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
    store = Store(args.db)
    engine = SearchEngine(store)
    engine.build()
    hits = engine.search(args.query, k=args.k, side=args.side,
                         author=args.author, min_read_ratio=args.min_read_ratio)
    if not hits:
        print("no results")
        return 0
    for i, h in enumerate(hits, 1):
        print(f"\n{i}. [{h.side}] {h.tag}")
        print(f"   cite : {h.cite_raw[:110]}")
        print(f"   block: {h.block}")
        print(f"   read : {h.read_text[:220]}")
        print(f"   score: {h.score:.4f}  (lex #{h.lexical_rank}, vec #{h.vector_rank})")
    return 0


def cmd_stats(args) -> int:
    store = Store(args.db)
    for k, v in store.stats().items():
        print(f"{k:24} {v}")
    return 0


def cmd_policy(args) -> int:
    print(explain_allowlist())
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
    i.add_argument("kind", choices=["local", "git", "caselist-archive", "openev"])
    i.add_argument("target", nargs="?", default="")
    i.add_argument("--workdir", default="data/corpus")
    i.add_argument("--limit", type=int, default=None)
    i.add_argument("--force", action="store_true",
                   help="re-ingest files already in the database")
    i.set_defaults(func=cmd_ingest)

    od = sub.add_parser("ingest-opendebate",
                        help="stream the published OpenDebateEvidence dataset")
    od.add_argument("--query", help="substring filter over tag/text")
    od.add_argument("--year-min", type=int, dest="year_min")
    od.add_argument("--year-max", type=int, dest="year_max")
    od.add_argument("--event", choices=["ld", "cx"])
    od.add_argument("--side", choices=["A", "N"])
    od.add_argument("--min-duplicates", type=int, dest="min_duplicates",
                    help="keep only cards many teams read (quality proxy)")
    od.add_argument("--limit", type=int, default=5000)
    od.add_argument("--full", action="store_true",
                    help="use the full dataset instead of the deduplicated one")
    od.set_defaults(func=cmd_ingest_opendebate)

    an = sub.add_parser("analyze", help="find weaknesses in your positions")
    an.add_argument("--top", type=int, default=8,
                    help="how many positions get the (paid) model pass")
    an.add_argument("--min-cards", type=int, default=1, dest="min_cards")
    an.add_argument("--no-llm", action="store_true",
                    help="deterministic checks only; fully offline")
    an.add_argument("--no-blocks", action="store_true",
                    help="skip answer-block generation")
    an.add_argument("--coverage-floor", type=float, default=0.35,
                    dest="coverage_floor")
    an.add_argument("--json", help="also write the full report to this path")
    an.add_argument("-v", "--verbose", action="store_true")
    an.set_defaults(func=cmd_analyze)

    g = sub.add_parser("graph")
    g.set_defaults(func=cmd_graph)

    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=10)
    s.add_argument("--side", choices=["aff", "neg", "both", "unknown"])
    s.add_argument("--author")
    s.add_argument("--min-read-ratio", type=float, dest="min_read_ratio")
    s.set_defaults(func=cmd_search)

    st = sub.add_parser("stats")
    st.set_defaults(func=cmd_stats)

    po = sub.add_parser("policy")
    po.set_defaults(func=cmd_policy)

    sv = sub.add_parser("serve")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except BrokenPipeError:
        # piping into `head` is normal usage, not an error
        try:
            sys.stdout.close()
        finally:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
