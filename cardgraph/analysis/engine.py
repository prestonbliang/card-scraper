"""The analysis orchestrator.

Ordering is the design. Deterministic checks run first and unconditionally;
model calls run second, over positions selected by the deterministic pass, and
receive its findings so they do not re-derive them. If no model is configured
the run still produces a real report — smaller, entirely offline, and honest
about what is missing rather than silently degraded.

Cost control matters here because a corpus is thousands of positions and a naive
implementation would call a model on every one. `max_positions` selects by
deterministic priority, so the model's budget goes to the positions that are
both load-bearing and already showing stress.
"""

from __future__ import annotations

import json
from collections import defaultdict

from ..index.search import SearchEngine
from ..index.store import Store
from ..models import answers_target
from ..llm import LLM
from . import deterministic as det
from . import llm_analyzers as lla
from .grounding import grounding_report
from .schema import ContentionAnalysis, CorpusReport, Severity

# Below this many cards, similarity is too noisy to support a coverage claim.
SMALL_CORPUS_CARDS = 200


def _positions(store: Store, owner: str | None = None) -> list[dict]:
    """Nodes that represent a position someone runs: has cards, is not itself
    an answer block.

    `owner` restricts to sources whose path matches a pattern, which is what
    makes the analysis mean anything on a shared corpus. Every check here is
    phrased as "your files" -- your sources, your contradictions, the answers
    *you* have no response to. Run unscoped over the published archive that
    becomes 46,712 positions belonging to 11,643 different teams and 129,410
    findings about other people's evidence, which is not a report anyone can
    act on. Scoping is not a display convenience; it is what the questions
    assume.
    """
    sql = "SELECT * FROM nodes WHERE card_count > 0"
    params: list = []
    if owner:
        sql += (" AND source_id IN (SELECT source_id FROM sources "
                "WHERE path LIKE ? OR title LIKE ?)")
        params += [f"%{owner}%", f"%{owner}%"]
    sql += " ORDER BY card_count DESC"
    out = []
    for r in store.conn.execute(sql, params):
        n = dict(r)
        if answers_target(n["title"]):
            continue
        n["path"] = json.loads(n.pop("path_json") or "[]")
        out.append(n)
    return out


def _existing_answers(store: Store, node_id: str) -> list[str]:
    rows = store.conn.execute(
        """SELECT n.title FROM edges e JOIN nodes n ON n.node_id = e.src_node_id
           WHERE e.kind='answers' AND e.dst_node_id=?""", (node_id,)).fetchall()
    return [r["title"] for r in rows]


def _side_of(cards: list[dict]) -> str:
    counts = defaultdict(int)
    for c in cards:
        counts[c.get("side") or "unknown"] += 1
    counts.pop("unknown", None)
    if not counts:
        return "unknown"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def analyze(
    store: Store,
    *,
    llm: LLM | None = None,
    max_positions: int = 8,
    min_cards: int = 1,
    owner: str | None = None,
    use_llm: bool = True,
    generate_blocks: bool = True,
    coverage_floor: float = 0.35,
    engine: SearchEngine | None = None,
    progress=None,
) -> CorpusReport:
    """Run the full pipeline. Safe to call with no model configured."""
    say = progress or (lambda *_: None)
    report = CorpusReport()

    llm = llm if llm is not None else LLM()
    have_llm = bool(use_llm and llm.available)
    report.llm_provider = llm.provider_name if have_llm else "none"
    if not have_llm:
        report.warnings.append(
            "No LLM provider configured — chain extraction, warrant critique "
            "and block generation were skipped. Set ANTHROPIC_API_KEY, or "
            "install the `claude` CLI, then re-run. Deterministic findings "
            "below are complete and unaffected.")

    # ---- deterministic, always -------------------------------------------
    say("running deterministic checks")
    newest = det.corpus_newest_year(store)
    report.corpus_findings = det.analyze_corpus(store, owner=owner)
    if owner:
        say(f"scoped to sources matching {owner!r}")

    # Unfilled outline templates are excluded: they are tags without evidence,
    # and scoring them produces a wall of true-but-useless findings that buries
    # the real ones. check_template_sources reports them once, above.
    templates = set(det.template_sources(store))
    positions = [p for p in _positions(store, owner=owner)
                 if p["card_count"] >= min_cards
                 and p.get("source_id") not in templates]
    if templates:
        say(f"excluded {len(templates)} unfilled outline source(s)")
    analyses: list[ContentionAnalysis] = []
    for pos in positions:
        cards = store.cards_for_node(pos["node_id"])
        if not cards:
            continue
        findings = det.analyze_position(pos["title"], cards, newest)
        support, exposure = det.score_position(cards, findings)
        years = [c["cite_year"] for c in cards if c.get("cite_year")]
        ca = ContentionAnalysis(
            node_id=pos["node_id"], title=pos["title"], side=_side_of(cards),
            findings=findings, card_count=len(cards),
            distinct_sources=len(det._distinct_sources(cards)),
            newest_year=max(years) if years else None,
            oldest_year=min(years) if years else None,
            support_score=support, exposure_score=exposure,
        )
        # What to fix first: exposed positions that carry weight. A broken
        # position nobody runs is not urgent; a broken position with eight
        # cards under it is.
        weight = min(len(cards) / 6.0, 1.0)
        ca.priority = round(0.65 * exposure + 0.35 * weight, 3)
        analyses.append(ca)

    analyses.sort(key=lambda a: -a.priority)
    say(f"{len(analyses)} positions scored")

    # ---- model-backed, on the top slice ----------------------------------
    if have_llm and analyses:
        if engine is None:
            engine = SearchEngine(store)
            engine.build()

        def make_search_fn(own_ids: set[str]):
            # Coverage check, not ranked search, and never against the
            # position's own cards -- see SearchEngine.covers.
            def fn(q: str):
                return engine.covers(q, exclude_card_ids=own_ids,
                                     floor=coverage_floor, k=4)
            return fn

        # On a small index the SVD space is degenerate and cosines inflate,
        # so every proposal looks half-covered. Below this many cards we refuse
        # to claim coverage at all and downgrade every match to "partial".
        total_cards = store.stats()["cards"]
        small_corpus = total_cards < SMALL_CORPUS_CARDS
        if small_corpus:
            report.warnings.append(
                f"Corpus has only {total_cards} cards; similarity scores are "
                f"not reliable at this size, so no proposed block is reported "
                f"as covered — all matches are shown as 'possibly related'. "
                f"Ingest more files for coverage claims to mean anything.")

        targets = analyses[:max_positions]
        say(f"model pass over top {len(targets)} position(s) "
            f"via {llm.provider_name}")
        for i, ca in enumerate(targets, 1):
            cards = store.cards_for_node(ca.node_id)
            say(f"  [{i}/{len(targets)}] {ca.title}")

            thesis, links, warns = lla.extract_chain(llm, ca.title, ca.side, cards)
            ca.thesis, ca.chain = thesis, links
            report.warnings.extend(f"{ca.title}: {w}" for w in warns)
            ca.findings.extend(lla.chain_findings(ca.title, links))

            crit, warns = lla.critique_position(llm, ca.title, ca.side, cards,
                                                ca.findings)
            ca.findings.extend(crit)
            report.warnings.extend(f"{ca.title}: {w}" for w in warns)

            if generate_blocks:
                own = {c["card_id"] for c in cards}
                blocks, warns = lla.generate_blocks(
                    llm, ca.title, ca.side, cards,
                    _existing_answers(store, ca.node_id),
                    search_fn=make_search_fn(own), small_corpus=small_corpus)
                report.generated_blocks.extend(blocks)
                report.warnings.extend(f"{ca.title}: {w}" for w in warns)

            # A verdict is only worth generating once every finding for this
            # position is in, since it summarizes them.
            ca.verdict = lla.summarize_position(llm, ca.title, ca.thesis,
                                                ca.findings)

            # rescore now that model findings are in
            support, exposure = det.score_position(cards, ca.findings)
            ca.support_score, ca.exposure_score = support, exposure
            weight = min(len(cards) / 6.0, 1.0)
            ca.priority = round(0.65 * exposure + 0.35 * weight, 3)

        # Same argument proposed against two positions is still one block.
        report.generated_blocks = lla.dedupe_blocks(report.generated_blocks)
        analyses.sort(key=lambda a: -a.priority)
        report.llm_usage = llm.ledger.summary()

    report.contentions = analyses

    model_findings = [f for a in analyses for f in a.findings
                      if a.findings and f.analyzer.startswith("llm:")]
    report.stats = {
        "positions_analyzed": len(analyses),
        "positions_model_analyzed": min(max_positions, len(analyses)) if have_llm else 0,
        "total_findings": len(report.all_findings()),
        "critical": sum(1 for f in report.all_findings()
                        if f.severity is Severity.CRITICAL),
        "major": sum(1 for f in report.all_findings()
                     if f.severity is Severity.MAJOR),
        "owner": owner,
        "generated_blocks": len(report.generated_blocks),
        "blocks_already_covered": sum(1 for b in report.generated_blocks if b.have_it),
        "blocks_partial": sum(1 for b in report.generated_blocks
                              if b.coverage == "partial"),
        "blocks_missing": sum(1 for b in report.generated_blocks
                              if b.coverage == "none"),
        "grounding": grounding_report(model_findings) if model_findings else None,
    }
    return report


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

_SEV_MARK = {Severity.CRITICAL: "!!", Severity.MAJOR: " !",
             Severity.MINOR: " ·", Severity.INFO: "  "}


def _wrap(text: str, width: int) -> list[str]:
    import textwrap
    return textwrap.wrap(text, width=width) or [""]


def render_text(report: CorpusReport, max_positions: int = 10,
                verbose: bool = False, corpus_per_kind: int = 3,
                findings_per_position: int = 6) -> str:
    L: list[str] = []
    s = report.stats
    L.append("=" * 72)
    L.append("CARDGRAPH ANALYSIS")
    L.append("=" * 72)
    if s.get("owner"):
        L.append(f"scope              : sources matching {s['owner']!r}")
    else:
        L.append("scope              : every source in the index "
                 "(use --owner to restrict to your own files)")
    L.append(f"positions analyzed : {s.get('positions_analyzed', 0)}"
             f"   (model pass on {s.get('positions_model_analyzed', 0)})")
    L.append(f"findings           : {s.get('total_findings', 0)}"
             f"   critical {s.get('critical', 0)} / major {s.get('major', 0)}")
    L.append(f"model              : {report.llm_provider}"
             + (f"   [{report.llm_usage}]" if report.llm_usage else ""))
    g = s.get("grounding")
    if g:
        L.append(f"grounding          : {g['grounded']}/{g['findings']} model "
                 f"findings cite a real card ({g['grounded_rate']:.0%})"
                 + (f", {g['dropped_quotes']} quote(s) dropped"
                    if g["dropped_quotes"] else ""))
    L.append("")

    if report.corpus_findings:
        L.append("CORPUS-WIDE")
        L.append("-" * 72)
        ordered = sorted(report.corpus_findings, key=lambda x: x.sort_key)
        # A report is something a person reads. Unscoped over the archive this
        # section held 6,634 findings, which is a database dump wearing a
        # report's clothes. Show the worst few per kind and count the rest.
        by_kind: dict[str, list] = {}
        for f in ordered:
            by_kind.setdefault(f.kind, []).append(f)
        for kind, group in sorted(
                by_kind.items(), key=lambda kv: -kv[1][0].severity.rank):
            shown = group[:corpus_per_kind]
            for f in shown:
                L.append(f"{_SEV_MARK[f.severity]} {f.title}")
                L.append(f"     {f.detail}")
                if f.fix:
                    L.append(f"     fix: {f.fix}")
                L.append("")
            if len(group) > len(shown):
                L.append(f"     … and {len(group) - len(shown)} more "
                         f"{kind.replace('_', ' ')} finding(s)")
                L.append("")

    total_pos = len(report.contentions)
    L.append(f"POSITIONS, WORST FIRST"
             + (f"  (showing {min(max_positions, total_pos)} of {total_pos})"
                if total_pos > max_positions else ""))
    L.append("-" * 72)
    for ca in report.contentions[:max_positions]:
        L.append(f"\n[{ca.side}] {ca.title}")
        L.append(f"     {ca.card_count} cards · {ca.distinct_sources} sources"
                 + (f" · {ca.oldest_year}-{ca.newest_year}" if ca.newest_year else "")
                 + f" · support {ca.support_score:.2f} · exposure {ca.exposure_score:.2f}")
        if ca.thesis:
            L.append(f"     thesis: {ca.thesis}")
        if ca.verdict:
            L.append("")
            for line in _wrap(ca.verdict, 68):
                L.append(f"     {line}")
            L.append("")
        if ca.chain:
            L.append("     chain:")
            for link in ca.chain:
                mark = {"supported": "ok  ", "weak": "weak", "missing": "GAP "}[link.status]
                cites = (" [" + ", ".join(c[:8] for c in link.card_ids) + "]"
                         if link.card_ids else "")
                L.append(f"       {mark} {link.step}. {link.claim}{cites}")
                if link.note and link.status != "supported":
                    L.append(f"            {link.note}")
        if not ca.findings:
            L.append("     no findings")
        pos_findings = [f for f in sorted(ca.findings, key=lambda x: x.sort_key)
                        if verbose or f.severity is not Severity.INFO]
        hidden = max(0, len(pos_findings) - findings_per_position)
        for f in pos_findings[:findings_per_position]:
            flag = "" if f.grounded else "  [ungrounded]"
            L.append(f"{_SEV_MARK[f.severity]} {f.title}{flag}")
            L.append(f"     {f.detail}")
            if f.fix:
                L.append(f"     fix: {f.fix}")
            if verbose and f.grounding_notes:
                for n in f.grounding_notes:
                    L.append(f"     · {n}")
        if hidden:
            L.append(f"     … and {hidden} more finding(s) on this position")

    if report.generated_blocks:
        L.append("")
        L.append("ANSWERS YOU DO NOT HAVE")
        L.append("-" * 72)
        for b in sorted(report.generated_blocks,
                        key=lambda x: -x.priority.rank):
            state = {
                "have": "you already have cards for this",
                "partial": "possibly related cards — check before relying on it",
                "none": "NOT IN YOUR FILES",
            }[b.coverage]
            L.append(f"\n [{b.priority.value}] vs {b.against}")
            L.append(f"   {b.title}")
            L.append(f"   {b.argument}")
            L.append(f"   -> {state}")
            if b.matched_card_ids:
                pairs = ", ".join(
                    f"{c[:8]}({sc:.2f})" for c, sc in
                    zip(b.matched_card_ids, b.match_scores or
                        [0.0] * len(b.matched_card_ids)))
                L.append(f"      matching cards: {pairs}")
            if b.coverage != "have" and b.search_query:
                L.append(f"      go cut: \"{b.search_query}\"")

    if report.warnings:
        L.append("")
        L.append("WARNINGS")
        L.append("-" * 72)
        for w in report.warnings[:20]:
            L.append(f"  · {w}")

    return "\n".join(L)
