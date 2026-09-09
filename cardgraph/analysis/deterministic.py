"""Weaknesses you can compute, no model required.

These run first, always, and they run offline. Two reasons that ordering
matters. First, a measurable finding beats a model's opinion: "this contention
has four cards and three of them are the same author" is a fact, and an
opponent can exploit it whether or not a language model noticed. Second, these
findings go into the model's prompt later, so the expensive analyzer spends its
attention on judgment rather than on counting.

Every check here is written to fire on a real, nameable failure mode from
rounds, not on a metric that merely looks tidy.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict

from ..index.store import Store
from ..models import answers_target
from ..parse.authors import author_key, distinct_authors
from .schema import Finding, Severity

# Tag language that promises a strong causal claim. Used to detect the gap
# between what the tag asserts and what the underlined text delivers.
STRONG_CLAIM = re.compile(
    r"\b(collapse[sd]?|collapsing|causes?|caused|guarantees?|ensures?|proves?|"
    r"destroys?|devastat\w+|extinction|inevitab\w+|necessar\w+|only way|"
    r"eliminat\w+|solves?|reverses?|triggers?|forces?)\b", re.IGNORECASE)

HEDGE = re.compile(
    r"\b(may|might|could|possibly|potentially|suggests?|appears?|seems?|"
    r"is likely|are likely|we estimate|projected|assum\w+|if current trends|"
    r"under some scenarios|preliminary)\b", re.IGNORECASE)

QUAL = re.compile(
    r"(prof(essor)?\b|ph\.?d|j\.?d\b|m\.?d\b|senior fellow|director\b|"
    r"analyst\b|research\w*\b|economist\b|lecturer\b|chair\b|scientist\b|"
    r"engineer\b|attorney\b|commissioner\b)", re.IGNORECASE)


def _distinct_sources(cards: list[dict]) -> set[str]:
    """How many genuinely different authors carry these cards.

    Canonicalized rather than string-matched, because "Nick Bostrom",
    "Bostrom" and "Bostrum, Nick. University" are one person and counting them
    as three makes a single-source position look well-sourced -- a false
    negative in the one check whose job is to catch exactly that.

    Cards with no parseable author each count as their own source: unknown is
    not evidence of sameness, and assuming otherwise would over-report
    concentration.
    """
    named = [c.get("cite_author") for c in cards if (c.get("cite_author") or "").strip()]
    keys = distinct_authors(named)
    anon = {f"?{c['card_id']}" for c in cards
            if not (c.get("cite_author") or "").strip()}
    return keys | anon


# ---------------------------------------------------------------------------
# per-position checks
# ---------------------------------------------------------------------------

def check_source_concentration(title: str, cards: list[dict]) -> list[Finding]:
    if len(cards) < 3:
        return []
    srcs = _distinct_sources(cards)
    ratio = len(srcs) / len(cards)
    if len(srcs) <= 2 and len(cards) >= 4:
        sev = Severity.MAJOR
    elif ratio <= 0.4:
        sev = Severity.MINOR
    else:
        return []
    top = Counter((c.get("cite_author") or "?") for c in cards).most_common(1)[0]
    return [Finding(
        kind="source_concentration", severity=sev,
        title=f"{title}: {len(cards)} cards from only {len(srcs)} distinct source(s)",
        detail=(f"{top[1]} of {len(cards)} cards cite {top[0]}. A single "
                f"author indict or a methodology press takes out most of the "
                f"position at once."),
        fix=f"Cut corroborating evidence from a different author and outlet for "
            f"the load-bearing claim in {title}.",
        card_ids=[c["card_id"] for c in cards],
        analyzer="deterministic:source_concentration",
        evidence={"cards": len(cards), "distinct_sources": len(srcs),
                  "top_source": top[0], "top_source_cards": top[1]},
    )]


def check_single_card(title: str, cards: list[dict]) -> list[Finding]:
    if len(cards) != 1:
        return []
    return [Finding(
        kind="single_card_contention", severity=Severity.MAJOR,
        title=f"{title} rests on a single card",
        detail="One card carries the whole position. If it is indicted, "
               "mis-highlighted, or simply answered, there is nothing behind it.",
        fix=f"Cut at least one independent card for {title}, ideally from a "
            f"different source type.",
        card_ids=[cards[0]["card_id"]],
        analyzer="deterministic:single_card",
        evidence={"cards": 1},
    )]


def check_power_tagging(cards: list[dict]) -> list[Finding]:
    """Tag promises causation; underlined text hedges or is too thin to deliver."""
    out: list[Finding] = []
    for c in cards:
        tag, read = c.get("tag") or "", c.get("read_text") or ""
        body = c.get("body") or ""
        rr = c.get("read_ratio") or 0.0
        if not STRONG_CLAIM.search(tag):
            continue
        # For a disclosure-only card read_text IS the body, so read_ratio is 1.0
        # by construction and carries no signal. Only the hedging test applies.
        disclosed = bool(c.get("disclosed_only"))
        reasons = []
        if read and HEDGE.search(read) and not HEDGE.search(tag):
            reasons.append("the underlined text hedges where the tag does not")
        if not disclosed and 0 < rr < 0.15:
            reasons.append(f"only {rr:.0%} of the card is underlined")
        if not disclosed and not read.strip():
            reasons.append("nothing is underlined at all")
        if not disclosed and body and HEDGE.search(body) and read and not HEDGE.search(read):
            reasons.append("the surrounding context hedges the claim in a way "
                           "the underlined portion hides")
        if not reasons:
            continue
        sev = Severity.MAJOR if len(reasons) > 1 or not read.strip() else Severity.MINOR
        out.append(Finding(
            kind="power_tagging", severity=sev,
            title=f"Tag may overclaim: {tag[:70]}",
            detail="The tag asserts a strong causal claim, but " +
                   "; ".join(reasons) + ".",
            fix="Re-read the card and either re-tag to what the underlined text "
                "actually establishes, or extend the highlighting to include "
                "the warrant.",
            card_ids=[c["card_id"]], quotes=[read[:200]] if read else [],
            analyzer="deterministic:power_tagging",
            evidence={"read_ratio": rr, "reasons": reasons},
        ))
    return out


def check_qualifications(title: str, cards: list[dict]) -> list[Finding]:
    unqualified = [c for c in cards
                   if not QUAL.search(c.get("cite_raw") or "")
                   and not (c.get("cite_pub") or "").strip()]
    if not unqualified:
        return []
    frac = len(unqualified) / len(cards)
    if len(cards) == 1 or frac >= 0.75:
        sev = Severity.MAJOR if len(cards) <= 2 else Severity.MINOR
    elif frac >= 0.5:
        sev = Severity.MINOR
    else:
        return []
    return [Finding(
        kind="qualification_gap", severity=sev,
        title=f"{title}: {len(unqualified)}/{len(cards)} cards carry no author qualification",
        detail="No credential or publication was parsed from the cite. Quals "
               "comparison is a standard press and these cards lose it by default.",
        fix="Add the author's position and outlet to the cite line, or replace "
            "with a qualified source.",
        card_ids=[c["card_id"] for c in unqualified],
        analyzer="deterministic:qualifications",
        evidence={"unqualified": len(unqualified), "total": len(cards)},
    )]


def check_read_text_health(cards: list[dict]) -> list[Finding]:
    """Read-health only means something for cards that *have* a full body.

    A caselist card is disclosed as its first and last lines by convention --
    there is no fuller body and no underlining to miss. Running these checks on
    an archive ingest produces one true, useless finding per card and buries
    everything else, so disclosure-only cards are excluded here rather than
    flagged for a defect they cannot have.
    """
    cards = [c for c in cards if not c.get("disclosed_only")]
    if not cards:
        return []
    out: list[Finding] = []
    none_read = [c for c in cards if not (c.get("read_text") or "").strip()]
    if none_read:
        out.append(Finding(
            kind="no_read_text", severity=Severity.MAJOR,
            title=f"{len(none_read)} card(s) have nothing underlined",
            detail="There is no read text on these cards. Either they were "
                   "never highlighted, or the parser could not see how this "
                   "file marks highlighting — check one in Word before "
                   "trusting anything else about them.",
            fix="Open the source file and confirm the underlining style; "
                "re-ingest if the marks are there but unread.",
            card_ids=[c["card_id"] for c in none_read],
            analyzer="deterministic:read_health",
            evidence={"count": len(none_read)},
        ))
    thin = [c for c in cards
            if 0 < (c.get("read_ratio") or 0) < 0.12
            and (c.get("read_text") or "").strip()]
    if thin:
        out.append(Finding(
            kind="thin_read", severity=Severity.MINOR,
            title=f"{len(thin)} card(s) are underlined very sparsely",
            detail="Under 12% of the body is read. Sometimes that is a tight "
                   "cut; often the warrant is in the part you skip, which is "
                   "exactly what a 'read the whole card' press exposes.",
            fix="Check that the underlined span still contains the reasoning, "
                "not just the conclusion.",
            card_ids=[c["card_id"] for c in thin],
            analyzer="deterministic:read_health",
            evidence={"count": len(thin)},
        ))
    return out


def check_recency(title: str, cards: list[dict], corpus_newest: int | None) -> list[Finding]:
    years = [c["cite_year"] for c in cards if c.get("cite_year")]
    if not years or corpus_newest is None:
        return []
    newest = max(years)
    lag = corpus_newest - newest
    if lag < 3:
        return []
    sev = Severity.MAJOR if lag >= 6 else Severity.MINOR
    return [Finding(
        kind="recency_decay", severity=sev,
        title=f"{title}: newest card is {lag} years behind the corpus",
        detail=f"The most recent evidence here is from {newest}, while the "
               f"corpus reaches {corpus_newest}. On any question where the "
               f"underlying facts move, a more recent card beats this one on "
               f"recency alone.",
        fix=f"Update the key card in {title} with post-{newest} evidence.",
        card_ids=[c["card_id"] for c in cards if c.get("cite_year") == newest],
        analyzer="deterministic:recency",
        evidence={"position_newest": newest, "corpus_newest": corpus_newest,
                  "lag_years": lag},
    )]


# ---------------------------------------------------------------------------
# corpus-level checks
# ---------------------------------------------------------------------------

def template_sources(store: Store, threshold: float = 0.9,
                     min_cards: int = 3) -> dict[str, dict]:
    """Sources that are outline templates rather than evidence.

    A file built from `seed/build_outline.py`, or any half-filled skeleton, has
    tags and headings but no cites and nothing underlined. Analyzing it produces
    a page of "this card has no read text" findings that drown the real ones,
    and every position in it scores as catastrophically weak — which is true and
    useless, because it is not a position yet.

    So they are detected, excluded from position analysis, and reported once.
    Returns {source_id: {title, cards, filled}}.
    """
    out: dict[str, dict] = {}
    rows = store.conn.execute(
        """SELECT s.source_id, s.title, COUNT(c.card_id) AS n,
                  SUM(CASE WHEN TRIM(COALESCE(c.read_text,'')) != ''
                            OR TRIM(COALESCE(c.cite_raw,'')) != ''
                            OR c.disclosed_only = 1
                       THEN 1 ELSE 0 END) AS filled
           FROM sources s JOIN cards c ON c.source_id = s.source_id
           GROUP BY s.source_id""").fetchall()
    for r in rows:
        n, filled = r["n"] or 0, r["filled"] or 0
        if n >= min_cards and (n - filled) / n >= threshold:
            out[r["source_id"]] = {"title": r["title"], "cards": n,
                                   "filled": filled}
    return out


def check_template_sources(store: Store) -> list[Finding]:
    tmpl = template_sources(store)
    if not tmpl:
        return []
    names = ", ".join(repr(v["title"]) for v in tmpl.values())
    total = sum(v["cards"] for v in tmpl.values())
    return [Finding(
        kind="no_read_text", severity=Severity.INFO,
        title=f"{len(tmpl)} source(s) are unfilled outlines, not evidence",
        detail=(f"{names} — {total} tags with no cites and nothing underlined. "
                f"Excluded from position analysis so they do not drown the "
                f"real findings."),
        fix="Paste your cards under these tags and re-ingest; they will then be "
            "analyzed like any other file.",
        analyzer="deterministic:template_detection",
        evidence={"sources": {k: v for k, v in tmpl.items()}},
    )]


def check_answer_coverage(store: Store) -> list[Finding]:
    """Positions you run that the corpus answers, where you have no comeback.

    The graph already knows that "AT: Ratepayer Harm" answers "Ratepayer Harm".
    So: for each position with cards, collect the AT-blocks aimed at it. Then
    ask whether anything in the corpus answers *those* back. If not, the
    opponent's answer is the last word.
    """
    rows = [dict(r) for r in store.conn.execute("SELECT * FROM nodes")]
    by_id = {n["node_id"]: n for n in rows}
    incoming: dict[str, list[dict]] = defaultdict(list)
    for e in store.conn.execute("SELECT * FROM edges WHERE kind='answers'"):
        incoming[e["dst_node_id"]].append(dict(e))

    # which nodes answer something (so we can see if an AT block is itself answered)
    answered_targets = {e["dst_node_id"] for e in
                        store.conn.execute("SELECT dst_node_id FROM edges WHERE kind='answers'")}

    out: list[Finding] = []
    for node in rows:
        if node["card_count"] < 1:
            continue
        if answers_target(node["title"]):
            continue  # this is itself an answer block
        attackers = incoming.get(node["node_id"], [])
        if not attackers:
            continue
        undefended = [a for a in attackers if a["src_node_id"] not in answered_targets]
        if not undefended:
            continue
        # Dedup by title: the same answer block filed in three ingested files is
        # one argument you have to beat, not three.
        seen: set[str] = set()
        names = []
        for a in undefended:
            t = by_id.get(a["src_node_id"], {}).get("title")
            if t and t.lower() not in seen:
                seen.add(t.lower())
                names.append(t)
        if not names:
            continue
        sev = Severity.MAJOR if len(names) >= 2 else Severity.MINOR
        out.append(Finding(
            kind="answer_gap", severity=sev,
            title=f"{node['title']}: {len(names)} answer block(s) with no response",
            detail="These blocks in the index attack this position and nothing "
                   "in your files answers them back: " + ", ".join(repr(n) for n in names),
            fix=f"Write a frontline extending {node['title']} through "
                f"{names[0] if names else 'the answer'}.",
            node_ids=[node["node_id"]] + [a["src_node_id"] for a in undefended],
            analyzer="deterministic:answer_coverage",
            evidence={"undefended_answers": names, "count": len(names)},
        ))
    return out


def check_unanswered_positions(store: Store) -> list[Finding]:
    """Positions with real card support that nothing in the index answers.

    Not automatically a defect — it may just mean you have not ingested the
    other side yet — so this is INFO unless the corpus is otherwise well
    connected, and the detail says which.
    """
    rows = [dict(r) for r in store.conn.execute(
        "SELECT * FROM nodes WHERE card_count >= 2")]
    answered = {e["dst_node_id"] for e in
                store.conn.execute("SELECT dst_node_id FROM edges WHERE kind='answers'")}
    total_at = store.conn.execute(
        "SELECT COUNT(*) FROM edges WHERE kind='answers'").fetchone()[0]
    bare = [n for n in rows
            if n["node_id"] not in answered and not answers_target(n["title"])]
    if not bare or total_at == 0:
        return []
    return [Finding(
        kind="unanswered_position", severity=Severity.INFO,
        title=f"{len(bare)} position(s) have no answer blocks anywhere in the index",
        detail="Nothing in the ingested corpus argues against these. That is "
               "either a genuine blind spot or simply a corpus you have not "
               "loaded the other side of.",
        fix="Ingest opposing files, or write the answers yourself and see "
            "whether the position survives them.",
        node_ids=[n["node_id"] for n in bare],
        analyzer="deterministic:unanswered",
        evidence={"positions": [n["title"] for n in bare][:12],
                  "count": len(bare)},
    )]


def _owner_of(row: dict) -> str:
    """Whose files a card belongs to.

    For a single team's folder this is constant and irrelevant. For an ingested
    archive it is essential: the published caselist is ~800 *different* teams,
    and any check that reasons about "your own files" has to know where one
    team's files end and another's begin. The team is the directory the source
    lives in, which is how both the archive and every real file layout are
    organized.
    """
    path = (row.get("source_path") or "").replace("\\", "/")
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 2:
        return parts[-2]
    return row.get("source_id") or ""


def check_self_contradiction(store: Store) -> list[Finding]:
    """The same source carrying opposite sides *within one team's files*.

    A real round-losing failure: if your aff cites Ember for "renewables can't
    firm the load" and your neg cites Ember for "large buyers drive clean
    procurement", a good opponent reads both of your cards back at you.

    Scoped by owner, and that scoping is the whole correctness of the check.
    Run unscoped over the published archive it reports that Bostrom, Baudrillard
    and forty others are "cited on both sides" -- which is true of the community
    and meaningless as advice, because the aff card is one school's and the neg
    card is another's. Nobody contradicted themselves.
    """
    rows = [dict(r) for r in store.conn.execute(
        "SELECT card_id, cite_author, cite_year, side, tag, source_id, "
        "source_path FROM cards WHERE cite_author IS NOT NULL "
        "AND cite_author != ''")]
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        key = author_key(r["cite_author"])
        if not key:
            continue
        by_key[(_owner_of(r), key)].append(r)

    out: list[Finding] = []
    for (owner, author), group in by_key.items():
        sides = {g["side"] for g in group} - {"unknown", None}
        if not {"aff", "neg"} <= sides:
            continue
        aff = [g for g in group if g["side"] == "aff"]
        neg = [g for g in group if g["side"] == "neg"]
        out.append(Finding(
            kind="self_contradiction", severity=Severity.MAJOR,
            title=(f"{group[0]['cite_author']} is cited on both sides"
                   + (f" in {owner}" if owner else "")),
            detail=(f"You read {group[0]['cite_author']} for the aff "
                    f"({aff[0]['tag'][:60]}…) and for the neg "
                    f"({neg[0]['tag'][:60]}…). If both files see a round, the "
                    f"opponent reads your own author against you."),
            fix="Decide which reading of this source you are committed to, or "
                "be ready to explain why the two claims are compatible.",
            card_ids=[g["card_id"] for g in (aff[:3] + neg[:3])],
            analyzer="deterministic:self_contradiction",
            evidence={"author": group[0]["cite_author"], "owner": owner,
                      "aff_cards": len(aff), "neg_cards": len(neg)},
        ))
    return out


def check_duplicate_bloat(store: Store) -> list[Finding]:
    rows = [dict(r) for r in store.conn.execute(
        "SELECT * FROM edges WHERE kind='duplicates'")]
    if len(rows) < 3:
        return []
    retagged = 0
    for r in rows:
        try:
            if json.loads(r["evidence"] or "{}").get("retagged"):
                retagged += 1
        except Exception:
            pass
    if retagged < 2:
        return []
    return [Finding(
        kind="duplicate_bloat", severity=Severity.MINOR,
        title=f"{retagged} card pair(s) are the same evidence under different tags",
        detail="The same underlying cut appears more than once with different "
               "taglines. In a file that is clutter; in a round it is an "
               "opportunity for the opponent to point out that your three "
               "cards are one card.",
        fix="Consolidate to the best-tagged version and delete the rest.",
        card_ids=[r["src_node_id"] for r in rows[:10]],
        analyzer="deterministic:duplicates",
        evidence={"retagged_pairs": retagged, "total_duplicate_pairs": len(rows)},
    )]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def corpus_newest_year(store: Store) -> int | None:
    row = store.conn.execute(
        "SELECT MAX(cite_year) FROM cards WHERE cite_year IS NOT NULL"
    ).fetchone()
    return row[0] if row and row[0] else None


def analyze_position(title: str, cards: list[dict],
                     corpus_newest: int | None) -> list[Finding]:
    findings: list[Finding] = []
    findings += check_single_card(title, cards)
    findings += check_source_concentration(title, cards)
    findings += check_qualifications(title, cards)
    findings += check_power_tagging(cards)
    findings += check_read_text_health(cards)
    findings += check_recency(title, cards, corpus_newest)
    return findings


def analyze_corpus(store: Store) -> list[Finding]:
    findings: list[Finding] = []
    findings += check_template_sources(store)
    findings += check_answer_coverage(store)
    findings += check_unanswered_positions(store)
    findings += check_self_contradiction(store)
    findings += check_duplicate_bloat(store)

    # The same position title can appear in several ingested files. Collapse
    # identical findings so the report reads as arguments, not as rows.
    seen: set[tuple[str, str]] = set()
    deduped: list[Finding] = []
    for f in findings:
        key = (f.kind, f.title)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(f)
    return deduped


def score_position(cards: list[dict], findings: list[Finding]) -> tuple[float, float]:
    """(support_score, exposure_score), both in [0,1].

    Support rewards card count with diminishing returns, source diversity, and
    read-text health. Exposure is severity-weighted findings, normalized so one
    critical finding dominates a pile of minor ones — which matches how rounds
    actually go.
    """
    if not cards:
        return 0.0, 1.0
    n = len(cards)
    breadth = min(n / 5.0, 1.0)
    srcs = len(_distinct_sources(cards))
    diversity = min(srcs / max(min(n, 4), 1), 1.0)
    read_ok = sum(1 for c in cards if (c.get("read_text") or "").strip()) / n
    ratios = [c.get("read_ratio") or 0 for c in cards if c.get("read_ratio")]
    depth = 1.0
    if ratios:
        med = statistics.median(ratios)
        depth = 1.0 if 0.15 <= med <= 0.9 else 0.6
    support = 0.35 * breadth + 0.3 * diversity + 0.25 * read_ok + 0.10 * depth

    weight = {Severity.CRITICAL: 1.0, Severity.MAJOR: 0.45,
              Severity.MINOR: 0.15, Severity.INFO: 0.0}
    raw = sum(weight[f.severity] * f.confidence for f in findings)
    exposure = 1 - 1 / (1 + raw)  # saturating, so it never exceeds 1
    return round(min(support, 1.0), 3), round(min(exposure, 1.0), 3)
