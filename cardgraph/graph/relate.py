"""Build the argument graph.

The outline already tells us a lot; the graph makes it queryable across files.
Two edge types, built by different means:

  answers    "AT: Ratepayer Harm" in one file points at "Ratepayer Harm"
             wherever it appears in the corpus. Derived from block naming
             conventions -- see models.answers_target. This is the edge that
             makes the app useful: it is how "show me every answer to my
             ratepayer contention" becomes a query rather than a memory test.

  duplicates The same card recut by different teams. Derived from cite+read
             overlap, not tags, because the tag is the thing that changes.

Matching block titles is fuzzy on purpose. "AT: Ratepayer Harm", "AT Ratepayer
DA", and "A/T: Rate Payer" all mean the same thing to a human and none of them
string-match. We normalize hard, then require a token-overlap threshold, and we
record a confidence so the UI can show weak links differently rather than
pretending the graph is clean. It is not clean. Nothing built on volunteer-
authored headings ever is.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

from ..index.store import Store
from ..models import answers_target, strip_speech_prefix

_STOP = {
    "the", "a", "an", "of", "and", "to", "for", "on", "in", "at", "da", "cp",
    "adv", "advantage", "block", "answers", "ans", "frontline", "ext",
    "extension", "1nc", "2nc", "1nr", "2nr", "1ac", "2ac", "1ar", "2ar",
}


def normalize_title(title: str) -> str:
    """Strip speech prefix and answer prefix, lowercase, drop punctuation.

    Stripping the answer prefix matters: without it "AT: Ratepayer Harm" and
    "Ratepayer Harm" never hit the exact-match shortcut in `similarity` and are
    left to Jaccard, which is fine for two-word titles and gets fragile for
    long ones ("AT: Cost Allocation Reform Solves Ratepayer Impacts").
    """
    t = strip_speech_prefix(title or "")
    t = answers_target(t) or t
    t = re.sub(r"[^\w\s]", " ", t.lower())
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokens(title: str) -> frozenset[str]:
    return frozenset(w for w in normalize_title(title).split() if w not in _STOP and len(w) > 2)


def similarity(a: str, b: str) -> float:
    """Jaccard over content tokens, with an exact-normalized-match shortcut."""
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    if not inter:
        return 0.0
    return inter / len(ta | tb)


def build_answer_edges(store: Store, threshold: float = 0.45) -> int:
    """Link every 'AT: X' node to nodes plausibly named X."""
    rows = store.conn.execute(
        "SELECT node_id, title, source_id, path_json FROM nodes"
    ).fetchall()
    nodes = [dict(r) for r in rows]

    # index candidate targets by token for a cheap first pass
    by_token: dict[str, list[dict]] = defaultdict(list)
    for n in nodes:
        if answers_target(n["title"]):
            continue  # an AT block is not itself a target
        for tok in tokens(n["title"]):
            by_token[tok].append(n)

    made = 0
    for n in nodes:
        target = answers_target(n["title"])
        if not target:
            continue
        cand_ids: dict[str, dict] = {}
        for tok in tokens(target):
            for c in by_token.get(tok, []):
                cand_ids[c["node_id"]] = c
        best: list[tuple[float, dict]] = []
        for c in cand_ids.values():
            if c["node_id"] == n["node_id"]:
                continue
            s = similarity(target, c["title"])
            if s >= threshold:
                best.append((s, c))
        best.sort(key=lambda x: -x[0])
        for s, c in best[:8]:
            store.add_edge(n["node_id"], c["node_id"], "answers", round(s, 3),
                           evidence=f"{n['title']!r} -> {c['title']!r}")
            made += 1
    return made


def build_duplicate_edges(store: Store, min_overlap: float = 0.8) -> int:
    """Group cards that are the same underlying evidence.

    Keyed on (author, year) then compared on read_text shingles. Deliberately
    not keyed on the tag: re-tagging is exactly the thing we want to see.
    """
    rows = store.conn.execute(
        "SELECT card_id, cite_author, cite_year, read_text, tag, source_id "
        "FROM cards WHERE read_text != ''"
    ).fetchall()

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = ((r["cite_author"] or "").lower(), r["cite_year"])
        if not key[0]:
            continue
        buckets[key].append(dict(r))

    def shingles(text: str, n: int = 5) -> frozenset[str]:
        words = normalize_title(text).split()
        if len(words) < n:
            return frozenset([" ".join(words)]) if words else frozenset()
        return frozenset(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))

    made = 0
    with store.tx() as c:
        for group in buckets.values():
            if len(group) < 2:
                continue
            sh = {g["card_id"]: shingles(g["read_text"]) for g in group}
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    sa, sb = sh[a["card_id"]], sh[b["card_id"]]
                    if not sa or not sb:
                        continue
                    ov = len(sa & sb) / min(len(sa), len(sb))
                    if ov < min_overlap:
                        continue
                    retag = a["tag"].strip().lower() != b["tag"].strip().lower()
                    c.execute(
                        """INSERT OR REPLACE INTO edges
                           (src_node_id, dst_node_id, kind, confidence, evidence)
                           VALUES (?,?,?,?,?)""",
                        (a["card_id"], b["card_id"], "duplicates", round(ov, 3),
                         json.dumps({"retagged": retag,
                                     "tag_a": a["tag"], "tag_b": b["tag"]})),
                    )
                    made += 1
    return made


def flag_warrants(store: Store) -> int:
    """Cheap, honest heuristics for evidence quality. These are *flags for a
    human*, not verdicts -- every one of them has false positives, and the UI
    labels them as prompts to go read the card.

    thin_read      very little of the body is actually read
    over_read      nearly the whole body is underlined (often a copy-paste
                   artifact rather than a genuinely dense card)
    no_read        parsed but nothing marked -- usually a parser miss
    hedged_body    body hedges ("may", "could", "suggests") in a way the tag does not
    undated        no year we could extract
    """
    hedge = re.compile(
        r"\b(may|might|could|suggests?|appears?|possibly|potentially|"
        r"is likely to|we estimate|projected)\b", re.IGNORECASE)
    strong_tag = re.compile(
        r"\b(collapse[sd]?|causes?|guarantees?|ensures?|proves?|destroys?|"
        r"extinction|inevitable|will\b)", re.IGNORECASE)

    rows = store.conn.execute(
        "SELECT card_id, tag, body, read_text, read_ratio, cite_year FROM cards"
    ).fetchall()
    n = 0
    with store.tx() as c:
        for r in rows:
            flags = []
            rr = r["read_ratio"] or 0.0
            if not (r["read_text"] or "").strip():
                flags.append("no_read")
            elif rr < 0.12:
                flags.append("thin_read")
            elif rr > 0.92:
                flags.append("over_read")
            if r["cite_year"] is None:
                flags.append("undated")
            body, tag = r["body"] or "", r["tag"] or ""
            if hedge.search(body) and strong_tag.search(tag) and not hedge.search(tag):
                flags.append("hedged_body")
            if flags:
                c.execute("UPDATE cards SET warrant_flags=? WHERE card_id=?",
                          (json.dumps(flags), r["card_id"]))
                n += 1
    return n


def build_all(store: Store) -> dict[str, int]:
    return {
        "answer_edges": build_answer_edges(store),
        "duplicate_edges": build_duplicate_edges(store),
        "flagged_cards": flag_warrants(store),
    }
