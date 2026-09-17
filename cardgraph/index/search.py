"""Hybrid retrieval: BM25 over FTS5 + TF-IDF cosine, fused with RRF.

Why both. Debate queries come in two flavors and they want different things:

  "Hallberg 30"                -> a lexical lookup. Exact-match wins; embeddings
                                  are actively harmful here because every card
                                  in the file is topically similar.
  "cards saying households
   subsidize industrial load"  -> a semantic query where the debater does not
                                  know the vocabulary the card uses.

Running one retriever means being bad at half the traffic. Reciprocal Rank
Fusion combines the rankings without needing the two scores to be on a
comparable scale, which they are not.

The vector side defaults to TF-IDF + SVD (scikit-learn) rather than a
transformer. That is a deliberate default, not laziness: it installs in seconds,
runs on CPU, and on a single-topic corpus -- which every debate corpus is --
lexical-semantic overlap is high enough that the quality gap is small. Set
`CARDGRAPH_EMBEDDINGS=sentence-transformers` when you want the better model and
are willing to carry a 2GB dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from dataclasses import dataclass, replace

import numpy as np

from .store import Store

RRF_K = 60


@dataclass
class Hit:
    card_id: str
    score: float
    tag: str
    read_text: str
    cite_raw: str
    block: str | None
    side: str
    source_id: str
    cite_author: str | None = None
    cite_year: int | None = None
    source_title: str = ""
    source_origin: str = ""
    source_url: str | None = None
    lexical_rank: int | None = None
    vector_rank: int | None = None
    matched_queries: list[str] | None = None
    match_reasons: list[str] | None = None
    confidence: str = "exploratory"
    match_type: str = "semantic"
    read_ratio: float = 0.0
    disclosed_only: bool = False
    warrant_flags: list[str] | None = None
    evidence_status: str = "needs-review"

    def to_dict(self) -> dict:
        return {
            "card_id": self.card_id, "score": round(self.score, 5),
            "tag": self.tag, "read_text": self.read_text,
            "cite_raw": self.cite_raw, "cite_author": self.cite_author,
            "cite_year": self.cite_year, "block": self.block, "side": self.side,
            "source_id": self.source_id,
            "lexical_rank": self.lexical_rank, "vector_rank": self.vector_rank,
            "source_title": self.source_title, "source_origin": self.source_origin,
            "source_url": self.source_url,
            "matched_queries": self.matched_queries or [],
            "match_reasons": self.match_reasons or [],
            "confidence": self.confidence,
            "match_type": self.match_type,
            "read_ratio": round(self.read_ratio, 4),
            "disclosed_only": self.disclosed_only,
            "warrant_flags": self.warrant_flags or [],
            "evidence_status": self.evidence_status,
        }


def _fts_escape(q: str) -> str:
    """FTS5 MATCH syntax is not user input. Quote every bare term, keep the
    handful of operators debaters actually type."""
    tokens = re.findall(r'"[^"]*"|\S+', q)
    out = []
    for t in tokens:
        if t.upper() in ("AND", "OR", "NOT"):
            out.append(t.upper())
        elif t.startswith('"'):
            out.append(t)
        else:
            cleaned = re.sub(r'["*()]', " ", t).strip()
            if cleaned:
                out.append(f'"{cleaned}"')
    return " ".join(out) or '""'


class VectorIndex:
    """TF-IDF -> SVD dense vectors over read_text (falling back to tag+body
    when a card has no read text, so unparsed files still retrieve).

    Persisted to disk, because fitting is the expensive part and it does not
    change between queries. Over the 166,816-card archive a cold fit takes about
    four minutes; without a cache that cost is paid by every `search`
    invocation and every `serve` startup, which makes the tool unusable at the
    scale it was built for. Loading the cached fit takes a couple of seconds.

    The cache is keyed on a fingerprint of the corpus, so it invalidates itself
    when cards are added rather than silently serving a stale index -- which
    would be the worse failure: search that quietly cannot see your newest
    evidence.
    """

    FORMAT = 2   # bump to invalidate every cache after a fitting change

    def __init__(self, dims: int = 256):
        self.dims = dims
        self.card_ids: list[str] = []
        self.matrix: np.ndarray | None = None
        self._vec = None
        self._svd = None

    def _text_for(self, row: dict) -> str:
        read = (row.get("read_text") or "").strip()
        base = read if len(read) > 40 else (row.get("body") or "")
        return f"{row.get('tag','')} \n {base}"

    def build(self, rows: list[dict]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        if not rows:
            self.card_ids, self.matrix = [], None
            return
        self.card_ids = [r["card_id"] for r in rows]
        corpus = [self._text_for(r) for r in rows]

        # max_df prunes terms that appear in almost every document, which is
        # what stops a single-topic corpus being dominated by its own vocabulary.
        # On a *small* or homogeneous corpus it prunes everything and sklearn
        # raises -- so a debater indexing their own six-card file, which is the
        # most likely first run this tool ever sees, gets a crash instead of a
        # search index. Retry without the filter rather than fail.
        def _fit(stop_words="english", **kw):
            vec = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 2),
                                  sublinear_tf=True, max_features=120_000, **kw)
            return vec, vec.fit_transform(corpus)

        try:
            self._vec, X = _fit(min_df=1, max_df=0.85)
        except ValueError:
            try:
                self._vec, X = _fit(min_df=1, max_df=1.0)
            except ValueError:
                # Every token is a stop word (or the corpus is near-empty).
                self._vec, X = _fit(min_df=1, max_df=1.0, stop_words=None)

        # SVD requires n_components < min(samples, features). The previous
        # lower bound of two crashed on a tiny corpus whose cards shared only
        # one vocabulary term; dense TF-IDF is the correct fallback there.
        n_comp = min(self.dims, max(1, min(X.shape) - 1))
        if X.shape[0] > 2 and n_comp >= 2:
            self._svd = TruncatedSVD(n_components=n_comp, random_state=0)
            dense = self._svd.fit_transform(X)
        else:
            self._svd = None
            dense = X.toarray()
        # float32 halves the cache and the resident matrix (341MB -> 171MB at
        # 166k x 256) with no measurable effect on cosine ranking.
        self.matrix = normalize(dense).astype(np.float32, copy=False)

    # -- persistence -------------------------------------------------------

    @staticmethod
    def fingerprint(card_ids) -> str:
        """Identify this corpus cheaply but safely.

        Count alone is not enough: deleting one card and adding another leaves
        it unchanged. Hashing every id is exact and costs well under a second
        at 166k cards, which is nothing next to a four-minute refit.

        Takes ids, not rows, on purpose -- see Store.card_ids.
        """
        h = hashlib.sha1()
        ids = list(card_ids)
        h.update(str(len(ids)).encode())
        for cid in sorted(ids):
            h.update(cid.encode())
        return h.hexdigest()[:16]

    def save(self, path: str, fingerprint: str,
             revision: tuple[int, int] | None = None) -> None:
        if self.matrix is None or self._vec is None:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump({
                "format": self.FORMAT, "fingerprint": fingerprint,
                "revision": revision,
                "dims": self.dims, "card_ids": self.card_ids,
                "vec": self._vec, "svd": self._svd, "matrix": self.matrix,
            }, fh, protocol=pickle.HIGHEST_PROTOCOL)
        # atomic: a half-written cache must never be loadable
        os.replace(tmp, path)

    def load(self, path: str, fingerprint: str,
             revision: tuple[int, int] | None = None) -> bool:
        if not path or not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as fh:
                blob = pickle.load(fh)
        except Exception:
            return False
        if not isinstance(blob, dict):
            return False
        if blob.get("format") != self.FORMAT:
            return False
        if blob.get("fingerprint") != fingerprint:
            return False    # corpus changed; refit rather than serve a stale index
        # `revision=None` preserves the small public load() API for callers
        # inspecting legacy caches. SearchEngine supplies it so a same-id
        # replacement (for example an edited card with identical content hash)
        # cannot reuse a stale fitted matrix.
        if revision is not None and blob.get("revision") != revision:
            return False
        self.dims = blob["dims"]
        self.card_ids = blob["card_ids"]
        self._vec = blob["vec"]
        self._svd = blob["svd"]
        self.matrix = blob["matrix"]
        return True

    def query(self, text: str, k: int = 50,
              allowed: set[str] | None = None,
              include_nonpositive: bool = False,
              min_similarity: float = 0.0) -> list[tuple[str, float]]:
        if k <= 0 or self.matrix is None or self._vec is None or not self.card_ids:
            return []
        from sklearn.preprocessing import normalize

        q = self._vec.transform([text])
        qd = self._svd.transform(q) if self._svd is not None else q.toarray()
        qd = normalize(qd).astype(self.matrix.dtype, copy=False)
        sims = (self.matrix @ qd.T).ravel()
        if allowed is not None:
            allowed_mask = np.fromiter(
                (cid in allowed for cid in self.card_ids), dtype=bool,
                count=len(self.card_ids),
            )
            sims[~allowed_mask] = -np.inf
        order = np.argsort(-sims)[:k]
        # Do not return zero/negative cosine entries. A zero query vector
        # (unknown words such as a typo) otherwise produces every card in
        # arbitrary corpus order, making search look confident when it found
        # nothing. Negative cosine values are also anti-matches, not results.
        return [(self.card_ids[i], float(sims[i])) for i in order
                if np.isfinite(sims[i]) and
                (include_nonpositive or sims[i] > min_similarity)]


def _like_pattern(value: str) -> str:
    """Build a literal substring pattern for SQLite LIKE."""
    return "%" + value.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


_SMART_PREFIX = re.compile(
    r"^(?:please\s+)?(?:find|show|search(?:\s+for)?|give me|i need|locate)\s+",
    re.IGNORECASE,
)
_SMART_FILLER = re.compile(
    r"\b(?:cards?|evidence|arguments?|that|which|saying|about|on|related to|"
    r"according to|for my case|in debate)\b",
    re.IGNORECASE,
)
_SMART_SIDE = re.compile(
    r"\b(?:on\s+the\s+)?(?P<side>affirmative|aff|negative|neg)\s+(?:side|cards?|case)\b",
    re.IGNORECASE,
)
_SMART_YEAR_RANGE = re.compile(
    r"\b(?:between\s+)?(?P<first>19\d{2}|20\d{2}|21\d{2})\s+"
    r"(?:and|to|through|-)\s+(?P<second>19\d{2}|20\d{2}|21\d{2})\b",
    re.IGNORECASE,
)
_SMART_YEAR_FROM = re.compile(
    r"\b(?:from|after|since)\s+(?P<year>19\d{2}|20\d{2}|21\d{2})\b",
    re.IGNORECASE,
)
_SMART_YEAR_TO = re.compile(
    r"\b(?:before|until|through|up\s+to)\s+(?P<year>19\d{2}|20\d{2}|21\d{2})\b",
    re.IGNORECASE,
)
_SMART_YEAR_IN = re.compile(
    r"\bin\s+(?P<year>19\d{2}|20\d{2}|21\d{2})\b",
    re.IGNORECASE,
)
_SMART_SYNONYMS = {
    "ban": ("ban", "prohibition", "moratorium"),
    "bans": ("ban", "prohibition", "moratorium"),
    "cost": ("cost", "rate", "burden"),
    "costs": ("cost", "rates", "burden"),
    "emissions": ("emissions", "pollution", "carbon"),
    "household": ("household", "ratepayer", "residential"),
    "households": ("households", "ratepayers", "residential"),
    "nuclear": ("nuclear", "reactor", "atomic"),
    "renewable": ("renewable", "clean energy", "wind solar"),
}


def parse_smart_request(query: str) -> tuple[str, dict[str, str | int]]:
    """Extract only unambiguous debate filters from a natural-language request.

    This intentionally recognizes a small grammar rather than guessing from
    arbitrary prose. For example, ``negative cards from 2024 about nuclear``
    becomes ``nuclear`` with ``side=neg`` and ``year_min=year_max=2024``.
    Unrecognized wording remains searchable as text.
    """
    text = " ".join(query.split()).strip()
    filters: dict[str, str | int] = {}
    match = _SMART_SIDE.search(text)
    if match:
        filters["side"] = "aff" if match.group("side").lower() in {"aff", "affirmative"} else "neg"
        text = text[:match.start()] + " " + text[match.end():]

    match = _SMART_YEAR_RANGE.search(text)
    if match:
        first, second = int(match.group("first")), int(match.group("second"))
        filters["year_min"], filters["year_max"] = min(first, second), max(first, second)
        text = text[:match.start()] + " " + text[match.end():]
    else:
        match = _SMART_YEAR_FROM.search(text)
        if match:
            filters["year_min"] = int(match.group("year"))
            text = text[:match.start()] + " " + text[match.end():]
        match = _SMART_YEAR_TO.search(text)
        if match:
            filters["year_max"] = int(match.group("year"))
            text = text[:match.start()] + " " + text[match.end():]
        match = _SMART_YEAR_IN.search(text)
        if match:
            year = int(match.group("year"))
            filters["year_min"] = filters["year_max"] = year
            text = text[:match.start()] + " " + text[match.end():]
    text = _SMART_PREFIX.sub("", text)
    text = _SMART_FILLER.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip(" .,?!;:"), filters


def smart_query_variants(query: str, *, limit: int = 5) -> list[str]:
    """Turn a conversational request into a few focused retrieval queries.

    This is deliberately local and explainable. It strips request framing and
    produces one synonym-focused variant at a time instead of stuffing every
    synonym into one query, which would make lexical search overly broad. An
    optional model layer in the API can add more variants, but it never writes
    or summarizes evidence.
    """
    original = " ".join(query.split()).strip()
    if not original:
        return []
    cleaned = _SMART_PREFIX.sub("", original)
    cleaned = _SMART_FILLER.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,?!;:")
    variants: list[str] = []
    for candidate in (original, cleaned):
        if candidate and candidate.lower() not in {v.lower() for v in variants}:
            variants.append(candidate)
    words = re.findall(r"[\w'-]+", cleaned.lower())
    for word in words:
        for synonym in _SMART_SYNONYMS.get(word, ()):
            if synonym != word:
                candidate = re.sub(rf"\b{re.escape(word)}\b", synonym, cleaned,
                                   count=1, flags=re.IGNORECASE)
                if candidate.lower() not in {v.lower() for v in variants}:
                    variants.append(candidate)
                break
        if len(variants) >= limit:
            break
    return variants[:limit]


class SearchEngine:
    def __init__(self, store: Store, cache_path: str | None = None):
        self.store = store
        self.vectors = VectorIndex()
        self._built = False
        self._revision: tuple[int, int] | None = None
        if cache_path is None and getattr(store, "path", None):
            cache_path = os.path.splitext(store.path)[0] + ".index"
        self.cache_path = cache_path

    def build(self, use_cache: bool = True, progress=None) -> None:
        say = progress or (lambda *_: None)
        # Fingerprint from ids alone, so a cache hit never pays to load the
        # whole corpus into memory.
        revision = self.store.card_revision()
        fp = VectorIndex.fingerprint(self.store.card_ids())
        if use_cache and self.cache_path and self.vectors.load(
                self.cache_path, fp, revision=revision):
            say(f"loaded cached index ({len(self.vectors.card_ids)} cards)")
            self._built = True
            self._fingerprint = fp
            self._revision = self.store.card_revision()
            return
        rows = self.store.all_cards()
        say(f"fitting index over {len(rows)} cards (cached after this)")
        self.vectors.build(rows)
        if use_cache and self.cache_path:
            try:
                self.vectors.save(self.cache_path, fp, revision=revision)
            except Exception:
                pass    # a cache we cannot write is not a reason to fail a search
        self._built = True
        self._fingerprint = fp
        self._revision = self.store.card_revision()

    def _ensure_current(self) -> None:
        """Refresh after another command adds cards to this live store."""
        current = self.store.card_revision()
        if not self._built or current != self._revision:
            self.build()

    def _lexical(self, query: str, k: int,
                 allowed: set[str] | None = None) -> list[tuple[str, float]]:
        params: list[object] = [_fts_escape(query)]
        allowed_clause = ""
        if allowed is not None:
            if not allowed:
                return []
            # SQLite builds commonly cap bound parameters at 999. A broad
            # source filter can contain thousands of cards, so rank each safe
            # chunk and merge the top results instead of failing with a runtime
            # "too many SQL variables" error.
            if len(allowed) > 900:
                ranked: list[tuple[str, float]] = []
                ids = sorted(allowed)
                for start in range(0, len(ids), 800):
                    ranked.extend(self._lexical(query, k, set(ids[start:start + 800])))
                return sorted(ranked, key=lambda item: -item[1])[:k]
            placeholders = ",".join("?" * len(allowed))
            allowed_clause = f" AND m.card_id IN ({placeholders})"
            params.extend(sorted(allowed))
        params.append(k)
        try:
            rows = self.store.conn.execute(
                """SELECT m.card_id AS card_id, bm25(cards_fts, 8.0, 4.0, 2.0, 1.0) AS score
                   FROM cards_fts
                   JOIN fts_map m ON m.rowid = cards_fts.rowid
                   WHERE cards_fts MATCH ?""" + allowed_clause +
                " ORDER BY score LIMIT ?", params,
            ).fetchall()
        except Exception:
            return []
        return [(r["card_id"], -float(r["score"])) for r in rows]

    def _candidate_ids(self, *, side: str | None = None,
                       author: str | None = None, year_min: int | None = None,
                       year_max: int | None = None, block: str | None = None,
                       min_read_ratio: float | None = None,
                       source: str | None = None) -> set[str] | None:
        """Return metadata-matching ids before ranking.

        Applying filters after a top-60 pool is incorrect: a narrow author or
        season filter can remove every early hit while matching cards remain
        below the pool. Candidate restriction is shared by BM25 and vectors so
        both rankers see the same universe and the requested k is meaningful.
        """
        where: list[str] = []
        params: list[object] = []
        if side and side != "both":
            where.append("c.side = ?")
            params.append(side)
        if author:
            where.append("lower(COALESCE(c.cite_author, '')) = ?")
            params.append(author.lower())
        if year_min is not None:
            where.append("COALESCE(c.cite_year, 0) >= ?")
            params.append(year_min)
        if year_max is not None:
            where.append("COALESCE(c.cite_year, 9999) <= ?")
            params.append(year_max)
        if block:
            where.append("lower(COALESCE(c.block, '')) LIKE ? ESCAPE '\\'")
            params.append(_like_pattern(block))
        if min_read_ratio is not None:
            where.append("COALESCE(c.read_ratio, 0) >= ?")
            params.append(min_read_ratio)
        if source:
            where.append(
                "lower(COALESCE(c.source_id, '') || ' ' || "
                "COALESCE(s.title, '') || ' ' || COALESCE(s.origin, '') || ' ' || "
                "COALESCE(c.source_path, '')) LIKE ? ESCAPE '\\'"
            )
            params.append(_like_pattern(source))
        if not where:
            return None
        rows = self.store.conn.execute(
            "SELECT c.card_id FROM cards c LEFT JOIN sources s ON s.source_id = c.source_id "
            "WHERE " + " AND ".join(where), params).fetchall()
        return {r["card_id"] for r in rows}

    @staticmethod
    def _matched_terms(query: str, row: dict) -> list[str]:
        haystack = " ".join(str(row.get(key) or "") for key in
                            ("tag", "read_text", "cite_raw")).lower()
        terms = []
        for term in re.findall(r"[\w'-]+", query.lower()):
            if len(term) >= 3 and term not in terms and re.search(
                    rf"(?<!\w){re.escape(term)}(?!\w)", haystack):
                terms.append(term)
        return terms

    @classmethod
    def _match_quality(cls, query: str, row: dict,
                       lexical_rank: int | None,
                       vector_rank: int | None) -> tuple[str, str]:
        """Return a human-readable signal class, not a fake probability.

        ``exact`` means query terms occur in the card and lexical retrieval
        found it; ``hybrid`` means lexical and semantic retrieval agree; and
        ``semantic`` is discovery-only. The confidence labels describe ranking
        evidence (high/medium/exploratory), never factual correctness.
        """
        exact = bool(cls._matched_terms(query, row))
        if exact and lexical_rank is not None:
            match_type = "exact"
            confidence = "high" if lexical_rank <= 5 else "medium"
        elif lexical_rank is not None and vector_rank is not None:
            match_type, confidence = "hybrid", "medium"
        elif lexical_rank is not None:
            match_type, confidence = "lexical", "medium"
        else:
            match_type, confidence = "semantic", "exploratory"
        return confidence, match_type

    @staticmethod
    def _evidence_status(row: dict) -> str:
        """Describe traceability, never factual truth or source legality."""
        has_citation = bool((row.get("cite_raw") or "").strip())
        has_source = bool((row.get("source_id") or "").strip())
        has_read_text = bool((row.get("read_text") or "").strip())
        disclosed = bool(row.get("disclosed_only"))
        if has_citation and has_source and (has_read_text or disclosed):
            return "traceable"
        if has_citation and has_source:
            return "attributed"
        return "needs-review"

    @staticmethod
    def _warrant_flags(row: dict) -> list[str]:
        value = row.get("warrant_flags")
        if isinstance(value, list):
            return value
        if not value:
            return []
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (TypeError, json.JSONDecodeError):
            return []

    @classmethod
    def _match_reasons(cls, query: str, row: dict, lexical_rank: int | None,
                       vector_rank: int | None) -> list[str]:
        """Explain the ranking signal without pretending it is a model verdict."""
        terms = cls._matched_terms(query, row)
        reasons: list[str] = []
        if terms:
            reasons.append("exact: " + ", ".join(terms[:6]))
        if lexical_rank is not None:
            reasons.append(f"lexical rank #{lexical_rank}")
        if vector_rank is not None:
            reasons.append(f"semantic rank #{vector_rank}")
        return reasons or ["semantic similarity"]

    def search(self, query: str, k: int = 25, *, side: str | None = None,
               author: str | None = None, year_min: int | None = None,
               year_max: int | None = None, block: str | None = None,
               min_read_ratio: float | None = None,
               source: str | None = None,
               mode: str = "balanced") -> list[Hit]:
        if k <= 0 or not query.strip():
            return []
        self._ensure_current()

        pool = max(k * 4, 60)
        allowed = self._candidate_ids(
            side=side, author=author, year_min=year_min, year_max=year_max,
            block=block, min_read_ratio=min_read_ratio, source=source)
        if allowed == set():
            return []

        if mode not in {"strict", "balanced", "explore"}:
            raise ValueError(f"unknown search mode: {mode}")
        semantic_floor = {"strict": 0.15, "balanced": 0.03, "explore": 0.0}[mode]
        lex = self._lexical(query, pool, allowed=allowed)
        vec = self.vectors.query(
            query, pool, allowed=allowed, min_similarity=semantic_floor,
        )

        lex_rank = {cid: i + 1 for i, (cid, _) in enumerate(lex)}
        vec_rank = {cid: i + 1 for i, (cid, _) in enumerate(vec)}

        fused: dict[str, float] = {}
        for cid, rank in lex_rank.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        for cid, rank in vec_rank.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        if not fused:
            return []

        ids = sorted(fused, key=lambda c: -fused[c])
        placeholders = ",".join("?" * len(ids))
        rows = {r["card_id"]: dict(r) for r in self.store.conn.execute(
            f"SELECT c.*, s.title AS source_title, s.origin AS source_origin, "
            f"s.url AS source_url FROM cards c "
            f"LEFT JOIN sources s ON s.source_id = c.source_id "
            f"WHERE c.card_id IN ({placeholders})", ids)}

        hits: list[Hit] = []
        for cid in ids:
            r = rows.get(cid)
            if not r:
                continue
            if side and side != "both" and r["side"] != side:
                continue
            if author and (r["cite_author"] or "").lower() != author.lower():
                continue
            if year_min is not None and (r["cite_year"] or 0) < year_min:
                continue
            if year_max and (r["cite_year"] or 9999) > year_max:
                continue
            if block and block.lower() not in (r["block"] or "").lower():
                continue
            if min_read_ratio is not None and (r["read_ratio"] or 0) < min_read_ratio:
                continue
            if source and source.lower() not in (
                    (r.get("source_id") or "") + " "
                    + (r.get("source_title") or "") + " "
                    + (r.get("source_origin") or "") + " "
                    + (r.get("source_path") or "")).lower():
                continue
            hits.append(Hit(
                card_id=cid, score=fused[cid], tag=r["tag"],
                read_text=r["read_text"] or "", cite_raw=r["cite_raw"] or "",
                cite_author=r.get("cite_author"), cite_year=r.get("cite_year"),
                block=r["block"], side=r["side"] or "unknown",
                source_id=r["source_id"] or "",
                source_title=r.get("source_title") or "",
                source_origin=r.get("source_origin") or "",
                source_url=r.get("source_url"),
                lexical_rank=lex_rank.get(cid), vector_rank=vec_rank.get(cid),
                match_reasons=self._match_reasons(
                    query, r, lex_rank.get(cid), vec_rank.get(cid),
                ),
                confidence=self._match_quality(
                    query, r, lex_rank.get(cid), vec_rank.get(cid),
                )[0],
                match_type=self._match_quality(
                    query, r, lex_rank.get(cid), vec_rank.get(cid),
                )[1],
                read_ratio=float(r.get("read_ratio") or 0.0),
                disclosed_only=bool(r.get("disclosed_only")),
                warrant_flags=self._warrant_flags(r),
                evidence_status=self._evidence_status(r),
            ))
            if len(hits) >= k:
                break
        return hits

    def smart_search(self, query: str, k: int = 25, *,
                     variants: list[str] | None = None,
                     mode: str = "balanced", **filters) -> tuple[list[Hit], list[str]]:
        """Search a plain-English request through several transparent queries.

        Results are fused by the number and quality of variants that found them,
        so a card that matches both the user's wording and its debate vocabulary
        rises above a one-off synonym hit. The returned variants are shown to the
        caller; this is a search aid, not an opaque AI answer.
        """
        interpreted_query, inferred = parse_smart_request(query)
        for key, value in inferred.items():
            if filters.get(key) is None:
                filters[key] = value
        queries = variants or smart_query_variants(interpreted_query)
        if not queries:
            return [], []
        merged: dict[str, tuple[Hit, float, list[str]]] = {}
        for index, candidate in enumerate(queries):
            for hit in self.search(
                    candidate, k=max(k * 2, 25), mode=mode, **filters):
                # Earlier variants preserve the user's wording; later variants
                # are useful recall expansions but receive slightly less weight.
                weight = 1.0 / (1.0 + index * 0.35)
                previous = merged.get(hit.card_id)
                if previous is None:
                    merged[hit.card_id] = (hit, hit.score * weight, [candidate])
                else:
                    old_hit, score, matched = previous
                    # The first conversational variant may find a card only
                    # semantically, while a later cleaned variant can prove an
                    # exact lexical match. Keep the strongest explanation while
                    # still accumulating every variant's retrieval score.
                    strength = {"exploratory": 0, "medium": 1, "high": 2}
                    best = hit if strength.get(hit.confidence, 0) > strength.get(
                        old_hit.confidence, 0) else old_hit
                    merged[hit.card_id] = (best, score + hit.score * weight,
                                           matched + [candidate])
        ranked = sorted(merged.values(), key=lambda item: -item[1])[:k]
        return [replace(hit, score=score, matched_queries=matched)
                for hit, score, matched in ranked], queries

    def covers(self, query: str, *, exclude_card_ids: set[str] | None = None,
               restrict_to: set[str] | None = None,
               floor: float = 0.35, k: int = 5) -> list[tuple[str, float]]:
        """Does the corpus already contain evidence for `query`?

        This is a different question from "what are the best matches", and
        answering it with `search` is a bug I shipped once and had to come back
        for: ranked retrieval always returns its top k, so an argument nothing
        in your files supports still comes back with four confident-looking
        hits, and the analysis then tells you that you have a block you do not
        have. That is worse than saying nothing.

        Two corrections:

        * A raw cosine floor, not a rank. Cosine is calibrated in [0,1] — an
          off-topic query scores ~0 against this index while an on-topic one
          scores high — whereas the fused RRF score is not comparable across
          queries and cannot carry a threshold.
        * `exclude_card_ids`. When asking whether you can answer a position,
          the cards *inside that position* are not candidates. Without this,
          every position trivially "answers" itself.

        `restrict_to` limits candidates to a set of card ids -- the third and
        last correction. "Do I have this?" means *I*, and on a shared corpus the
        index holds 11,643 other teams' files. Without it, an analysis scoped to
        one school answered "you already have cards for this" by pointing at
        another school's evidence, which is the same misattribution the scope
        parameter exists to prevent.

        The floor is a judgment call and depends on corpus breadth. It is a
        parameter, it is reported alongside each match, and `analysis/engine.py`
        surfaces the score so you can see how strong "you have this" really is.
        """
        if k <= 0 or not query.strip():
            return []
        self._ensure_current()
        exclude = exclude_card_ids or set()
        # When restricted, rank only the owner's candidates. Scanning the global
        # top-N and filtering afterward can miss an owner's good card entirely
        # on a shared archive, no matter how large N becomes.
        allowed = None
        if restrict_to is not None:
            allowed = set(restrict_to) - exclude
            if not allowed:
                return []
        depth = (min(k + len(exclude) + 10, len(allowed))
                 if allowed is not None else k + len(exclude) + 10)
        out: list[tuple[str, float]] = []
        for cid, score in self.vectors.query(
                query, k=depth, allowed=allowed,
                include_nonpositive=floor <= 0.0,
        ):
            if cid in exclude:
                continue
            if score < floor:
                continue
            out.append((cid, round(float(score), 4)))
            if len(out) >= k:
                break
        return out

    def similar(self, card_id: str, k: int = 10) -> list[Hit]:
        """Cards making a similar argument. Doubles as duplicate detection --
        the same card recut by three teams should surface as three near-ties."""
        if k <= 0:
            return []
        row = self.store.card(card_id)
        if not row:
            return []
        text = row["read_text"] or row["body"] or row["tag"]
        return [h for h in self.search(text, k=k + 1) if h.card_id != card_id][:k]
