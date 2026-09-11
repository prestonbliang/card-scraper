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
import os
import pickle
import re
from dataclasses import dataclass

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
    lexical_rank: int | None = None
    vector_rank: int | None = None

    def to_dict(self) -> dict:
        return {
            "card_id": self.card_id, "score": round(self.score, 5),
            "tag": self.tag, "read_text": self.read_text,
            "cite_raw": self.cite_raw, "block": self.block, "side": self.side,
            "source_id": self.source_id,
            "lexical_rank": self.lexical_rank, "vector_rank": self.vector_rank,
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

        n_comp = min(self.dims, max(2, min(X.shape) - 1))
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

    def save(self, path: str, fingerprint: str) -> None:
        if self.matrix is None or self._vec is None:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump({
                "format": self.FORMAT, "fingerprint": fingerprint,
                "dims": self.dims, "card_ids": self.card_ids,
                "vec": self._vec, "svd": self._svd, "matrix": self.matrix,
            }, fh, protocol=pickle.HIGHEST_PROTOCOL)
        # atomic: a half-written cache must never be loadable
        os.replace(tmp, path)

    def load(self, path: str, fingerprint: str) -> bool:
        if not path or not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as fh:
                blob = pickle.load(fh)
        except Exception:
            return False
        if blob.get("format") != self.FORMAT:
            return False
        if blob.get("fingerprint") != fingerprint:
            return False    # corpus changed; refit rather than serve a stale index
        self.dims = blob["dims"]
        self.card_ids = blob["card_ids"]
        self._vec = blob["vec"]
        self._svd = blob["svd"]
        self.matrix = blob["matrix"]
        return True

    def query(self, text: str, k: int = 50) -> list[tuple[str, float]]:
        if self.matrix is None or self._vec is None or not self.card_ids:
            return []
        from sklearn.preprocessing import normalize

        q = self._vec.transform([text])
        qd = self._svd.transform(q) if self._svd is not None else q.toarray()
        qd = normalize(qd).astype(self.matrix.dtype, copy=False)
        sims = (self.matrix @ qd.T).ravel()
        order = np.argsort(-sims)[:k]
        return [(self.card_ids[i], float(sims[i])) for i in order]


class SearchEngine:
    def __init__(self, store: Store, cache_path: str | None = None):
        self.store = store
        self.vectors = VectorIndex()
        self._built = False
        if cache_path is None and getattr(store, "path", None):
            cache_path = os.path.splitext(store.path)[0] + ".index"
        self.cache_path = cache_path

    def build(self, use_cache: bool = True, progress=None) -> None:
        say = progress or (lambda *_: None)
        # Fingerprint from ids alone, so a cache hit never pays to load the
        # whole corpus into memory.
        fp = VectorIndex.fingerprint(self.store.card_ids())
        if use_cache and self.cache_path and self.vectors.load(self.cache_path, fp):
            say(f"loaded cached index ({len(self.vectors.card_ids)} cards)")
            self._built = True
            return
        rows = self.store.all_cards()
        say(f"fitting index over {len(rows)} cards (cached after this)")
        self.vectors.build(rows)
        if use_cache and self.cache_path:
            try:
                self.vectors.save(self.cache_path, fp)
            except Exception:
                pass    # a cache we cannot write is not a reason to fail a search
        self._built = True

    def _lexical(self, query: str, k: int) -> list[tuple[str, float]]:
        try:
            rows = self.store.conn.execute(
                """SELECT m.card_id AS card_id, bm25(cards_fts, 8.0, 4.0, 2.0, 1.0) AS s
                   FROM cards_fts
                   JOIN fts_map m ON m.rowid = cards_fts.rowid
                   WHERE cards_fts MATCH ?
                   ORDER BY s LIMIT ?""",
                (_fts_escape(query), k),
            ).fetchall()
        except Exception:
            return []
        return [(r["card_id"], -float(r["s"])) for r in rows]

    def search(self, query: str, k: int = 25, *, side: str | None = None,
               author: str | None = None, year_min: int | None = None,
               year_max: int | None = None, block: str | None = None,
               min_read_ratio: float | None = None) -> list[Hit]:
        if not self._built:
            self.build()

        pool = max(k * 4, 60)
        lex = self._lexical(query, pool)
        vec = self.vectors.query(query, pool)

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
            f"SELECT * FROM cards WHERE card_id IN ({placeholders})", ids)}

        hits: list[Hit] = []
        for cid in ids:
            r = rows.get(cid)
            if not r:
                continue
            if side and r["side"] != side:
                continue
            if author and (r["cite_author"] or "").lower() != author.lower():
                continue
            if year_min and (r["cite_year"] or 0) < year_min:
                continue
            if year_max and (r["cite_year"] or 9999) > year_max:
                continue
            if block and block.lower() not in (r["block"] or "").lower():
                continue
            if min_read_ratio is not None and (r["read_ratio"] or 0) < min_read_ratio:
                continue
            hits.append(Hit(
                card_id=cid, score=fused[cid], tag=r["tag"],
                read_text=r["read_text"] or "", cite_raw=r["cite_raw"] or "",
                block=r["block"], side=r["side"] or "unknown",
                source_id=r["source_id"] or "",
                lexical_rank=lex_rank.get(cid), vector_rank=vec_rank.get(cid),
            ))
            if len(hits) >= k:
                break
        return hits

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
        if not self._built:
            self.build()
        exclude = exclude_card_ids or set()
        # When restricted, scan deeper: the owner's cards may sit well below the
        # global top-k on a corpus this size, and stopping early would report
        # "not in your files" for evidence that is.
        depth = k + len(exclude) + (400 if restrict_to is not None else 10)
        out: list[tuple[str, float]] = []
        for cid, score in self.vectors.query(query, k=depth):
            if cid in exclude:
                continue
            if restrict_to is not None and cid not in restrict_to:
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
        row = self.store.card(card_id)
        if not row:
            return []
        text = row["read_text"] or row["body"] or row["tag"]
        return [h for h in self.search(text, k=k + 1) if h.card_id != card_id][:k]
