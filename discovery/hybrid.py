"""Hybrid ranker: fuse collaborative-filtering and semantic-search signals.

Given a query that names seed book(s) and/or free text, we compute two signals:
  * cf:       item-based CF score against the seed books
  * semantic: LSA cosine against (free text + seed-book similarity)

These are fused by one of two methods (default: rrf):

  weighted : hybrid = alpha * cf + (1 - alpha) * semantic, after each component
             is max-normalized to [0, 1]. Simple, but a linear blend dilutes a
             signal that ranks a book #1 if the other signal ranks it low.

  rrf      : Reciprocal Rank Fusion. Rank books by each signal, then score each
             book by  w_cf / (rrf_k + rank_cf) + w_sem / (rrf_k + rank_sem).
             RRF is rank-based (scale-free) and rewards a book ranked highly by
             *either* signal, so it captures the union of what each channel finds
             instead of averaging them away. This is why the hybrid can beat
             both single signals: CF and content surface different correct books.

Hard filters (genre/tag/length) are applied as a post-filter on the candidate
set, not baked into the score, so they are transparent and explainable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .collab import ItemCF
from .data import Catalog
from .semantic import SemanticIndex


@dataclass
class Recommendation:
    book_id: int
    title: str
    author: str
    score: float
    cf: float
    semantic: float
    match: float = 0.0          # 0-1 display score (normalized within a result set)
    reasons: list[str] = field(default_factory=list)


class HybridRecommender:
    def __init__(
        self,
        catalog: Catalog,
        cf: ItemCF,
        semantic: SemanticIndex,
        alpha: float = 0.5,
        fusion: str = "rrf",
        rrf_k: int = 30,
    ):
        self.catalog = catalog
        self.cf = cf
        self.semantic = semantic
        self.alpha = alpha          # weight on CF (used by both fusion methods)
        self.fusion = fusion        # "rrf" (default) or "weighted"
        self.rrf_k = rrf_k

    def _semantic_scores(self, text: str, seed_ids: list[int]) -> dict[int, float]:
        """Blend free-text query similarity with seed-book similarity."""
        scores: dict[int, float] = {}
        if text and text.strip():
            scores = self.semantic.search(text)
        if seed_ids:
            seed_blend: dict[int, float] = {}
            for s in seed_ids:
                for b, v in self.semantic.similar_to(s).items():
                    seed_blend[b] = max(seed_blend.get(b, 0.0), v)
            if scores:
                scores = {b: 0.5 * scores.get(b, 0.0) + 0.5 * seed_blend.get(b, 0.0)
                          for b in self.catalog.ids}
            else:
                scores = seed_blend
        return scores

    @staticmethod
    def _ranks(scores: dict[int, float]) -> dict[int, int]:
        """Map book_id -> 1-based rank by descending score (ties broken by id)."""
        order = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return {b: i + 1 for i, (b, _) in enumerate(order)}

    def _candidate_scores(self, text: str, seed_ids: list[int], alpha: float, mode: str):
        cf_scores = self.cf.score(seed_ids) if seed_ids else {}
        sem_scores = self._semantic_scores(text, seed_ids)
        ids = [b for b in self.catalog.ids if b not in seed_ids]

        if mode == "cf":
            return {b: (cf_scores.get(b, 0.0), cf_scores.get(b, 0.0), sem_scores.get(b, 0.0))
                    for b in ids}
        if mode == "semantic":
            return {b: (sem_scores.get(b, 0.0), cf_scores.get(b, 0.0), sem_scores.get(b, 0.0))
                    for b in ids}

        # hybrid fusion
        if self.fusion == "weighted":
            return {b: (alpha * cf_scores.get(b, 0.0) + (1 - alpha) * sem_scores.get(b, 0.0),
                        cf_scores.get(b, 0.0), sem_scores.get(b, 0.0))
                    for b in ids}

        # Reciprocal Rank Fusion (default). Only rank books each signal actually
        # supports (score > 0); unsupported books get no contribution from that
        # signal, so a book ranked highly by EITHER signal still scores well.
        cf_pos = {b: v for b, v in cf_scores.items() if v > 0 and b not in seed_ids}
        sem_pos = {b: v for b, v in sem_scores.items() if v > 0 and b not in seed_ids}
        cf_rank = self._ranks(cf_pos)
        sem_rank = self._ranks(sem_pos)
        out = {}
        for b in ids:
            s = 0.0
            if b in cf_rank:
                s += alpha / (self.rrf_k + cf_rank[b])
            if b in sem_rank:
                s += (1 - alpha) / (self.rrf_k + sem_rank[b])
            out[b] = (s, cf_scores.get(b, 0.0), sem_scores.get(b, 0.0))
        return out

    def rank(
        self,
        text: str = "",
        seed_ids: list[int] | None = None,
        genres: list[str] | None = None,
        tags: list[str] | None = None,
        length: str | None = None,
        exclude: list[int] | None = None,
        alpha: float | None = None,
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> list[Recommendation]:
        """Rank books. mode in {"hybrid", "cf", "semantic"} selects the signal(s)."""
        seed_ids = seed_ids or []
        exclude = set(exclude or []) | set(seed_ids)
        a = self.alpha if alpha is None else alpha

        scored = self._candidate_scores(text, seed_ids, a, mode)
        results = []
        for b, (score, cf, sem) in scored.items():
            if b in exclude:
                continue
            if not self._passes_filters(b, genres, tags, length):
                continue
            results.append((b, score, cf, sem))

        results.sort(key=lambda x: x[1], reverse=True)
        top = results[:top_k]
        # normalize the raw fusion score to a friendly 0-1 "match" for display
        # (ranking is unchanged; RRF raw scores are tiny by construction).
        scores = [s for _, s, _, _ in top]
        hi = max(scores) if scores else 0.0
        lo = min(scores) if scores else 0.0
        span = (hi - lo) or 1.0
        recs = []
        for b, score, cf, sem in top:
            row = self.catalog.row(b)
            match = 0.5 + 0.5 * (score - lo) / span if hi > 0 else 0.0
            recs.append(
                Recommendation(
                    book_id=b,
                    title=str(row["title"]),
                    author=str(row["author"]),
                    score=round(score, 6),
                    cf=round(cf, 4),
                    semantic=round(sem, 4),
                    match=round(match, 3),
                )
            )
        return recs

    def _passes_filters(self, book_id, genres, tags, length) -> bool:
        row = self.catalog.row(book_id)
        if genres:
            book_genres = {g.lower() for g in row["genres"]}
            if not any(g.lower() in book_genres for g in genres):
                return False
        if tags:
            book_tags = {t.lower() for t in row["tags"]}
            if not any(t.lower() in book_tags for t in tags):
                return False
        if length:
            if str(row["length"]).lower() != length.lower():
                return False
        return True
