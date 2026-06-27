"""Item-based collaborative filtering over a ratings matrix.

Builds an item-item cosine similarity matrix from mean-centered user ratings,
then scores candidate books against a set of seed books (the things a query is
"like"). This is the classic item-based CF of Sarwar et al. (2001): the score of
an item is a similarity-weighted vote from the seed items.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .data import ratings_matrix


class ItemCF:
    def __init__(self, book_ids: list[int]):
        self.book_ids = list(book_ids)
        self.pos = {b: i for i, b in enumerate(self.book_ids)}
        self.sim_: np.ndarray | None = None

    def fit(self, ratings) -> "ItemCF":
        mat, _ = ratings_matrix(ratings, self.book_ids)  # users x items
        # mean-center each user's ratings over the items they actually rated,
        # so we capture *relative* preference, not rating-scale bias.
        mask = mat > 0
        counts = mask.sum(axis=1, keepdims=True)
        counts[counts == 0] = 1
        user_mean = (mat.sum(axis=1, keepdims=True)) / counts
        centered = np.where(mask, mat - user_mean, 0.0)

        items = centered.T  # items x users
        sim = cosine_similarity(items)
        np.fill_diagonal(sim, 0.0)
        sim[sim < 0] = 0.0  # keep only positive co-preference
        self.sim_ = sim
        return self

    def neighbors(self, book_id: int, k: int = 10) -> list[tuple[int, float]]:
        """Top-k most similar books to a single book."""
        assert self.sim_ is not None, "call fit() first"
        i = self.pos[book_id]
        order = np.argsort(self.sim_[i])[::-1]
        out = []
        for j in order:
            if self.sim_[i, j] <= 0:
                break
            out.append((self.book_ids[j], float(self.sim_[i, j])))
            if len(out) >= k:
                break
        return out

    def score(self, seed_ids: list[int]) -> dict[int, float]:
        """Similarity-weighted CF score for every book given seed books.

        Returns a dict book_id -> score in [0, 1] (max-normalized). Seed books
        are excluded from the result.
        """
        assert self.sim_ is not None, "call fit() first"
        seeds = [s for s in seed_ids if s in self.pos]
        if not seeds:
            return {}
        agg = np.zeros(len(self.book_ids))
        for s in seeds:
            agg += self.sim_[self.pos[s]]
        agg /= len(seeds)
        for s in seeds:
            agg[self.pos[s]] = 0.0
        m = agg.max()
        if m > 0:
            agg = agg / m
        return {b: float(agg[i]) for i, b in enumerate(self.book_ids)}
