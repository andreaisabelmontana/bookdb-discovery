r"""Held-out evaluation: does the hybrid blend beat either signal alone?

Leave-one-out protocol over the simulated users:
  * For each user who liked >= 6 books (rated 4-5 stars), hold ONE liked book
    out as the target t. The user's other liked books are the "seeds" S -- the
    things we already know they enjoy.
  * Each ranker scores all books from S (excluding S); a hit = target in top-K.
      - cf       : item-based collaborative filtering over the seeds.
      - semantic : LSA text similarity to the seed books' descriptions.
      - hybrid   : Reciprocal Rank Fusion of the two (HybridRecommender, rrf).
  * Report Hit-Rate@K and MRR.

Why the hybrid wins: CF and content surface DIFFERENT correct books -- CF finds
co-rated targets that are textually unlike the seeds, content finds topically
similar targets that are thinly co-rated. RRF rewards a book ranked highly by
EITHER signal, so the fused list captures the union and beats both single
signals on Hit-Rate@10 (robustly across random held-out draws).

A second, sparse-data view (`evaluate_sparse`) trains CF on a fraction of the
ratings: there CF collapses while content holds, so the hybrid is also the most
ROBUST ranker -- it is never the worst.

All numbers printed by demo.py and asserted by the tests come from this module
on the committed data; nothing is hard-coded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .collab import ItemCF
from .data import Catalog
from .hybrid import HybridRecommender
from .semantic import SemanticIndex

MODES = ["cf", "semantic", "hybrid"]


def _liked_books_per_user(ratings: pd.DataFrame, min_liked: int = 6):
    liked = ratings[ratings["rating"] >= 4]
    groups = liked.groupby("user_id")["book_id"].apply(list)
    return {u: list(bs) for u, bs in groups.items() if len(bs) >= min_liked}


def evaluate(
    catalog: Catalog,
    ratings: pd.DataFrame,
    train_ratings: pd.DataFrame | None = None,
    k: int = 10,
    alpha: float = 0.6,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Leave-one-out Hit-Rate@K and MRR for cf / semantic / hybrid.

    `train_ratings` (what CF is fitted on) may be a subsample of `ratings`; the
    held-out targets always come from the full `ratings`. If None, CF trains on
    the full ratings.
    """
    train = ratings if train_ratings is None else train_ratings
    cf = ItemCF(catalog.ids).fit(train)
    sem = SemanticIndex().fit(catalog)
    hybrid = HybridRecommender(catalog, cf, sem, alpha=alpha, fusion="rrf")

    user_liked = _liked_books_per_user(ratings)
    rng = np.random.default_rng(seed)

    hits = {m: 0 for m in MODES}
    rr = {m: 0.0 for m in MODES}
    n = 0

    for books in user_liked.values():
        target = books[int(rng.integers(len(books)))]
        seeds = [b for b in books if b != target]
        if not seeds:
            continue
        n += 1
        seed_text = " ".join(catalog.text_blob(b) for b in seeds)
        for m in MODES:
            text = "" if m == "cf" else seed_text
            recs = hybrid.rank(text=text, seed_ids=seeds, exclude=seeds, top_k=k, mode=m)
            ranked = [r.book_id for r in recs]
            if target in ranked:
                hits[m] += 1
                rr[m] += 1.0 / (ranked.index(target) + 1)

    return {
        m: {"hit_rate": hits[m] / n if n else 0.0,
            "mrr": rr[m] / n if n else 0.0,
            "n": n}
        for m in MODES
    }


def evaluate_mean(
    catalog: Catalog,
    ratings: pd.DataFrame,
    seeds: int = 5,
    k: int = 10,
    alpha: float = 0.6,
) -> dict[str, dict[str, float]]:
    """Average the leave-one-out metrics over `seeds` held-out draws.

    Averaging removes the per-draw noise of which single book is held out, giving
    a stable comparison: on this the hybrid beats both single signals.
    """
    runs = [evaluate(catalog, ratings, k=k, alpha=alpha, seed=s) for s in range(seeds)]
    out = {}
    for m in MODES:
        out[m] = {
            "hit_rate": float(np.mean([r[m]["hit_rate"] for r in runs])),
            "mrr": float(np.mean([r[m]["mrr"] for r in runs])),
            "n": runs[0][m]["n"],
            "seeds": seeds,
        }
    return out


def evaluate_sparse(
    catalog: Catalog,
    ratings: pd.DataFrame,
    frac: float = 0.15,
    k: int = 10,
    alpha: float = 0.6,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Same evaluation, but CF is trained on a `frac` subsample of the ratings."""
    train = ratings.sample(frac=frac, random_state=seed + 1)
    return evaluate(catalog, ratings, train_ratings=train, k=k, alpha=alpha, seed=seed)


def report(catalog: Catalog, ratings: pd.DataFrame, k: int = 10, alpha: float = 0.6):
    """Return the headline (mean) and sparse-data tables for demo.py / README."""
    return {
        "k": k,
        "alpha": alpha,
        "full": evaluate_mean(catalog, ratings, seeds=5, k=k, alpha=alpha),
        "sparse": evaluate_sparse(catalog, ratings, frac=0.15, k=k, alpha=alpha),
    }
