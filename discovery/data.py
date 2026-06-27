"""Catalog and ratings loading."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@dataclass
class Catalog:
    """In-memory book catalog with convenient lookups.

    books: DataFrame indexed by book_id with columns
           title, author, genres (list), tags (list), length, year, description.
    """

    books: pd.DataFrame

    @property
    def ids(self) -> list[int]:
        return list(self.books.index)

    def title(self, book_id: int) -> str:
        return str(self.books.loc[book_id, "title"])

    def row(self, book_id: int) -> pd.Series:
        return self.books.loc[book_id]

    def find_by_title(self, query: str) -> int | None:
        """Case-insensitive title match. Exact first, then substring, then token overlap."""
        q = query.strip().lower()
        if not q:
            return None
        titles = self.books["title"].str.lower()
        exact = titles[titles == q]
        if len(exact):
            return int(exact.index[0])
        contains = titles[titles.str.contains(_escape(q), regex=True)]
        if len(contains):
            # shortest containing title is the most specific match
            return int(contains.str.len().idxmin())
        # token overlap fallback (e.g. "lord of the rings" -> "The Lord of the Rings")
        q_tokens = set(q.split())
        best_id, best_score = None, 0
        for bid, t in titles.items():
            score = len(q_tokens & set(t.split()))
            if score > best_score:
                best_id, best_score = int(bid), score
        return best_id if best_score >= 2 else None

    def text_blob(self, book_id: int) -> str:
        r = self.books.loc[book_id]
        genres = " ".join(r["genres"])
        tags = " ".join(r["tags"])
        # weight genres/tags by repetition so they carry signal in TF-IDF
        return f"{r['title']}. {genres} {genres} {tags} {tags}. {r['description']}"


def _escape(s: str) -> str:
    import re

    return re.escape(s)


def load_catalog(path: str | None = None) -> Catalog:
    path = path or os.path.join(DATA_DIR, "books.csv")
    df = pd.read_csv(path)
    df["genres"] = df["genres"].apply(lambda s: s.split("|"))
    df["tags"] = df["tags"].apply(lambda s: s.split("|"))
    df = df.set_index("book_id")
    return Catalog(books=df)


def load_ratings(path: str | None = None) -> pd.DataFrame:
    path = path or os.path.join(DATA_DIR, "ratings.csv")
    return pd.read_csv(path)


def ratings_matrix(ratings: pd.DataFrame, book_ids: list[int]):
    """Return (users x items) dense matrix aligned to book_ids and the user index."""
    users = sorted(ratings["user_id"].unique())
    u_pos = {u: i for i, u in enumerate(users)}
    b_pos = {b: i for i, b in enumerate(book_ids)}
    mat = np.zeros((len(users), len(book_ids)), dtype=np.float64)
    for u, b, r in ratings[["user_id", "book_id", "rating"]].itertuples(index=False):
        if b in b_pos:
            mat[u_pos[u], b_pos[b]] = r
    return mat, users
