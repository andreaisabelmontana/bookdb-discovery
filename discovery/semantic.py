"""Semantic-search signal: TF-IDF + LSA (latent semantic analysis) over book text.

Each book is represented by a blob of title + genres + tags + description.
We fit a TF-IDF vectorizer, then reduce to a dense LSA space with TruncatedSVD
so that topically-related books (and free-text queries) land near each other
even when they share few exact words. Cosine similarity in the LSA space gives
both "books like this book" and "books matching this free-text query".
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .data import Catalog


class SemanticIndex:
    def __init__(self, n_components: int = 60):
        self.n_components = n_components
        self.book_ids: list[int] = []
        self.pos: dict[int, int] = {}
        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        self.embeddings_: np.ndarray | None = None  # books x dim, L2-normalized

    def fit(self, catalog: Catalog) -> "SemanticIndex":
        self.book_ids = catalog.ids
        self.pos = {b: i for i, b in enumerate(self.book_ids)}
        docs = [catalog.text_blob(b) for b in self.book_ids]

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        tfidf = self.vectorizer.fit_transform(docs)

        n_comp = min(self.n_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        emb = self.svd.fit_transform(tfidf)
        self.embeddings_ = normalize(emb)
        return self

    def _embed_text(self, text: str) -> np.ndarray:
        assert self.vectorizer is not None and self.svd is not None, "call fit() first"
        vec = self.vectorizer.transform([text])
        emb = self.svd.transform(vec)
        return normalize(emb)[0]

    def search(self, query: str) -> dict[int, float]:
        """Cosine similarity of every book to a free-text query, in [0, 1]."""
        q = self._embed_text(query)
        sims = self.embeddings_ @ q  # cosine (both normalized)
        sims = np.clip(sims, 0.0, None)
        m = sims.max()
        if m > 0:
            sims = sims / m
        return {b: float(sims[i]) for i, b in enumerate(self.book_ids)}

    def similar_to(self, book_id: int) -> dict[int, float]:
        """Cosine similarity of every book to a given book, in [0, 1]."""
        assert self.embeddings_ is not None, "call fit() first"
        v = self.embeddings_[self.pos[book_id]]
        sims = self.embeddings_ @ v
        sims[self.pos[book_id]] = 0.0
        sims = np.clip(sims, 0.0, None)
        m = sims.max()
        if m > 0:
            sims = sims / m
        return {b: float(sims[i]) for i, b in enumerate(self.book_ids)}
