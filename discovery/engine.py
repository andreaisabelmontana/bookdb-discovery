"""DiscoveryEngine: the conversational front door.

Parses a natural-language request into an Intent, runs the hybrid recommender
with the extracted seed books / filters, applies "drop" constraints from mood
modifiers ("but lighter" -> avoid sad/eerie), and attaches a human-readable
explanation to every recommendation.
"""
from __future__ import annotations

from dataclasses import dataclass

from .collab import ItemCF
from .data import Catalog, load_catalog, load_ratings
from .hybrid import HybridRecommender, Recommendation
from .query import Intent, parse_query
from .semantic import SemanticIndex


@dataclass
class Answer:
    intent: Intent
    recommendations: list[Recommendation]


class DiscoveryEngine:
    def __init__(self, catalog: Catalog, ratings, alpha: float = 0.5):
        self.catalog = catalog
        self.cf = ItemCF(catalog.ids).fit(ratings)
        self.semantic = SemanticIndex().fit(catalog)
        self.hybrid = HybridRecommender(catalog, self.cf, self.semantic, alpha=alpha)

    @classmethod
    def from_data(cls, alpha: float = 0.5) -> "DiscoveryEngine":
        return cls(load_catalog(), load_ratings(), alpha=alpha)

    def ask(self, text: str, top_k: int = 5, mode: str = "hybrid") -> Answer:
        intent = parse_query(text, self.catalog)

        # mood "drop" tags become a soft exclude: filter out books that carry a
        # tag the user asked to avoid (e.g. "but lighter" -> drop sad/eerie).
        exclude = self._drop_excludes(intent)

        recs = self.hybrid.rank(
            text=intent.free_text,
            seed_ids=intent.seed_ids,
            genres=intent.genres or None,
            tags=None,  # tags steer via free_text/semantic, not hard genre filter
            length=intent.length,
            exclude=exclude,
            top_k=top_k,
            mode=mode,
        )
        for r in recs:
            r.reasons = self._reasons(r, intent)
        return Answer(intent=intent, recommendations=recs)

    def _drop_excludes(self, intent: Intent) -> list[int]:
        if not intent.drop_tags:
            return []
        drop = {t.lower() for t in intent.drop_tags}
        out = []
        for b in self.catalog.ids:
            row = self.catalog.row(b)
            book_tags = {t.lower() for t in row["tags"]}
            if book_tags & drop:
                out.append(b)
        return out

    def _reasons(self, rec: Recommendation, intent: Intent) -> list[str]:
        reasons = []
        row = self.catalog.row(rec.book_id)
        if intent.seed_ids and rec.cf > 0.2:
            seed = self.catalog.title(intent.seed_ids[0])
            reasons.append(f"readers who liked {seed} also rate this highly (CF {rec.cf:.2f})")
        if rec.semantic > 0.2:
            shared = []
            for g in intent.genres:
                if g in [x.lower() for x in row["genres"]]:
                    shared.append(g)
            for t in intent.tags:
                if t in [x.lower() for x in row["tags"]]:
                    shared.append(t)
            if shared:
                reasons.append(f"matches your request on: {', '.join(sorted(set(shared)))} (semantic {rec.semantic:.2f})")
            else:
                reasons.append(f"topically close to your query (semantic {rec.semantic:.2f})")
        if intent.length and str(row["length"]).lower() == intent.length:
            reasons.append(f"{intent.length} length, as asked")
        if not reasons:
            reasons.append("strong overall blend of both signals")
        return reasons
