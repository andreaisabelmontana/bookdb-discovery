"""BookDB discovery: a hybrid book recommender with a conversational query layer.

Three signals:
  * collaborative filtering  -> discovery.collab.ItemCF
  * semantic search (TF-IDF/LSA) -> discovery.semantic.SemanticIndex
  * hybrid fusion            -> discovery.hybrid.HybridRecommender

Conversational layer:
  * discovery.query.parse_query  -> structured intent from free text
  * discovery.engine.DiscoveryEngine -> ties intent + signals + explanations
"""
from .data import load_catalog, load_ratings, Catalog
from .collab import ItemCF
from .semantic import SemanticIndex
from .hybrid import HybridRecommender
from .query import parse_query, Intent
from .engine import DiscoveryEngine

__all__ = [
    "load_catalog",
    "load_ratings",
    "Catalog",
    "ItemCF",
    "SemanticIndex",
    "HybridRecommender",
    "parse_query",
    "Intent",
    "DiscoveryEngine",
]
