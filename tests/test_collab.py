"""Item-based collaborative filtering produces sensible neighbours."""
from discovery import ItemCF


def _genre_share(catalog, neigh, genre):
    return sum(1 for b, _ in neigh if genre in catalog.row(b)["genres"])


def test_dune_neighbours_are_mostly_scifi(catalog, ratings):
    cf = ItemCF(catalog.ids).fit(ratings)
    neigh = cf.neighbors(1, k=5)  # 1 = Dune
    # people who rate Dune highly also rate other science fiction highly, so the
    # bulk of Dune's top co-rating neighbours should be science fiction.
    assert _genre_share(catalog, neigh, "science fiction") >= 3


def test_mistborn_neighbours_are_mostly_fantasy(catalog, ratings):
    cf = ItemCF(catalog.ids).fit(ratings)
    neigh = cf.neighbors(7, k=5)  # 7 = Mistborn
    assert _genre_share(catalog, neigh, "fantasy") >= 3


def test_neighbour_scores_descending(catalog, ratings):
    cf = ItemCF(catalog.ids).fit(ratings)
    scores = [s for _, s in cf.neighbors(1, k=10)]
    assert scores == sorted(scores, reverse=True)
    assert all(s > 0 for s in scores)


def test_score_excludes_seeds(catalog, ratings):
    cf = ItemCF(catalog.ids).fit(ratings)
    scores = cf.score([1, 2])
    assert scores.get(1, 0.0) == 0.0
    assert scores.get(2, 0.0) == 0.0


def test_score_is_normalized(catalog, ratings):
    cf = ItemCF(catalog.ids).fit(ratings)
    scores = cf.score([7])  # Mistborn
    assert scores
    assert max(scores.values()) <= 1.0 + 1e-9
    assert min(scores.values()) >= 0.0
