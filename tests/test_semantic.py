"""Semantic search returns topically-correct books."""
from discovery import SemanticIndex


def top_titles(catalog, scores, n=5):
    order = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [catalog.title(b) for b, _ in order[:n]]


def test_search_japanese_mystery(catalog):
    idx = SemanticIndex().fit(catalog)
    top = top_titles(catalog, idx.search("mystery murder set in Japan"), n=5)
    # the three Japanese mysteries should dominate the top of the list
    japan_mysteries = {"The Tokyo Zodiac Murders", "Malice", "The Devotion of Suspect X"}
    assert len(japan_mysteries & set(top)) >= 2


def test_search_cozy_found_family(catalog):
    idx = SemanticIndex().fit(catalog)
    top = top_titles(catalog, idx.search("cozy warm found family"), n=3)
    assert "The House in the Cerulean Sea" in top or "Legends & Lattes" in top


def test_search_space_survival(catalog):
    idx = SemanticIndex().fit(catalog)
    top = top_titles(catalog, idx.search("astronaut stranded space survival"), n=4)
    assert "The Martian" in top or "Project Hail Mary" in top


def test_similar_to_dune_is_scifi(catalog):
    idx = SemanticIndex().fit(catalog)
    top_ids = sorted(idx.similar_to(1).items(), key=lambda kv: kv[1], reverse=True)[:5]
    # nearest neighbours of Dune (id 1) should be science fiction
    for b, _ in top_ids:
        genres = catalog.row(b)["genres"]
        assert "science fiction" in genres, catalog.title(b)


def test_similar_excludes_self(catalog):
    idx = SemanticIndex().fit(catalog)
    assert idx.similar_to(1).get(1, 0.0) == 0.0


def test_scores_are_normalized(catalog):
    idx = SemanticIndex().fit(catalog)
    scores = idx.search("a fantasy quest with dragons and magic")
    assert max(scores.values()) <= 1.0 + 1e-9
    assert min(scores.values()) >= 0.0
