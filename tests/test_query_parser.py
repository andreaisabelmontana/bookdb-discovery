"""The conversational parser extracts the right seed book + filters."""
from discovery import parse_query


def _titles(catalog, ids):
    return [catalog.title(i) for i in ids]


def test_like_extracts_seed_book(catalog):
    intent = parse_query("something like Dune but lighter", catalog)
    assert _titles(catalog, intent.seed_ids) == ["Dune"]


def test_like_with_fuzzy_title(catalog):
    intent = parse_query("anything similar to the lord of the rings", catalog)
    assert "The Lord of the Rings" in _titles(catalog, intent.seed_ids)


def test_lighter_modifier_sets_drop_and_add_tags(catalog):
    intent = parse_query("something like Dune but lighter", catalog)
    assert "lighter" in intent.modifiers
    # "lighter" should steer toward cozy/warm and away from sad/eerie
    assert {"cozy", "warm"} & set(intent.tags)
    assert {"sad", "eerie"} & set(intent.drop_tags)


def test_darker_modifier(catalog):
    intent = parse_query("a fantasy like Mistborn but darker", catalog)
    assert "Mistborn" in _titles(catalog, intent.seed_ids)
    assert "eerie" in intent.tags or "psychological" in intent.tags
    assert "cozy" in intent.drop_tags


def test_short_mystery_in_japan(catalog):
    intent = parse_query("a short mystery set in Japan", catalog)
    assert intent.length == "short"
    assert "mystery" in intent.genres
    assert "japan" in intent.tags
    assert intent.seed_ids == []  # no "like X" -> no seed


def test_cozy_fantasy_found_family(catalog):
    intent = parse_query("a cozy fantasy with found family, not too long", catalog)
    assert "fantasy" in intent.genres
    assert "cozy" in intent.tags
    assert "found family" in intent.tags
    assert intent.length == "short"  # "not too long" -> short


def test_thriller_twist(catalog):
    intent = parse_query("a thriller with a big twist", catalog)
    assert "thriller" in intent.genres
    assert "twist" in intent.tags


def test_no_false_seed_for_plain_query(catalog):
    # a plain descriptive query with no "like X" must not invent a seed book
    intent = parse_query("a sad literary novel", catalog)
    assert intent.seed_ids == []
    assert "literary fiction" in intent.genres
    assert "sad" in intent.tags


def test_free_text_drops_the_like_clause(catalog):
    intent = parse_query("something like Dune but lighter", catalog)
    # the seed title should not bleed into the semantic free text
    assert "dune" not in intent.free_text.lower()


def test_explanation_is_populated(catalog):
    intent = parse_query("a short mystery set in Japan", catalog)
    assert intent.explanation  # every extracted field is traceable
    assert any("length" in e for e in intent.explanation)
