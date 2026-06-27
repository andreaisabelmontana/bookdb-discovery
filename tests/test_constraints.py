""""like X but <constraint>" respects the constraint."""


def _tags(engine, rec):
    return set(engine.catalog.row(rec.book_id)["tags"])


def test_lighter_removes_sad_and_eerie(engine):
    """'like Never Let Me Go' surfaces sad books; 'but lighter' must drop them."""
    plain = engine.ask("something like Never Let Me Go", top_k=6).recommendations
    plain_titles = {r.title for r in plain}
    # the unconstrained list contains explicitly sad books...
    assert {"Klara and the Sun", "A Little Life", "Beloved"} & plain_titles

    lighter = engine.ask("something like Never Let Me Go but lighter", top_k=6).recommendations
    # ...and the 'lighter' list contains no sad/eerie books at all
    for r in lighter:
        assert not (_tags(engine, r) & {"sad", "eerie"}), r.title


def test_short_constraint_only_returns_short_books(engine):
    ans = engine.ask("a short mystery", top_k=5)
    assert ans.intent.length == "short"
    for r in ans.recommendations:
        assert str(engine.catalog.row(r.book_id)["length"]) == "short", r.title


def test_genre_constraint_respected(engine):
    ans = engine.ask("a thriller with a big twist", top_k=5)
    for r in ans.recommendations:
        assert "thriller" in engine.catalog.row(r.book_id)["genres"], r.title


def test_like_dune_but_lighter_changes_ranking(engine):
    plain = [r.title for r in engine.ask("something like Dune", top_k=5).recommendations]
    lighter = [r.title for r in engine.ask("something like Dune but lighter", top_k=5).recommendations]
    # the constraint must actually change the result ordering / membership
    assert plain != lighter


def test_japan_query_returns_japanese_books(engine):
    titles = [r.title for r in engine.ask("a mystery set in Japan", top_k=4).recommendations]
    japan = {"The Tokyo Zodiac Murders", "Malice", "The Devotion of Suspect X"}
    assert len(japan & set(titles)) >= 2


def test_recommendations_carry_explanations(engine):
    ans = engine.ask("a cozy fantasy with found family", top_k=3)
    assert ans.recommendations
    for r in ans.recommendations:
        assert r.reasons and isinstance(r.reasons[0], str)
