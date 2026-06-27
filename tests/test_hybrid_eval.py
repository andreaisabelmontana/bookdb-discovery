"""The hybrid ranker beats CF-only and semantic-only on a held-out metric."""
from discovery.evaluate import evaluate_mean, evaluate_sparse


def test_hybrid_beats_both_on_full_data(catalog, ratings):
    """Mean leave-one-out Hit-Rate@10 over 5 held-out draws: hybrid wins outright."""
    res = evaluate_mean(catalog, ratings, seeds=5, k=10, alpha=0.6)
    cf = res["cf"]["hit_rate"]
    sem = res["semantic"]["hit_rate"]
    hyb = res["hybrid"]["hit_rate"]
    assert hyb > cf, f"hybrid {hyb:.3f} should beat cf {cf:.3f}"
    assert hyb > sem, f"hybrid {hyb:.3f} should beat semantic {sem:.3f}"


def test_hybrid_is_robust_when_cf_is_starved(catalog, ratings):
    """With CF trained on only 15% of ratings, CF collapses but the hybrid stays
    well ahead of CF -- the content channel keeps it robust."""
    res = evaluate_sparse(catalog, ratings, frac=0.15, k=10, alpha=0.6)
    cf = res["cf"]["hit_rate"]
    hyb = res["hybrid"]["hit_rate"]
    assert hyb > cf + 0.05, f"sparse: hybrid {hyb:.3f} should clearly beat cf {cf:.3f}"


def test_semantic_is_the_weakest_alone_on_full_data(catalog, ratings):
    """Sanity: on dense data CF and hybrid both beat content-only retrieval."""
    res = evaluate_mean(catalog, ratings, seeds=5, k=10, alpha=0.6)
    assert res["hybrid"]["hit_rate"] > res["semantic"]["hit_rate"]
    assert res["cf"]["hit_rate"] > res["semantic"]["hit_rate"]
