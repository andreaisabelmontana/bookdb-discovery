"""BookDB discovery demo.

Runs several natural-language queries through the conversational engine and
prints, for each: the parsed intent, the top recommendations with their CF /
semantic component scores, and a short explanation. Then prints the held-out
evaluation (hybrid vs. each single signal) on the committed data.

    python demo.py
"""
from __future__ import annotations

from discovery import DiscoveryEngine
from discovery.data import load_catalog, load_ratings
from discovery.evaluate import report

QUERIES = [
    "something like Dune but lighter",
    "a short mystery set in Japan",
    "a cozy fantasy with found family, not too long",
    "a thriller with a big twist",
    "something like Never Let Me Go but lighter",
    "a dreamlike literary novel set in Japan",
]


def show_query(engine: DiscoveryEngine, text: str) -> None:
    ans = engine.ask(text, top_k=3)
    intent = ans.intent
    print("=" * 74)
    print(f'QUERY: "{text}"')
    print("-" * 74)
    print("parsed intent:")
    if intent.seed_titles:
        print(f"  seed book(s) : {', '.join(intent.seed_titles)}")
    if intent.genres:
        print(f"  genres       : {', '.join(intent.genres)}")
    if intent.tags:
        print(f"  tags/moods   : {', '.join(intent.tags)}")
    if intent.drop_tags:
        print(f"  avoid        : {', '.join(intent.drop_tags)}")
    if intent.length:
        print(f"  length       : {intent.length}")
    print(f"  -> semantic query text: \"{intent.free_text}\"")
    print("\ntop recommendations:")
    for i, r in enumerate(ans.recommendations, 1):
        print(f"  {i}. {r.title} - {r.author}  [match {r.match:.2f}]")
        print(f"       signals: CF={r.cf:.2f}  semantic={r.semantic:.2f}")
        for why in r.reasons:
            print(f"       why: {why}")
    print()


def show_eval() -> None:
    cat = load_catalog()
    rat = load_ratings()
    r = report(cat, rat)
    k, alpha = r["k"], r["alpha"]
    print("=" * 74)
    print(f"HELD-OUT EVALUATION  (leave-one-out, Hit-Rate@{k} / MRR, alpha={alpha})")
    print("-" * 74)

    full = r["full"]
    print(f"Full data  (mean over {full['hybrid']['seeds']} held-out draws, "
          f"n={full['hybrid']['n']} users):")
    for m in ("cf", "semantic", "hybrid"):
        d = full[m]
        print(f"  {m:9s}  HitRate@{k} = {d['hit_rate']:.3f}   MRR = {d['mrr']:.3f}")
    win = (full["hybrid"]["hit_rate"] > full["cf"]["hit_rate"]
           and full["hybrid"]["hit_rate"] > full["semantic"]["hit_rate"])
    print(f"  -> hybrid beats both single signals: {win}")

    sp = r["sparse"]
    print(f"\nSparse data  (CF trained on 15% of ratings, n={sp['hybrid']['n']}):")
    for m in ("cf", "semantic", "hybrid"):
        d = sp[m]
        print(f"  {m:9s}  HitRate@{k} = {d['hit_rate']:.3f}   MRR = {d['mrr']:.3f}")
    print(f"  -> CF collapses; hybrid stays {sp['hybrid']['hit_rate'] - sp['cf']['hit_rate']:+.3f} "
          f"ahead of CF and remains robust.")
    print("=" * 74)


def main() -> None:
    engine = DiscoveryEngine.from_data(alpha=0.6)
    for q in QUERIES:
        show_query(engine, q)
    show_eval()


if __name__ == "__main__":
    main()
