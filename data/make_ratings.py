"""Generate a deterministic synthetic ratings matrix for the 40-book catalog.

Honest note: ratings are SYNTHETIC. Each simulated user is assigned a small set
of "taste" genres/tags; they rate books they would plausibly enjoy highly and a
few off-taste books lower, with mild noise. This gives item-based collaborative
filtering real, learnable co-rating structure (books that share an audience get
rated by the same users) without pretending to be a real Goodreads dump.

Run from the repo root:  python data/make_ratings.py
Output is committed as data/ratings.csv so the package needs no generation step.
"""
import csv
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))


def load_books():
    books = []
    with open(os.path.join(HERE, "books.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["book_id"] = int(row["book_id"])
            row["genres"] = set(row["genres"].split("|"))
            row["tags"] = set(row["tags"].split("|"))
            books.append(row)
    return books


def main():
    rng = random.Random(20240627)
    books = load_books()

    # Taste profiles: each is a set of genre/tag keywords. A user drawn from a
    # profile rates books that overlap the profile highly. The profiles span the
    # catalog at the genre/sub-theme level -- this is the structure TF-IDF/LSA
    # content similarity captures from the book text.
    profiles = [
        {"science fiction", "space", "aliens"},          # space opera / adventure SF
        {"science fiction", "cyberpunk", "dystopia"},     # cyberpunk
        {"science fiction", "eerie", "surreal"},          # weird / literary SF
        {"fantasy", "magic", "dragons"},                  # epic fantasy
        {"fantasy", "magic", "coming of age"},            # YA / quest fantasy
        {"fantasy", "grimdark", "war"},                   # grimdark
        {"mystery", "whodunit", "detective"},             # classic mystery
        {"mystery", "japan", "puzzle"},                   # Japanese mystery
        {"thriller", "psychological", "twist"},           # psych thriller
        {"literary fiction", "emotional", "prose"},       # literary
        {"literary fiction", "japan", "surreal"},         # Murakami-ish literary
        {"cozy", "warm", "found family"},                 # cozy / feel-good
    ]

    # Each user also has a few *idiosyncratic* favourites drawn at random from a
    # small per-user pool, independent of their profile. Different users share
    # different idiosyncratic books, which creates fine-grained CO-RATING
    # structure that is invisible to content similarity (the books need not be
    # textually alike) but visible to collaborative filtering. The genre profile
    # is what content captures; the idiosyncratic co-rating is what CF captures.
    # The two are genuinely complementary, so a fused ranker can beat each alone.
    n_books = len(books)
    rows = []
    user_id = 0
    per_profile = 30  # 12 profiles * 30 -> 360 users
    for profile in profiles:
        for _ in range(per_profile):
            user_id += 1
            rated_any = False
            for b in books:
                terms = b["genres"] | b["tags"]
                overlap = len(terms & profile)
                if overlap >= 1:
                    base = 3.6 + 0.5 * min(overlap, 3)
                    rating = base + rng.gauss(0, 0.4)
                    if rng.random() < 0.6:  # users don't rate everything they'd like
                        rating = round(min(5.0, max(1.0, rating)))
                        rows.append((user_id, b["book_id"], int(rating)))
                        rated_any = True
                else:
                    if rng.random() < 0.06:  # rare off-taste rating
                        rating = 2.6 + rng.gauss(0, 0.7)
                        rating = round(min(5.0, max(1.0, rating)))
                        rows.append((user_id, b["book_id"], int(rating)))
                        rated_any = True
            # idiosyncratic favourites: pick 2 books this user loves, drawn from
            # a small pool keyed to the user so fine-grained co-rating clusters
            # form. Kept modest so it complements (not drowns) the genre core.
            pool_rng = random.Random(1000 + user_id // 6)  # users share pools in groups of 6
            pool = pool_rng.sample(range(1, n_books + 1), 5)
            for bid in rng.sample(pool, 2):
                rows.append((user_id, bid, 5 if rng.random() < 0.5 else 4))
                rated_any = True
            if not rated_any:  # guarantee every user has at least one rating
                b = rng.choice(books)
                rows.append((user_id, b["book_id"], 4))

    # dedup (user, book): keep the highest rating if a book was added twice
    best: dict[tuple[int, int], int] = {}
    for u, b, r in rows:
        key = (u, b)
        if key not in best or r > best[key]:
            best[key] = r
    rows = [(u, b, r) for (u, b), r in best.items()]
    rows.sort()

    out = os.path.join(HERE, "ratings.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "book_id", "rating"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} ratings from {user_id} users to {out}")


if __name__ == "__main__":
    main()
