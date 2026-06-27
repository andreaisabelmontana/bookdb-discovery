"""Conversational query parser.

Transparent, rule/keyword based intent extraction. No large language model:
just regex for the "like X" pattern, a vocabulary of genre/mood/length cues, and
a fuzzy title lookup against the catalog. Turns free text such as

    "something like Dune but lighter"
    "a short mystery set in Japan"
    "cozy fantasy with found family, not too long"

into a structured Intent (seed books, genre/tag filters, mood, length, free text)
that the hybrid recommender consumes. Being keyword-based keeps it fully
explainable -- every extracted field can be traced to the phrase that triggered it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .data import Catalog

# genre cue word -> canonical catalog genre
GENRE_CUES = {
    "sci-fi": "science fiction",
    "scifi": "science fiction",
    "science fiction": "science fiction",
    "space opera": "space opera",
    "cyberpunk": "cyberpunk",
    "fantasy": "fantasy",
    "epic fantasy": "fantasy",
    "mystery": "mystery",
    "whodunit": "mystery",
    "detective": "mystery",
    "thriller": "thriller",
    "suspense": "thriller",
    "literary": "literary fiction",
    "literary fiction": "literary fiction",
    "romance": "romance",
    "adventure": "adventure",
}

# mood/tag cue word -> canonical catalog tag
TAG_CUES = {
    "cozy": "cozy",
    "cosy": "cozy",
    "comfort": "cozy",
    "comforting": "cozy",
    "wholesome": "cozy",
    "feel-good": "cozy",
    "feel good": "cozy",
    "warm": "warm",
    "heartwarming": "warm",
    "found family": "found family",
    "twist": "twist",
    "twisty": "twist",
    "psychological": "psychological",
    "eerie": "eerie",
    "creepy": "eerie",
    "unsettling": "eerie",
    "sad": "sad",
    "heartbreaking": "sad",
    "devastating": "sad",
    "tearjerker": "sad",
    "emotional": "emotional",
    "moving": "emotional",
    "magic": "magic",
    "dragons": "dragons",
    "space": "space",
    "aliens": "aliens",
    "ai": "ai",
    "artificial intelligence": "artificial intelligence",
    "obscure": "obscure",
    "gentle": "gentle",
    "quirky": "quirky",
    "offbeat": "offbeat",
}

# setting cue: phrase -> tag in catalog
SETTING_CUES = {
    "japan": "japan",
    "japanese": "japan",
    "tokyo": "japan",
    "in japan": "japan",
}

# "lighter / darker" style modifiers -> (add_tags, remove_tags) relative to seed
MOOD_MODIFIERS = {
    "lighter": {"add": ["cozy", "warm"], "drop": ["sad", "eerie"]},
    "light": {"add": ["cozy", "warm"], "drop": ["sad", "eerie"]},
    "happier": {"add": ["warm", "cozy"], "drop": ["sad"]},
    "funnier": {"add": ["humor"], "drop": ["sad"]},
    "darker": {"add": ["eerie", "psychological"], "drop": ["cozy", "warm"]},
    "sadder": {"add": ["sad", "emotional"], "drop": ["cozy"]},
    "scarier": {"add": ["eerie"], "drop": ["cozy"]},
    "more romantic": {"add": ["romance"], "drop": []},
}

LENGTH_CUES = {
    "short": "short",
    "quick": "short",
    "novella": "short",
    "not too long": "short",
    "long": "long",
    "epic": "long",
    "chunky": "long",
    "doorstopper": "long",
}

# patterns that introduce a seed book
LIKE_PATTERNS = [
    re.compile(r"(?:something|books?|read|anything|more)?\s*(?:like|similar to)\s+(.+?)(?:\s+but\b|\s+except\b|\s+that\b|\s+set\b|[,.;]|$)", re.I),
    re.compile(r"(?:in the vein of|reminds me of|fans? of)\s+(.+?)(?:\s+but\b|[,.;]|$)", re.I),
]


@dataclass
class Intent:
    raw: str
    free_text: str = ""           # cleaned text fed to semantic search
    seed_titles: list[str] = field(default_factory=list)
    seed_ids: list[int] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    drop_tags: list[str] = field(default_factory=list)   # constraints to penalize/exclude
    length: str | None = None
    modifiers: list[str] = field(default_factory=list)
    explanation: list[str] = field(default_factory=list)


def _find_seed(text: str, catalog: Catalog):
    for pat in LIKE_PATTERNS:
        m = pat.search(text)
        if m:
            cand = m.group(1).strip(" '\"")
            bid = catalog.find_by_title(cand)
            if bid is not None:
                return cand, bid, m.span()
    return None, None, None


def parse_query(text: str, catalog: Catalog) -> Intent:
    intent = Intent(raw=text)
    low = text.lower()

    # 1) seed book via "like X"
    cand, bid, span = _find_seed(text, catalog)
    if bid is not None:
        intent.seed_titles.append(catalog.title(bid))
        intent.seed_ids.append(bid)
        intent.explanation.append(f'seed book = "{catalog.title(bid)}" (from "like {cand}")')

    # 2) genres
    for cue, canon in GENRE_CUES.items():
        if re.search(rf"\b{re.escape(cue)}\b", low) and canon not in intent.genres:
            intent.genres.append(canon)
            intent.explanation.append(f'genre filter = {canon} (cue "{cue}")')

    # 3) tags / moods
    for cue, canon in TAG_CUES.items():
        if re.search(rf"\b{re.escape(cue)}\b", low) and canon not in intent.tags:
            intent.tags.append(canon)
            intent.explanation.append(f'tag = {canon} (cue "{cue}")')

    # 4) setting
    for cue, canon in SETTING_CUES.items():
        if re.search(rf"\b{re.escape(cue)}\b", low) and canon not in intent.tags:
            intent.tags.append(canon)
            intent.explanation.append(f'setting = {canon} (cue "{cue}")')

    # 5) mood modifiers ("but lighter")
    for cue, eff in MOOD_MODIFIERS.items():
        if re.search(rf"\b{re.escape(cue)}\b", low):
            intent.modifiers.append(cue)
            for t in eff["add"]:
                if t not in intent.tags:
                    intent.tags.append(t)
            for t in eff["drop"]:
                if t not in intent.drop_tags:
                    intent.drop_tags.append(t)
            intent.explanation.append(
                f'modifier "{cue}" -> prefer {eff["add"]}, avoid {eff["drop"]}'
            )

    # 6) length (check multi-word cues first)
    for cue in sorted(LENGTH_CUES, key=len, reverse=True):
        if cue in low:
            intent.length = LENGTH_CUES[cue]
            intent.explanation.append(f'length = {intent.length} (cue "{cue}")')
            break

    # 7) build free text for semantic search: drop the "like X" span and
    # stopword-ish filler so the query embeds on its descriptive content.
    ft = text
    if span:
        ft = (text[: span[0]] + " " + text[span[1] :]).strip()
    ft = re.sub(r"\b(something|anything|book|books|a|an|the|me|find|want|read|please|set|in)\b",
                " ", ft, flags=re.I)
    ft = re.sub(r"\s+", " ", ft).strip(" ,.;")
    # enrich free text with extracted tags/genres so semantic search has signal
    enrich = " ".join(intent.genres + intent.tags)
    intent.free_text = (ft + " " + enrich).strip()

    return intent
