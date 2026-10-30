"""How things work (kind H): GenericsKB facts, ATOMIC 2020 commonsense, and Cosmopedia's
children's stories and wikiHow articles about everyday things."""

import csv
import re
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

from eve.data.draw import pick, shuffled
from eve.data.tinytools.common import TRAVEL, Context, Doc, chunks, clean, plural

# ---- everyday nouns --------------------------------------------------------------------------


def everyday_noun(word: str, nouns: frozenset[str]) -> str | None:
    """The everyday noun ``word`` is a form of (kettle, kettles, batteries -> battery), or None."""
    word = word.lower()
    for form in (word, word[:-3] + "y" if word.endswith("ies") else "", word[:-2], word[:-1]):
        if form in nouns and (form == word or word.endswith("s")):
            return form
    return None


def everyday_nouns_in(text: str, nouns: frozenset[str]) -> set[str]:
    """The distinct everyday nouns a text uses."""
    found = (everyday_noun(w, nouns) for w in re.findall(r"[a-z]+", text.lower()))
    return {n for n in found if n}


# ---- GenericsKB ------------------------------------------------------------------------------

# Sentences people wrote; ConceptNet, WordNet and TupleKB ones are made from triples ("Dogs
# has-part mouths.") and ConceptNet's are share-alike.
GENERICS_NATURAL = ("Waterloo", "ARC", "SimpleWikipedia")
GENERICS_MIN_SCORE = 0.5  # GenericsKB's own quality score; lower ones are often odd
GENERICS_MAX_WORDS = 25
GENERICS_LOW, GENERICS_HIGH = 3, 8  # facts per document


def genericskb(raw: Path, ctx: Context) -> Iterator[Doc]:
    """GenericsKB-Best's sentences that people wrote, about everyday things (not about kinds of
    people, where its generalizations go wrong), a few facts about one noun per document."""
    columns = ["source", "term", "generic_sentence", "score"]
    table = pq.read_table(raw / "genericskb/best.parquet", columns=columns)
    by_noun: dict[str, list[str]] = {}
    things = ctx.nouns - ctx.person_nouns
    rows = zip(*(table.column(c).to_pylist() for c in columns), strict=True)
    for source, term, sentence, score in rows:
        sentence, words = clean(sentence), (term or "").split()
        if source not in GENERICS_NATURAL:
            ctx.drop("synthesized from triples")
        elif float(score or 0) < GENERICS_MIN_SCORE:
            ctx.drop("low quality score")
        elif not words or (noun := everyday_noun(words[-1], things)) is None:
            ctx.drop("not about an everyday thing")
        elif len(sentence.split()) > GENERICS_MAX_WORDS:
            ctx.drop("long sentence")
        elif TRAVEL.search(sentence):
            ctx.drop("travel topic")
        else:
            by_noun.setdefault(noun, []).append(sentence)
    for noun, sentences in sorted(by_noun.items()):
        sentences = list(dict.fromkeys(sentences))
        for part in chunks(shuffled(ctx.rng, sentences), ctx.rng, GENERICS_LOW, GENERICS_HIGH):
            many = plural(noun)
            header = pick(ctx.rng, ("", f"About {many}.\n", f"{many.capitalize()}:\n"))
            yield Doc(header + " ".join(part), "how")


# ---- ATOMIC 2020 -----------------------------------------------------------------------------

ATOMIC_OBJECT = {
    "ObjectUse": (
        "You can use {a} {head} to {tail}.",
        "{A} {head} is used to {tail}.",
        "People use {a} {head} to {tail}.",
    ),
    "CapableOf": ("{A} {head} can {tail}.", "{A} {head} is able to {tail}."),
}  # HasProperty is left out: its tails are often nouns ("A government office is calculator.")
ATOMIC_MIN_FACTS = 2  # an object with fewer facts makes too thin a document
ATOMIC_FACTS_PER_HEAD = 6
ATOMIC_EVENTS_LOW, ATOMIC_EVENTS_HIGH = 4, 8  # xIntent events per document
_PERSON = re.compile(r"\bPerson([XYZ])\b")


def _article(head: str, capital: bool) -> str:
    if head.endswith("s") and not head.endswith("ss"):
        return ""
    article = "an" if head[:1].lower() in "aeiou" else "a"
    return article.capitalize() if capital else article


def _object_fact(relation: str, head: str, tail: str, ctx: Context) -> str:
    template = pick(ctx.rng, ATOMIC_OBJECT[relation])
    tail = re.sub(r"^to\s+", "", tail)
    text = template.format(a=_article(head, False), A=_article(head, True), head=head, tail=tail)
    text = clean(text)
    if head.endswith("s") and not head.endswith("ss"):  # "Scissors is" -> "Scissors are"
        text = text.replace(f"{head} is ", f"{head} are ")
    return text[:1].upper() + text[1:]


def _intent_fact(head: str, tail: str, ctx: Context) -> str:
    names = {"X": ctx.name(), "Y": ctx.name(), "Z": ctx.name()}
    event = _PERSON.sub(lambda m: names[m.group(1)], head).replace("___", "something")
    tail = tail.rstrip(" .")
    return pick(
        ctx.rng,
        (
            f"{event} because {names['X']} wants {tail}.",
            f"{event}. The reason: {names['X']} wants {tail}.",
            f"{names['X']} wants {tail}, so {event}.",
        ),
    )


def atomic(raw: Path, ctx: Context) -> Iterator[Doc]:
    """ATOMIC 2020's ObjectUse and CapableOf facts about everyday things (a few per object; its
    other objects include weapons) and xIntent (why people do things) where the reason is a
    "to ..." phrase, in varied wordings."""
    things = ctx.nouns - ctx.person_nouns
    objects: dict[str, list[tuple[str, str]]] = {}
    intents: dict[str, list[str]] = {}
    for split in ("train", "dev", "test"):
        with (raw / f"atomic2020/{split}.tsv").open(newline="", encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(row) != 3:
                    continue
                head, relation, tail = (clean(x) for x in row)
                if not tail or tail.lower() == "none" or TRAVEL.search(head + " " + tail):
                    continue
                if relation in ATOMIC_OBJECT:
                    if everyday_noun(head.split()[-1], things):
                        objects.setdefault(head, []).append((relation, tail))
                    else:
                        ctx.drop("not about an everyday thing")
                elif relation == "xIntent":
                    if tail.startswith("to "):  # not "successful" or "expert help"
                        intents.setdefault(head, []).append(tail)
                    else:
                        ctx.drop("reason not a to-phrase")
    for head, facts in sorted(objects.items()):
        picked = shuffled(ctx.rng, list(dict.fromkeys(facts)))[:ATOMIC_FACTS_PER_HEAD]
        if len(picked) < ATOMIC_MIN_FACTS:
            ctx.drop("object with one fact")
            continue
        yield Doc(" ".join(_object_fact(r, head, t, ctx) for r, t in picked), "how")
    events = shuffled(ctx.rng, sorted(intents))
    for part in chunks(events, ctx.rng, ATOMIC_EVENTS_LOW, ATOMIC_EVENTS_HIGH):
        reasons = [_intent_fact(head, pick(ctx.rng, intents[head]), ctx) for head in part]
        yield Doc(" ".join(reasons), "how")


# ---- Cosmopedia ------------------------------------------------------------------------------

COSMO_MIN_NOUNS = 3  # distinct everyday nouns a text must use
COSMO_MAX_SENTENCE_WORDS = 20.0  # mean words per sentence
COSMO_MAX_LONG_WORDS = 0.12  # share of words with 9+ letters
_SENTENCE_END = re.compile(r"[.!?]+(?:\s|$)")


def readable(text: str) -> bool:
    """Short sentences and mostly short words: plain enough for a small model."""
    words = re.findall(r"[A-Za-z]+", text)
    sentences = max(1, len(_SENTENCE_END.findall(text)))
    if not words:
        return False
    long_share = sum(len(w) >= 9 for w in words) / len(words)
    return len(words) / sentences <= COSMO_MAX_SENTENCE_WORDS and long_share <= COSMO_MAX_LONG_WORDS


def _cosmopedia(paths: list[Path], audience: str | None, ctx: Context) -> Iterator[Doc]:
    for path in paths:
        table = pq.read_table(path, columns=["text", "audience"])
        for text, who in zip(*(table.column(c).to_pylist() for c in range(2)), strict=True):
            text = (text or "").strip()
            if audience and who != audience:
                ctx.drop(f"audience not {audience}")
            elif TRAVEL.search(text):
                ctx.drop("travel topic")
            elif len(everyday_nouns_in(text, ctx.nouns)) < COSMO_MIN_NOUNS:
                ctx.drop("few everyday nouns")
            elif not readable(text):
                ctx.drop("hard to read")
            else:
                yield Doc(text, "how")


def cosmopedia_stories(raw: Path, ctx: Context) -> Iterator[Doc]:
    """Cosmopedia's stories written for young children."""
    paths = sorted((raw / "cosmopedia/stories").glob("*.parquet"))
    return _cosmopedia(paths, "young_children", ctx)


def cosmopedia_wikihow(raw: Path, ctx: Context) -> Iterator[Doc]:
    """Cosmopedia's wikiHow-style articles about everyday things."""
    return _cosmopedia(sorted((raw / "cosmopedia/wikihow").glob("*.parquet")), None, ctx)
