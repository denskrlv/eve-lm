"""TinyTools: a TinyStories-style pretraining corpus for a small tool-calling model.

TinyStories (Eldan & Li, 2023) taught models under 10M parameters to write fluent English by
pretraining them on LLM-written stories in a small child's vocabulary. Each prompt named three
random words and a random feature the story had to use, so two million stories didn't repeat
each other. TinyTools does the same for Eve's narrow domain: what people ask assistants for and
what tools do, in everyday words, so a ~30M model can learn that "20 plus 10" and "Add two
numbers" mean the same thing before it is fine-tuned.

``make_prompts`` draws the prompts from ``data/tinytools`` (word lists, domains); a generator
writes one document per prompt (see ``data/tinytools/README.md``); ``check_corpus`` keeps the
documents that follow their prompt and writes them out for ``eve corpus encode``.
"""

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from eve.data.generate.validate import normalize
from eve.data.schema import Tool
from eve.tokenizer.bpe import SPECIAL_TOKENS

KINDS = {"scene": 0.40, "explain": 0.20, "ways": 0.15, "card": 0.15, "dialogue": 0.10}
WORDS = {  # (fewest, most) words a document of each kind may have; prompts ask for less
    "scene": (50, 300),
    "explain": (40, 250),
    "ways": (25, 250),
    "card": (30, 250),
    "dialogue": (25, 300),
}
FEATURES = (  # one per scene, "ways" list and dialogue, as TinyStories added "has dialogue"
    "a number written in words, like twenty-five",
    'an indirect request, like "it\'s freezing in here" instead of "turn up the heat"',
    "a correction: the person changes their mind or fixes a detail",
    "a negation: the person says what they do not want",
    'a time or date phrase, like "tomorrow at seven"',
    "two requests in one sentence",
    "a request typed in a hurry, in lowercase with one small typo",
    "a very short request of two or three words",
    "a long, polite request that gives a reason",
    "an amount or comparison, like more, less, half or double",
)
SPEAKERS = (  # who asks, in scenes and dialogues
    "a child",
    "a grandparent",
    "a busy parent",
    "a teenager texting",
    "someone in a hurry",
    "a cook",
    "a student",
    "a nurse after a night shift",
    "a gardener",
    "a driver",
    "someone who just woke up",
    "a shop owner",
    "someone new to the city",
    "an office worker",
    "a retired teacher",
    "a person who is not good with technology",
)
CONCEPT_ONLY = "concept-only"  # a domains.txt flag: no tool cards for this domain
FILES = {"nouns": "nouns.txt", "verbs": "verbs.txt", "adjectives": "adjectives.txt"}
PREAMBLE = re.compile(r"^\W*(sure|certainly|of course|here is|here's|okay, here)\b", re.I)
SHINGLE, DUPLICATE = 5, 0.5  # near-duplicates share half their 5-word shingles (Jaccard)


@dataclass(frozen=True)
class Domain:
    """A topic a document is about: ``lights``, ``arithmetic``."""

    name: str
    about: str  # what the topic covers, for the prompt
    concept_only: bool = False  # a held-out tool's domain: no tool cards


@dataclass(frozen=True)
class Lexicon:
    """The words and domains prompts are drawn from. A word maps to its irregular forms."""

    nouns: dict[str, tuple[str, ...]]
    verbs: dict[str, tuple[str, ...]]
    adjectives: dict[str, tuple[str, ...]]
    domains: tuple[Domain, ...]

    @classmethod
    def load(cls, directory: str | Path) -> "Lexicon":
        """``nouns.txt``, ``verbs.txt``, ``adjectives.txt`` and ``domains.txt`` in ``directory``."""
        directory = Path(directory)
        words = {kind: _read_words(directory / name) for kind, name in FILES.items()}
        return cls(**words, domains=_read_domains(directory / "domains.txt"))


def _lines(path: Path) -> list[str]:
    """The lines of a word file, without comments and blank lines."""
    lines = (line.strip() for line in path.read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def _read_words(path: Path) -> dict[str, tuple[str, ...]]:
    """``buy bought`` -> {"buy": ("bought",)}; a word listed twice counts once."""
    out: dict[str, tuple[str, ...]] = {}
    for line in _lines(path):
        word, *forms = line.lower().split()
        out.setdefault(word, tuple(forms))
    return out


def _read_domains(path: Path) -> tuple[Domain, ...]:
    """``name | about`` or ``name | about | concept-only`` per line."""
    domains = []
    for line in _lines(path):
        name, about, *flags = (part.strip() for part in line.split("|"))
        unknown = set(flags) - {CONCEPT_ONLY}
        if unknown:
            raise ValueError(f"{path}: unknown flags {sorted(unknown)} for {name!r}")
        domains.append(Domain(name, about, CONCEPT_ONLY in flags))
    return tuple(domains)


@dataclass(frozen=True)
class PromptSpec:
    """One document to write: its kind, topic, the three words it must use, and the prompt."""

    id: str
    kind: str
    domain: str
    noun: str
    verb: str
    adjective: str
    feature: str  # "" for kinds without one
    speaker: str  # "" for kinds without one
    prompt: str

    def to_json(self) -> dict[str, str]:
        """The spec as plain values, one line of ``prompts.jsonl``."""
        return asdict(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "PromptSpec":
        """A spec from a line of ``prompts.jsonl``."""
        return cls(**{key: str(data[key]) for key in cls.__dataclass_fields__})


def make_prompts(lexicon: Lexicon, n: int, seed: int = 0, prefix: str = "tt") -> list[PromptSpec]:
    """``n`` prompts. Prompt ``i`` depends only on ``(seed, i)``, so a run can be extended
    without changing the prompts it already has. Tool cards never use a concept-only domain."""
    kinds, weights = list(KINDS), list(KINDS.values())
    carded = [d for d in lexicon.domains if not d.concept_only]
    nouns, verbs, adjectives = (list(words) for words in _parts(lexicon))
    out = []
    for i in range(n):
        rng = np.random.default_rng((seed, i))
        kind = str(rng.choice(kinds, p=weights))
        pool = carded if kind == "card" else lexicon.domains
        domain = pool[int(rng.integers(len(pool)))]
        noun, verb, adjective = (str(rng.choice(words)) for words in (nouns, verbs, adjectives))
        feature = str(rng.choice(FEATURES)) if kind in ("scene", "ways", "dialogue") else ""
        speaker = str(rng.choice(SPEAKERS)) if kind in ("scene", "dialogue") else ""
        words = (noun, verb, adjective)
        prompt = render_prompt(kind, domain, words, feature, speaker)
        out.append(
            PromptSpec(f"{prefix}-{i:06d}", kind, domain.name, *words, feature, speaker, prompt)
        )
    return out


def _parts(lexicon: Lexicon) -> tuple[dict[str, tuple[str, ...]], ...]:
    return lexicon.nouns, lexicon.verbs, lexicon.adjectives


def render_prompt(
    kind: str, domain: Domain, words: tuple[str, str, str], feature: str, speaker: str
) -> str:
    """The text a generator gets for one document."""
    noun, verb, adjective = words
    topic = f"{domain.name} ({domain.about})"
    if kind == "scene":
        task = (
            f"Write a short story (80 to 200 words) about {speaker} who asks a voice assistant, "
            f"phone or smart device for help with something about {topic}. The person asks in "
            "their own words, and the story says plainly what the assistant understood and did."
        )
    elif kind == "explain":
        task = (
            f"Write a short explanation (60 to 150 words) of how something about {topic} works in "
            "everyday life: what it is for, what people change about it, and the words they use "
            "when they ask for those changes."
        )
    elif kind == "ways":
        task = (
            f"Write a list of different ways people ask for one thing about {topic}. First write "
            "one line saying plainly what they want. Then write 6 to 10 ways to ask for it, one "
            "per line, from short and casual to long and polite. Most ways should not repeat the "
            "words of the first line."
        )
    elif kind == "card":
        task = (
            f"Invent one tool an assistant could have for {topic}. Use exactly this format:\n"
            "name: the tool's name in snake_case\n"
            "does: one sentence saying what it does\n"
            "inputs: what it needs, in plain words\n"
            "people say:\n"
            "- then 4 to 6 things people say when they need it, one per line, mostly in words "
            "different from the tool's name"
        )
    elif kind == "dialogue":
        task = (
            f"Write a short conversation (4 to 10 turns) between {speaker} and an assistant about "
            f"{topic}. The person asks for something but leaves out a detail the assistant "
            "needs, and the assistant asks a short question to get it; or the person asks for "
            "something the assistant can't do, and it says so politely. Start each turn on a new "
            'line with "Person:" or "Assistant:".'
        )
    else:
        raise ValueError(f"unknown kind {kind!r}; choose from {sorted(KINDS)}")
    lines = [
        task,
        f'Use the noun "{noun}", the verb "{verb}" and the adjective "{adjective}" somewhere '
        "(any form of them is fine, like a plural or the past tense).",
    ]
    if feature:
        lines.append(f"Include {feature}.")
    lines.append(
        "Use simple, everyday English. Write only the text itself: no title, no notes, and no "
        "angle brackets."
    )
    return "\n".join(lines)


# ---- checking ---------------------------------------------------------------------------------


def word_forms(word: str, irregular: Sequence[str] = ()) -> set[str]:
    """The forms of ``word`` a document may use: plurals, -ed, -ing, -er, -est, -ly, and the
    listed irregular ones. Some aren't English ("childs"); they only make the check lenient."""
    forms = {word, *irregular, word + "s", word + "es", word + "ed", word + "d", word + "ing"}
    forms |= {word + "er", word + "est", word + "r", word + "st", word + "ly"}
    if word.endswith("e"):
        forms |= {word[:-1] + "ing", word[:-1] + "y"}
    if word.endswith("y"):
        stem = word[:-1]
        forms |= {stem + "ies", stem + "ied", stem + "ier", stem + "iest", stem + "ily"}
    if re.search(r"[^aeiou][aeiou][bdgklmnprt]$", word):  # stop -> stopped, big -> bigger
        double = word + word[-1]
        forms |= {double + "ed", double + "ing", double + "er", double + "est"}
    if word.endswith("f"):
        forms.add(word[:-1] + "ves")
    if word.endswith("fe"):
        forms.add(word[:-2] + "ves")
    return forms


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def _structure_errors(kind: str, text: str) -> list[str]:
    """What a document of ``kind`` must look like beyond its words."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if kind == "dialogue":
        turns = [line for line in lines if re.match(r"(person|assistant)\s*:", line, re.I)]
        if len(turns) < 4:
            return [
                f"a dialogue needs 4+ turns starting with Person: or Assistant:, has {len(turns)}"
            ]
    elif kind == "card":
        errors = []
        if not any(re.fullmatch(r"name:\s*[a-z][a-z0-9]*(_[a-z0-9]+)*", line) for line in lines):
            errors.append("a tool card needs a line 'name: snake_case_name'")
        if not any(line.lower().startswith("does:") for line in lines):
            errors.append("a tool card needs a 'does:' line")
        if sum(line.startswith("- ") for line in lines) < 3:
            errors.append("a tool card needs 3+ requests under 'people say:', each starting '- '")
        return errors
    elif kind == "ways" and len(lines) < 6:
        return [f"a 'ways' list needs 6+ lines, has {len(lines)}"]
    return []


def _leaks(text: str, kind: str, heldout: Sequence[Tool]) -> list[str]:
    """Held-out tools named as tools: a snake_case name anywhere (``convert_units``), a short
    name used like a function (``add(`` or ``name: add``), or, in a tool card, a description
    word for word. In prose "add two numbers" is just English, so only cards are checked."""
    plain = normalize(text)
    out = []
    for tool in heldout:
        name = re.escape(tool.name)
        named = rf"(?<![\w]){name}(?![\w])" if "_" in tool.name else rf"(?<![\w]){name}\s*\("
        if re.search(named, text, re.I) or re.search(rf"name:\s*{name}\s*$", text, re.I | re.M):
            out.append(f"names the held-out tool {tool.name}")
        elif kind == "card" and normalize(tool.description) in plain:
            out.append(f"copies the description of the held-out tool {tool.name}")
    return out


def check_document(
    spec: PromptSpec, text: str, lexicon: Lexicon, heldout: Sequence[Tool] = ()
) -> list[str]:
    """Why ``text`` doesn't follow its prompt (empty: it does). Near-duplicates are found
    across the corpus, in ``check_corpus``."""
    errors = []
    tokens = _tokens(text)
    low, high = WORDS[spec.kind]
    if not low <= len(tokens) <= high:
        errors.append(f"{len(tokens)} words; a {spec.kind} needs {low} to {high}")
    present = set(tokens)
    for word, forms in zip((spec.noun, spec.verb, spec.adjective), _parts(lexicon), strict=True):
        if not word_forms(word, forms.get(word, ())) & present:
            errors.append(f"doesn't use {word!r}")
    if found := [t for t in SPECIAL_TOKENS if t in text]:
        errors.append(f"contains the special token {found[0]}")
    if PREAMBLE.match(text):
        errors.append("starts with a reply to the prompt instead of the text")
    errors += _structure_errors(spec.kind, text)
    errors += _leaks(text, spec.kind, heldout)
    return errors


def _shingles(text: str) -> set[int]:
    words = _tokens(text)
    return {hash(tuple(words[i : i + SHINGLE])) for i in range(max(1, len(words) - SHINGLE + 1))}


def near_duplicates(texts: Mapping[str, str]) -> dict[str, str]:
    """For each document that repeats an earlier one (Jaccard of 5-word shingles at least
    ``DUPLICATE``), the id of the earlier one. Earlier = first in ``texts``."""
    index: dict[int, list[str]] = {}
    sizes: dict[str, int] = {}
    out = {}
    for doc_id, text in texts.items():
        shingles = _shingles(text)
        shared: Counter[str] = Counter()
        for shingle in shingles:
            shared.update(index.get(shingle, ()))
        for other, n in shared.most_common():
            if n / (len(shingles) + sizes[other] - n) >= DUPLICATE:
                out[doc_id] = other
                break
        if doc_id not in out:  # only originals are compared against
            sizes[doc_id] = len(shingles)
            for shingle in shingles:
                index.setdefault(shingle, []).append(doc_id)
    return out


@dataclass
class CorpusReport:
    """What ``check_corpus`` kept and why it rejected the rest."""

    accepted: list[tuple[PromptSpec, str]] = field(default_factory=list)
    rejected: dict[str, list[str]] = field(default_factory=dict)  # id -> reasons
    missing: list[str] = field(default_factory=list)  # prompts without a document
    unknown: list[str] = field(default_factory=list)  # documents without a prompt

    def stats(self, lexicon: Lexicon) -> dict[str, Any]:
        """Counts per kind, reject reasons, length, and how much of the vocabulary and the
        domains the accepted documents cover."""
        kinds = Counter(spec.kind for spec, _ in self.accepted)
        reasons = Counter(_reason(r) for rs in self.rejected.values() for r in rs)
        tokens = [set(_tokens(text)) for _, text in self.accepted]
        used = set().union(*tokens) if tokens else set()
        words = {w: f for part in _parts(lexicon) for w, f in part.items()}
        covered = sum(bool(word_forms(w, f) & used) for w, f in words.items())
        n_words = [len(_tokens(text)) for _, text in self.accepted]
        return {
            "prompts": len(self.accepted) + len(self.rejected) + len(self.missing),
            "accepted": len(self.accepted),
            "rejected": len(self.rejected),
            "missing": len(self.missing),
            "unknown": len(self.unknown),
            "accepted_by_kind": dict(kinds),
            "reject_reasons": dict(reasons.most_common()),
            "words": sum(n_words),
            "mean_words": sum(n_words) / len(n_words) if n_words else 0.0,
            "vocabulary_covered": covered / len(words) if words else 0.0,
            "domains_covered": len({spec.domain for spec, _ in self.accepted}),
        }


def _reason(reason: str) -> str:
    """A reason without its specifics, for counting: "doesn't use 'lamp'" -> "doesn't use"."""
    reason = re.sub(r"'[^'\s]*'|\b\d+\b", "…", reason)  # a quoted word, not "doesn't"
    return re.sub(r"(held-out tool) \w+", r"\1 …", reason)


def check_corpus(
    specs: Sequence[PromptSpec],
    documents: Mapping[str, str],
    lexicon: Lexicon,
    heldout: Sequence[Tool] = (),
) -> CorpusReport:
    """Check every document against its prompt, in prompt order, and drop near-duplicates of
    an earlier accepted document."""
    report = CorpusReport()
    known = {spec.id for spec in specs}
    report.unknown = [doc_id for doc_id in documents if doc_id not in known]
    passed = {}
    for spec in specs:
        text = documents.get(spec.id)
        if text is None:
            report.missing.append(spec.id)
        elif errors := check_document(spec, text, lexicon, heldout):
            report.rejected[spec.id] = errors
        else:
            passed[spec.id] = (spec, text.strip())
    repeats = near_duplicates({doc_id: text for doc_id, (_, text) in passed.items()})
    for doc_id, (spec, text) in passed.items():
        if doc_id in repeats:
            report.rejected[doc_id] = [f"nearly repeats {repeats[doc_id]}"]
        else:
            report.accepted.append((spec, text))
    return report


def write_parquet(accepted: Sequence[tuple[PromptSpec, str]], path: str | Path) -> None:
    """The accepted documents as parquet, with the ``text`` column ``eve corpus encode`` reads
    (plus ``id``, ``kind`` and ``domain``)."""
    columns = {
        "text": [text for _, text in accepted],
        "id": [spec.id for spec, _ in accepted],
        "kind": [spec.kind for spec, _ in accepted],
        "domain": [spec.domain for spec, _ in accepted],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)
