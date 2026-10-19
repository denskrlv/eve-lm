import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from colibri.config import ModelConfig
from colibri.data.draw import pick, shuffled
from colibri.data.generate.library import ToolSpec
from colibri.data.generate.templates import MAX_CALLS, Filled, fill
from colibri.data.render import Call, TaskExample

INTENT_MIX = {"call": 0.55, "ask": 0.15, "unsupported": 0.15, "none": 0.15}
CONNECTORS = (" and ", ", then ", " and then ", ". Also ", ", and also ")
PREFIXES = ("please ", "can you ", "could you ", "hey, ", "ok, ", "I want to ")
SUFFIXES = (" please", "?", ".", " thanks", " now", "!")
PREFIX_SHARE, SUFFIX_SHARE, CAPITALIZE_SHARE = 0.25, 0.3, 0.3  # how often _decorate adds each
MAX_PROMPT_TOOLS = ModelConfig().max_tools  # a prompt offers 1 to this many tools
SAME_DOMAIN_BONUS = 0.3  # added to the similarity of two tools from one domain
SIMILARITY_FLOOR = 0.002  # every allowed distractor keeps a small chance
CALL_COUNTS, CALL_COUNT_WEIGHTS = (1, 2, 3), (0.75, 0.2, 0.05)  # calls in a CALL example
REPEAT_TOOL_SHARE = 0.4  # a later call uses the same tool again, with other values
LISTED_UNSUPPORTED_SHARE = 0.3  # UNSUPPORTED requests taken from unsupported.txt
PICK_TRIES, FILL_TRIES, ASK_TRIES = 20, 10, 50  # attempts before a sampler gives up
_QUESTION = re.compile(
    r"^(what|how|when|where|who|which|is|are|was|do|does|did|can|will|any|if)\b", re.I
)
_STOPWORDS = set(
    "the a an of or to in on and for with set get is it from by as at that this another be "
    "current new".split()
)


def _words(spec: ToolSpec) -> set[str]:
    text = spec.name.replace("_", " ") + " " + spec.tool.description
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOPWORDS}


class ToolBank:
    """The tools examples are drawn from, with how similar each pair is.

    Similarity is word overlap of names and descriptions (Jaccard), plus 0.3 for the same
    domain: a rough stand-in for what a retriever (M11) would rank close.
    """

    def __init__(self, specs: Sequence[ToolSpec]):
        self.specs = list(specs)
        self.index = {spec.name: i for i, spec in enumerate(self.specs)}
        words = [_words(spec) for spec in self.specs]
        n_specs = len(self.specs)
        self.similarity = np.zeros((n_specs, n_specs))
        for i in range(n_specs):
            for j in range(i + 1, n_specs):
                jaccard = len(words[i] & words[j]) / max(1, len(words[i] | words[j]))
                same = SAME_DOMAIN_BONUS if self.specs[i].domain == self.specs[j].domain else 0.0
                self.similarity[i, j] = self.similarity[j, i] = jaccard + same
        self.weights = np.array([spec.weight if spec.templates else 0.0 for spec in self.specs])
        self.same_capability: dict[str, list[int]] = {}
        for i, spec in enumerate(self.specs):
            self.same_capability.setdefault(spec.capability, []).append(i)

    def pick(self, rng: np.random.Generator, pool: Sequence[int] | None = None) -> int:
        """A tool to be the correct one, by weight."""
        candidates = np.arange(len(self.specs)) if pool is None else np.asarray(pool)
        weights = self.weights[candidates]
        return int(candidates[rng.choice(len(candidates), p=weights / weights.sum())])

    def distractors(
        self,
        gold: Sequence[int],
        count: int,
        rng: np.random.Generator,
        unsupported: bool = False,
        hard: bool = True,
    ) -> list[int]:
        """``k`` other tools, mostly similar ones when ``hard``.

        Never a tool with a gold tool's capability. When the request is labelled UNSUPPORTED,
        also never a tool that could serve it anyway (``overlaps``).
        """
        caps = {self.specs[g].capability for g in gold}
        names = {self.specs[g].name for g in gold}
        ok = np.array(
            [
                s.capability not in caps
                and not (unsupported and ("*" in s.overlaps or names & set(s.overlaps)))
                for s in self.specs
            ]
        )
        if hard and gold:
            weights = self.similarity[list(gold)].max(axis=0) ** 2 + SIMILARITY_FLOOR
        else:
            weights = np.ones(len(self.specs))
        weights = np.where(ok, weights, 0.0)
        chosen = []
        for _ in range(count):
            if weights.sum() == 0:
                break
            i = int(rng.choice(len(weights), p=weights / weights.sum()))
            chosen.append(i)
            weights[self.same_capability[self.specs[i].capability]] = 0
        return chosen


@dataclass
class Sample:
    example: TaskExample
    meaning: str | None  # the template text before decoration; what chat writers rephrase
    missing: list[str] = field(default_factory=list)  # ASK: the required arguments left out
    implied: set[tuple[int, str]] = field(default_factory=set)  # (call, argument) set by a template


def _decorate(text: str, rng: np.random.Generator) -> str:
    if rng.random() < PREFIX_SHARE and not _QUESTION.match(text):
        text = pick(rng, PREFIXES) + text
    if rng.random() < SUFFIX_SHARE:
        text += pick(rng, SUFFIXES)
    if rng.random() < CAPITALIZE_SHARE:
        text = text[0].upper() + text[1:]
    return text


def _fill_any(spec: ToolSpec, rng: np.random.Generator, ask: bool = False) -> Filled | None:
    """The spec's templates in random order until one fills. None only for ``ask``: without
    it every template fills, and ``ToolBank.pick`` only picks specs that have templates."""
    for template_index in rng.permutation(len(spec.templates)):
        filled = fill(spec.tool, spec.templates[template_index], spec.values, rng, ask=ask)
        if filled is not None and not (ask and filled.fanout):
            return filled
    return None


def prompt_size(rng: np.random.Generator) -> int:
    """How many tools a prompt offers: 1 to ``MAX_PROMPT_TOOLS``, uniformly."""
    return int(rng.integers(1, MAX_PROMPT_TOOLS + 1))


def _prompt(
    bank: ToolBank,
    gold: list[int],
    rng: np.random.Generator,
    unsupported: bool = False,
    hard: bool = True,
) -> list[int]:
    n_tools = max(len(gold), prompt_size(rng))
    if unsupported:
        n_tools = prompt_size(rng)
    tools = ([] if unsupported else list(gold)) + bank.distractors(
        gold, n_tools - (0 if unsupported else len(gold)), rng, unsupported, hard
    )
    return shuffled(rng, tools)


def _pick_new_capability(
    bank: ToolBank, rng: np.random.Generator, pool: Sequence[int] | None, gold: list[int]
) -> int | None:
    """A tool, by weight, whose capability no earlier call has; None after ``PICK_TRIES``."""
    caps = {bank.specs[i].capability for i in gold}
    candidates = (bank.pick(rng, pool) for _ in range(PICK_TRIES))
    return next((i for i in candidates if bank.specs[i].capability not in caps), None)


def sample_call(
    bank: ToolBank, rng: np.random.Generator, pool: Sequence[int] | None = None
) -> Sample:
    n_calls = int(rng.choice(CALL_COUNTS, p=CALL_COUNT_WEIGHTS))
    gold: list[int] = []
    parts: list[str] = []
    calls: list[Call] = []
    implied: set[tuple[int, str]] = set()
    for call_number in range(n_calls):
        if call_number > 0 and rng.random() < REPEAT_TOOL_SHARE and bank.specs[gold[-1]].tool.args:
            gold_index = gold[-1]  # the same tool again, with other values
        else:
            picked = _pick_new_capability(bank, rng, pool, gold)
            if picked is None:
                break
            gold_index = picked
        spec = bank.specs[gold_index]
        for _ in range(FILL_TRIES):
            filled = _fill_any(spec, rng)
            assert filled is not None  # see _fill_any
            new = filled.expand(spec.tool)
            repeated = any(call.tool == spec.name and call.args in new for call in calls)
            if not repeated and len(calls) + len(new) <= MAX_CALLS:
                break
        else:
            break
        gold.append(gold_index)
        parts.append(filled.text)
        for args in new:  # a fan-out ("lights off everywhere") adds one call per option
            implied |= {(len(calls), name) for name in filled.implied}
            calls.append(Call(spec.name, args))
    text = parts[0]
    for part in parts[1:]:
        text += pick(rng, CONNECTORS) + part
    distinct = list(dict.fromkeys(gold))
    tools = _prompt(bank, distinct, rng)
    example = TaskExample(
        _decorate(text, rng), tuple(bank.specs[i].tool for i in tools), "call", tuple(calls)
    )
    return Sample(example, text, implied=implied)


def sample_ask(
    bank: ToolBank, rng: np.random.Generator, pool: Sequence[int] | None = None
) -> Sample:
    for _ in range(ASK_TRIES):
        gold_index = bank.pick(rng, pool)
        spec = bank.specs[gold_index]
        filled = _fill_any(spec, rng, ask=True)
        if filled is not None:
            break
    else:
        raise RuntimeError("no tool in the pool can make an ASK example")
    tools = _prompt(bank, [gold_index], rng)
    missing = [k for k, v in filled.args.items() if v == "?"]
    example = TaskExample(
        _decorate(filled.text, rng),
        tuple(bank.specs[i].tool for i in tools),
        "ask",
        (Call(spec.name, filled.args),),
    )
    return Sample(example, filled.text, missing, {(0, name) for name in filled.implied})


def sample_unsupported(
    bank: ToolBank,
    rng: np.random.Generator,
    pool: Sequence[int] | None = None,
    requests: Sequence[str] = (),
) -> Sample:
    """A request no offered tool can do: either a real tool's request with that tool (and
    anything else that could serve it) left out, or one from a list of unsupported requests."""
    if requests and rng.random() < LISTED_UNSUPPORTED_SHARE:
        text = pick(rng, requests)
        tools = _prompt(bank, [], rng, unsupported=True, hard=False)
    else:
        gold_index = bank.pick(rng, pool)
        filled = _fill_any(bank.specs[gold_index], rng)
        assert filled is not None  # see _fill_any
        text = filled.text
        tools = _prompt(bank, [gold_index], rng, unsupported=True) or _prompt(
            bank, [gold_index], rng, unsupported=True, hard=False
        )
    example = TaskExample(
        _decorate(text, rng), tuple(bank.specs[i].tool for i in tools), "unsupported"
    )
    return Sample(example, text)


def sample_none(
    bank: ToolBank,
    rng: np.random.Generator,
    pool: Sequence[int] | None = None,
    chitchat: Sequence[str] = (),
) -> Sample:
    """Small talk with random tools in the prompt (with ``pool``, at least one from it)."""
    text = pick(rng, chitchat) if chitchat else None
    first = [] if pool is None else [bank.pick(rng, pool)]
    n_tools = prompt_size(rng) - len(first)
    tools = shuffled(rng, first + bank.distractors(first, n_tools, rng, hard=False))
    example = TaskExample(text or "thanks", tuple(bank.specs[i].tool for i in tools), "none")
    return Sample(example, None)


def sample(
    bank: ToolBank,
    rng: np.random.Generator,
    intent: str | None = None,
    pool: Sequence[int] | None = None,
    chitchat: Sequence[str] = (),
    unsupported: Sequence[str] = (),
) -> Sample:
    """One example. ``pool`` restricts which tools can be the correct one (e.g. held-out)."""
    if intent is None:
        intent = str(rng.choice(list(INTENT_MIX), p=list(INTENT_MIX.values())))
    if intent == "call":
        return sample_call(bank, rng, pool)
    if intent == "ask":
        return sample_ask(bank, rng, pool)
    if intent == "unsupported":
        return sample_unsupported(bank, rng, pool, unsupported)
    return sample_none(bank, rng, pool, chitchat)
