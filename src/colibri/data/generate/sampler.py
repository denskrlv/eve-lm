import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from colibri.data.generate.library import ToolSpec
from colibri.data.generate.templates import MAX_CALLS, fill
from colibri.data.render import Call, TaskExample

INTENT_MIX = {"call": 0.55, "ask": 0.15, "unsupported": 0.15, "none": 0.15}
CONNECTORS = (" and ", ", then ", " and then ", ". Also ", ", and also ")
PREFIXES = ("please ", "can you ", "could you ", "hey, ", "ok, ", "I want to ")
SUFFIXES = (" please", "?", ".", " thanks", " now", "!")
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
        self.index = {s.name: i for i, s in enumerate(self.specs)}
        words = [_words(s) for s in self.specs]
        n = len(self.specs)
        self.sim = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                jaccard = len(words[i] & words[j]) / max(1, len(words[i] | words[j]))
                same = 0.3 if self.specs[i].domain == self.specs[j].domain else 0.0
                self.sim[i, j] = self.sim[j, i] = jaccard + same
        self.weights = np.array([s.weight if s.templates else 0.0 for s in self.specs])
        self.same_capability: dict[str, list[int]] = {}
        for i, s in enumerate(self.specs):
            self.same_capability.setdefault(s.capability, []).append(i)

    def pick(self, rng: np.random.Generator, pool: Sequence[int] | None = None) -> int:
        """A tool to be the correct one, by weight."""
        pool = np.arange(len(self.specs)) if pool is None else np.asarray(pool)
        w = self.weights[pool]
        return int(pool[rng.choice(len(pool), p=w / w.sum())])

    def distractors(
        self,
        gold: Sequence[int],
        k: int,
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
            weights = self.sim[list(gold)].max(axis=0) ** 2 + 0.002
        else:
            weights = np.ones(len(self.specs))
        weights = np.where(ok, weights, 0.0)
        chosen = []
        for _ in range(k):
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
    if rng.random() < 0.25 and not _QUESTION.match(text):
        text = PREFIXES[rng.integers(len(PREFIXES))] + text
    if rng.random() < 0.3:
        text += SUFFIXES[rng.integers(len(SUFFIXES))]
    if rng.random() < 0.3:
        text = text[0].upper() + text[1:]
    return text


def _fill_any(spec: ToolSpec, rng: np.random.Generator, ask: bool = False):
    for t in rng.permutation(len(spec.templates)):
        filled = fill(spec.tool, spec.templates[t], spec.values, rng, ask=ask)
        if filled is not None and not (ask and filled.fanout):
            return filled
    return None


def _prompt(bank: ToolBank, gold: list[int], rng, unsupported=False, hard=True) -> list[int]:
    k = max(len(gold), int(rng.integers(1, 6)))
    if unsupported:
        k = int(rng.integers(1, 6))
    tools = ([] if unsupported else list(gold)) + bank.distractors(
        gold, k - (0 if unsupported else len(gold)), rng, unsupported, hard
    )
    return [tools[i] for i in rng.permutation(len(tools))]


def sample_call(bank: ToolBank, rng: np.random.Generator, pool=None) -> Sample:
    n_calls = int(rng.choice([1, 2, 3], p=[0.75, 0.2, 0.05]))
    gold: list[int] = []
    parts, calls, implied = [], [], set()
    for c in range(n_calls):
        if c > 0 and rng.random() < 0.4 and bank.specs[gold[-1]].tool.args:
            g = gold[-1]  # the same tool again, with other values
        else:
            caps = {bank.specs[i].capability for i in gold}
            g = next(
                (
                    i
                    for i in (bank.pick(rng, pool) for _ in range(20))
                    if bank.specs[i].capability not in caps
                ),
                None,
            )
            if g is None:
                break
        spec = bank.specs[g]
        for _ in range(10):
            filled = _fill_any(spec, rng)
            new = filled.expand(spec.tool)
            repeated = any(c.tool == spec.name and c.args in new for c in calls)
            if not repeated and len(calls) + len(new) <= MAX_CALLS:
                break
        else:
            break
        gold.append(g)
        parts.append(filled.text)
        for args in new:  # a fan-out ("lights off everywhere") adds one call per option
            implied |= {(len(calls), name) for name in filled.implied}
            calls.append(Call(spec.name, args))
    text = parts[0]
    for part in parts[1:]:
        text += CONNECTORS[rng.integers(len(CONNECTORS))] + part
    distinct = list(dict.fromkeys(gold))
    tools = _prompt(bank, distinct, rng)
    example = TaskExample(
        _decorate(text, rng), tuple(bank.specs[i].tool for i in tools), "call", tuple(calls)
    )
    return Sample(example, text, implied=implied)


def sample_ask(bank: ToolBank, rng: np.random.Generator, pool=None) -> Sample:
    for _ in range(50):
        g = bank.pick(rng, pool)
        spec = bank.specs[g]
        filled = _fill_any(spec, rng, ask=True)
        if filled is not None:
            break
    else:
        raise RuntimeError("no tool in the pool can make an ASK example")
    tools = _prompt(bank, [g], rng)
    missing = [k for k, v in filled.args.items() if v == "?"]
    example = TaskExample(
        _decorate(filled.text, rng),
        tuple(bank.specs[i].tool for i in tools),
        "ask",
        (Call(spec.name, filled.args),),
    )
    return Sample(example, filled.text, missing, {(0, name) for name in filled.implied})


def sample_unsupported(
    bank: ToolBank, rng: np.random.Generator, pool=None, requests: Sequence[str] = ()
) -> Sample:
    """A request no offered tool can do: either a real tool's request with that tool (and
    anything else that could serve it) left out, or one from a list of unsupported requests."""
    if requests and rng.random() < 0.3:
        text = requests[rng.integers(len(requests))]
        tools = _prompt(bank, [], rng, unsupported=True, hard=False)
    else:
        g = bank.pick(rng, pool)
        text = _fill_any(bank.specs[g], rng).text
        tools = _prompt(bank, [g], rng, unsupported=True) or _prompt(
            bank, [g], rng, unsupported=True, hard=False
        )
    example = TaskExample(
        _decorate(text, rng), tuple(bank.specs[i].tool for i in tools), "unsupported"
    )
    return Sample(example, text)


def sample_none(
    bank: ToolBank, rng: np.random.Generator, pool=None, chitchat: Sequence[str] = ()
) -> Sample:
    """Small talk with random tools in the prompt (with ``pool``, at least one from it)."""
    text = chitchat[rng.integers(len(chitchat))] if chitchat else None
    first = [] if pool is None else [bank.pick(rng, pool)]
    k = int(rng.integers(1, 6)) - len(first)
    tools = first + bank.distractors(first, k, rng, hard=False)
    tools = [tools[i] for i in rng.permutation(len(tools))]
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
