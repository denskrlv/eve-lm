"""Retrieval (M11): how often a scorer puts every tool a request needs in the top 5 of a
registry, and the components that train a ``Retriever`` to do it.

Eve reads 5 tools per request, so a tool the scorer ranks 6th is never called. The test: for
each CALL or ASK example of a task split, a registry of ``size`` tools made of the example's
own (its correct tools and the similar distractors the generator picked), filled up with
random tools from the rest of the split, in a random order. Random fillers mostly come from other
domains, which is the easy case: a calculator's ``add`` among lights and timers. ``similar``
fillers are the hard case, ``add`` among other arithmetic: tools of a correct tool's domain
first, then tools whose arguments have the same types, then random ones. The scores
(``measure``):

- ``all@5``: every correct tool is in the top 5, so Eve can answer the whole request;
- ``recall@5``: the share of correct tools in the top 5;
- ``mrr``: 1 / the rank of the best-ranked correct tool, averaged.

Training (``eve run configs/experiments/retriever-base.toml``): each request of a CALL or ASK
example is paired with each of its correct tools. The loss is InfoNCE: the request's cosine
with its tool, against every other tool in the batch, the first ``hard_negatives`` of the
example's own distractors among them (hard negatives: the generator picked them to look
alike). A tool that also serves the
request is never its negative: its other correct tools, any tool of the same name, or of the
same capability (``ToolSpec.capability``).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from eve import defaults
from eve.api.toolbox import Dense, Scorer
from eve.components import Batch, DataModule
from eve.config import ModelConfig
from eve.data.modules import load_examples
from eve.data.render import TaskExample
from eve.data.schema import Tool
from eve.model.retriever import Retriever, RetrieverConfig, padded, query_ids, tool_ids
from eve.registry import register
from eve.tokenizer.bpe import BOS, EOS, TOOLS, BPETokenizer
from eve.train.loop import Column, Report

TASK = "retrieval"  # meta["task"] of the checkpoints a retriever run writes
EVAL_SIZE = 25  # the registry size the training report scores
FILLER = [BOS, TOOLS, EOS]  # pads a batch's tools to its full count: no one's tool

TOP = 5  # the tools one request shows Eve (ModelConfig.max_tools)
SIZES = (10, 25, 50)  # registry sizes the eval reports
SPLITS = {  # split -> its file in the task directory
    "val": "val.jsonl",
    "heldout": "heldout-templates.jsonl",  # tools never trained on: the generalization test
    "massive": "massive-test.jsonl",
}
WHOLE = {"massive": None}  # splits whose registry is every tool in them (47), not ``SIZES``
FILLS = ("random", "similar")  # how a registry is filled up (see the module)
RETRIEVED = ("call", "ask")  # the intents that name tools


@dataclass(frozen=True)
class Case:
    """One request, the tools it needs and the registry they are ranked in."""

    text: str
    gold: frozenset[str]
    tools: tuple[Tool, ...]


def pool_of(examples: Sequence[TaskExample]) -> dict[str, Tool]:
    """Every tool of ``examples`` by name; a name seen twice keeps its first schema."""
    pool: dict[str, Tool] = {}
    for example in examples:
        for tool in example.tools:
            pool.setdefault(tool.name, tool)
    return pool


def signature(tool: Tool) -> tuple[str, ...]:
    """The types of a tool's arguments: ``add(a: float, b: float)`` is ``("float", "float")``."""
    return tuple(sorted(arg.type for arg in tool.args))


def cases(
    examples: Sequence[TaskExample], size: int | None, seed: int = 0, fill: str = "random"
) -> list[Case]:
    """A ``Case`` per CALL or ASK example that names tools, its registry ``size`` tools
    (``None``: all of the split's) filled up the ``fill`` way (see the module), in a random
    order so ties don't favour the correct tools. Seeded: the same registries every time."""
    if fill not in FILLS:
        raise ValueError(f"fill must be one of {list(FILLS)}, got {fill!r}")
    rng = np.random.default_rng(seed)
    pool = pool_of(examples)
    names = list(pool)
    out = []
    for example in examples:
        if example.intent not in RETRIEVED or not example.calls:  # an ASK may name no tool
            continue
        gold = frozenset(call.tool for call in example.calls)
        own = sorted(example.tools, key=lambda t: t.name not in gold)  # correct tools first
        chosen = {t.name: t for t in (own if size is None else own[: max(size, len(gold))])}
        wanted = len(names) if size is None else size
        order = [names[i] for i in rng.permutation(len(names))]
        if fill == "similar":
            correct = [t for t in own if t.name in gold]
            domains = {t.domain for t in correct} - {""}
            shapes = {signature(t) for t in correct}
            order.sort(
                key=lambda n: (
                    0 if pool[n].domain in domains else 1 + (signature(pool[n]) not in shapes)
                )
            )  # stable: random within each tier
        for name in order:
            if len(chosen) >= wanted:
                break
            chosen.setdefault(name, pool[name])
        tools = list(chosen.values())
        out.append(
            Case(example.utterance, gold, tuple(tools[i] for i in rng.permutation(len(tools))))
        )
    return out


def ranked(scorer: Scorer, case: Case) -> list[str]:
    """The registry's names, best first; ties keep the registry's order."""
    scores = scorer.scores(case.text, case.tools)
    return [case.tools[i].name for i in sorted(range(len(scores)), key=lambda i: -scores[i])]


def measure(scorer: Scorer, cases: Sequence[Case], top: int = TOP) -> dict[str, float]:
    """``all@5``, ``recall@5`` and ``mrr`` of ``scorer`` on ``cases`` (see the module)."""
    if prime := getattr(scorer, "prime", None):  # a dense scorer embeds everything in batches
        prime([c.text for c in cases], [t for c in cases for t in c.tools])
    every = found = reciprocal = 0.0
    for case in cases:
        names = ranked(scorer, case)
        hits = case.gold & set(names[:top])
        every += hits == case.gold
        found += len(hits) / len(case.gold)
        reciprocal += 1 / (1 + min(names.index(name) for name in case.gold))
    n = max(1, len(cases))
    return {f"all@{top}": every / n, f"recall@{top}": found / n, "mrr": reciprocal / n}


def load_splits(
    library: dict[str, Tool],
    task_dir: str | Path = defaults.TASK_DIR,
    splits: Sequence[str] = tuple(SPLITS),
) -> dict[str, list[TaskExample]]:
    """The examples of each split, with tools looked up in ``library``."""
    return {name: load_examples(Path(task_dir) / SPLITS[name], library) for name in splits}


def suite(
    examples: dict[str, list[TaskExample]],
    sizes: Sequence[int] = SIZES,
    seed: int = 0,
    fills: Sequence[str] = ("random",),
) -> dict[str, list[Case]]:
    """The cases of every split, registry size and fill, keyed ``"val@25"``, ``"val@25
    similar"`` for the ``similar`` fill, and ``"massive@all"`` for a split in ``WHOLE`` (every
    tool: no fill to choose)."""
    out = {}
    for name, split in examples.items():
        if name in WHOLE:
            out[f"{name}@all"] = cases(split, None, seed)
            continue
        for fill in fills:
            for size in sizes:
                key = f"{name}@{size}" + ("" if fill == "random" else f" {fill}")
                out[key] = cases(split, size, seed, fill)
    return out


def evaluate_scorers(
    scorers: dict[str, Scorer], suites: dict[str, list[Case]], top: int = TOP
) -> dict[str, dict[str, dict[str, float]]]:
    """scorer -> registry (``"val@25"``) -> its scores."""
    return {
        name: {key: measure(scorer, cases_, top) for key, cases_ in suites.items()}
        for name, scorer in scorers.items()
    }


def results_rows(results: dict[str, dict[str, dict[str, float]]], metric: str) -> list[list[Any]]:
    """One row per scorer: its name, then ``metric`` for each registry."""
    return [
        [name, *(scores[metric] for scores in by_suite.values())]
        for name, by_suite in results.items()
    ]


# ---- training ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    """A request and one tool that serves it, as indices into ``RetrievalData.tools``."""

    query: list[int]  # the request's token ids
    positive: int
    negatives: tuple[int, ...]  # the example's distractors
    related: frozenset[int]  # its other correct tools: never its negatives


@dataclass(frozen=True)
class RetrievalData:
    """Request-tool pairs of the train and val splits, every tool's token ids, and the
    registries the report scores. Batches have ``q_ids``, ``q_mask`` (B, n): the requests;
    ``t_ids``, ``t_mask`` (T, m): every tool of the batch once, then fillers; ``positive``
    (B,): each request's tool among them; ``allowed`` (B, T): False for tools that also serve
    it, and for the fillers.

    Every batch of B requests has one shape: requests padded to ``query_len``, tools to
    ``tool_len``, and T = B * (1 + ``hard``) rows. On MPS, the allocator keeps a cache for
    every shape it has seen and rarely gives it back, so varying shapes grow memory without
    bound."""

    train: list[Pair]
    val: list[Pair]
    tools: list[list[int]] = field(repr=False)
    groups: list[str] = field(repr=False)  # per tool: its capability, or its name
    suites: dict[str, list[Case]] = field(repr=False)
    hard: int = 1  # how many of each example's distractors join the batch
    query_len: int = 128
    tool_len: int = 128

    def collate(self, items: Sequence[Pair]) -> Batch:
        """Pad the requests and the batch's tools, and mark which tools may be negatives."""
        columns: dict[int, int] = {}
        for pair in items:
            for tool in (pair.positive, *pair.negatives[: self.hard]):
                columns.setdefault(tool, len(columns))
        queries = [pair.query for pair in items]
        tools = [self.tools[tool] for tool in columns]
        tools += [FILLER] * (len(items) * (1 + self.hard) - len(tools))
        q_ids, q_mask = padded(queries, self.query_len)
        t_ids, t_mask = padded(tools, self.tool_len)
        allowed = torch.zeros(len(items), len(tools), dtype=torch.bool)
        allowed[:, : len(columns)] = True
        for row, pair in enumerate(items):
            group = self.groups[pair.positive]
            for tool, column in columns.items():
                if tool != pair.positive and (tool in pair.related or self.groups[tool] == group):
                    allowed[row, column] = False
        positive = torch.tensor([columns[pair.positive] for pair in items])
        return {
            "q_ids": q_ids,
            "q_mask": q_mask,
            "t_ids": t_ids,
            "t_mask": t_mask,
            "positive": positive,
            "allowed": allowed,
        }

    def batch_tokens(self, batch: Batch) -> int:
        """The real request and tool tokens of a batch."""
        return int(batch["q_mask"].sum() + batch["t_mask"].sum())


class _Table:
    """Each distinct tool once, by schema, with its token ids."""

    def __init__(self, tokenizer: BPETokenizer, cfg: ModelConfig) -> None:
        lengths = cfg if isinstance(cfg, RetrieverConfig) else RetrieverConfig(**cfg.to_dict())
        self.tokenizer = tokenizer
        self.query_len, self.tool_len = lengths.query_len, lengths.tool_len
        self.index: dict[Tool, int] = {}
        self.ids: list[list[int]] = []

    def __call__(self, tool: Tool) -> int:
        if tool not in self.index:
            self.index[tool] = len(self.ids)
            self.ids.append(tool_ids(self.tokenizer, tool, self.tool_len))
        return self.index[tool]


def pairs(examples: Sequence[TaskExample], table: _Table) -> list[Pair]:
    """A ``Pair`` per correct tool of each CALL or ASK example."""
    out = []
    for example in examples:
        if example.intent not in RETRIEVED:
            continue
        by_name = {tool.name: tool for tool in example.tools}
        gold = {table(by_name[call.tool]) for call in example.calls}
        negatives = tuple(
            table(t) for t in example.tools if t.name not in {c.tool for c in example.calls}
        )
        query = query_ids(table.tokenizer, example.utterance, table.query_len)
        out += [Pair(query, tool, negatives, frozenset(gold - {tool})) for tool in sorted(gold)]
    return out


@register("data", "retrieval")
def retrieval(
    tokenizer: BPETokenizer,
    model_cfg: ModelConfig,
    seed: int,
    path: str = defaults.TASK_DIR,
    hard_negatives: int = 1,
    eval_size: int = EVAL_SIZE,
    library: dict[str, Tool] | None = None,
) -> RetrievalData:
    """``path``'s ``train.jsonl`` and ``val.jsonl`` as request-tool pairs, and the report's
    registries of ``eval_size`` tools from the val, held-out and MASSIVE files it has
    (``suite``). Each request brings ``hard_negatives`` of its example's distractors into its
    batch (0 to 4)."""
    del seed  # the batch order is the trainer's
    from eve.data.generate.build import Sources

    sources = Sources.load()
    library = sources.library if library is None else library
    capability = {spec.name: spec.capability for spec in sources.train + sources.heldout}
    task_dir = Path(path)
    table = _Table(tokenizer, model_cfg)
    train = pairs(load_examples(task_dir / "train.jsonl", library), table)
    val = pairs(load_examples(task_dir / "val.jsonl", library), table)
    groups = [""] * len(table.ids)
    for tool, i in table.index.items():
        groups[i] = capability.get(tool.name, tool.name)
    there = [name for name, file in SPLITS.items() if (task_dir / file).is_file()]
    suites = suite(load_splits(library, task_dir, there), (eval_size,))
    return RetrievalData(
        train, val, table.ids, groups, suites, hard_negatives, table.query_len, table.tool_len
    )


def _logits(model: Retriever, batch: Batch) -> torch.Tensor:
    """(B, T) scaled cosines of each request with each tool, -inf where not a negative."""
    queries = model(batch["q_ids"], batch["q_mask"])
    tools = model(batch["t_ids"], batch["t_mask"])
    logits: torch.Tensor = model.scale * queries @ tools.T
    return logits.masked_fill(~batch["allowed"], float("-inf"))


@register("loss", "retrieval")
def retrieval_loss(model: Retriever, batch: Batch) -> dict[str, torch.Tensor]:
    """InfoNCE: cross-entropy of each request's tool among the batch's tools."""
    return {"contrastive": F.cross_entropy(_logits(model, batch), batch["positive"])}


@register("metrics", "retrieval")
def retrieval_metrics(model: Retriever, batch: Batch) -> dict[str, torch.Tensor]:
    """The loss, and how often the request's tool is the batch's best match (``top1_acc``)."""
    logits = _logits(model, batch)
    return {
        "contrastive": F.cross_entropy(logits, batch["positive"]),
        "top1_acc": (logits.argmax(-1) == batch["positive"]).float().mean(),
    }


def _summary(results: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (key, " · ".join(f"{name} {value:.1%}" for name, value in scores.items()))
        for key, scores in results.items()
        if "@" in key and isinstance(scores, dict)
    ]


@register("report", "retrieval_scores")
def retrieval_scores(data: DataModule, tokenizer: BPETokenizer, batch_size: int) -> Report:
    """At each eval, ``measure`` of the model as a ``Dense`` scorer on the data's registries:
    val and held-out tools in registries of ``eval_size``, all of MASSIVE's 47."""
    del batch_size
    if not isinstance(data, RetrievalData):
        raise TypeError(
            f"the retrieval_scores report needs retrieval data, got {type(data).__name__}"
        )
    columns = [Column(f"{key.split('@')[0]} all@{TOP}", f"{key}.all@{TOP}") for key in data.suites]

    def on_eval(model: Retriever) -> dict[str, Any]:
        scorer = Dense(model, tokenizer)
        results = {key: measure(scorer, cases_) for key, cases_ in data.suites.items()}
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()  # the embedding passes' shapes would stay cached
        return results

    return Report(columns, on_eval, _summary)
