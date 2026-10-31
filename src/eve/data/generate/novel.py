"""The novel-tools benchmark: made-up tools that no training split has, with requests written by
hand, about half of them sharing no word with the tool's name.

Its source is ``data/novel-tools``: one file of tools per domain (``tools/*.json``, the format of
``data/tools``) and ``requests.jsonl`` (``utterance``, ``intent``, ``calls``). ``build`` turns it
into ``novel-tools.jsonl``: each request gets its gold tools plus distractors from other domains,
1 to 5 tools in a random order, as whole schemas. Every CALL and ASK row gets a ``group``:
"shared word" if the request has a word of a gold tool's name, else "no shared word". Models
trained so far match words, not meaning (md/PLAN.md: exact 71% with a shared word, 30% without),
so the second group is the real test.
"""

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from eve import defaults
from eve.data.generate.templates import MAX_PROMPT_TOOLS
from eve.data.generate.validate import check_example
from eve.data.jsonl import read_jsonl
from eve.data.render import TaskExample
from eve.data.schema import SchemaError, Tool, load_library

SHARED, UNSHARED = "shared word", "no shared word"
SUFFIXES = ("ing", "es", "ed", "s")  # "spelling" and "spell" share a word, "tyres" and "tire" don't
MIN_STEM = 3


def stem(word: str) -> str:
    """``word`` without one common ending, if enough of it is left: boxes -> box."""
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= MIN_STEM:
            return word[: -len(suffix)]
    return word


def _stems(words: Iterable[str]) -> set[str]:
    return {stem(w) for w in words}


def shares_word(utterance: str, names: Iterable[str]) -> bool:
    """Whether the request has a word of one of the tool names (``check_uv_index``: check, uv,
    index), endings aside."""
    said = _stems(re.findall(r"[a-z0-9]+", utterance.lower()))
    return any(said & _stems(name.split("_")) for name in names)


def load_novel(
    directory: str | Path = defaults.NOVEL_TOOLS_DIR,
) -> tuple[dict[str, Tool], list[dict[str, Any]]]:
    """The benchmark's tools by name and its requests."""
    directory = Path(directory)
    return load_library(directory / "tools"), read_jsonl(directory / "requests.jsonl")


def _prompt(gold: list[Tool], tools: dict[str, Tool], rng: np.random.Generator) -> tuple[Tool, ...]:
    """The gold tools and distractors from other domains, so no distractor could serve the
    request too: 1 to ``MAX_PROMPT_TOOLS`` tools (at least the gold ones), shuffled."""
    domains = {t.domain for t in gold}
    others = sorted(name for name, t in tools.items() if t.domain not in domains)
    count = int(rng.integers(max(1, len(gold)), MAX_PROMPT_TOOLS + 1))
    picked = [tools[others[i]] for i in rng.permutation(len(others))[: count - len(gold)]]
    shown = gold + picked
    return tuple(shown[i] for i in rng.permutation(len(shown)))


def novel_examples(
    rng: np.random.Generator,
    library: dict[str, Tool],
    directory: str | Path = defaults.NOVEL_TOOLS_DIR,
) -> list[dict[str, Any]]:
    """The benchmark's rows (see the module). Raises ``SchemaError`` listing every problem: a
    tool that is also in ``library`` (then it isn't new) or a request that breaks the rules of
    ``check_example``."""
    tools, requests = load_novel(directory)
    errors = [f"{name} is in the training library" for name in tools if name in library]
    rows = []
    for i, request in enumerate(requests, start=1):
        try:
            gold = [tools[name] for name in dict.fromkeys(c["tool"] for c in request["calls"])]
            shown = _prompt(gold, tools, rng)
            example = TaskExample.from_json(
                {**request, "tools": [t.to_dict() for t in shown]}, None
            )
        except (KeyError, SchemaError, TypeError) as error:
            errors.append(f"request {i}: {error!r}")
            continue
        if problems := check_example(example):
            errors.append(f"request {i} ({example.utterance!r}): {'; '.join(problems)}")
            continue
        row = {**example.to_json(inline=True), "source": "novel"}
        if gold:
            shared = shares_word(example.utterance, [t.name for t in gold])
            row["group"] = SHARED if shared else UNSHARED
        rows.append(row)
    if errors:
        raise SchemaError("novel-tools: " + "\n".join(errors))
    return rows
