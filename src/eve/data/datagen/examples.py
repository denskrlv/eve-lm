"""Approved shards as training rows: the prompt around each generated request is put together
here, at build time, so changing the mix needs a rebuild, not new generation.

Each row shows its tool under a random one of the names and descriptions its request shares no
word with, with distractors
from the training library and the other generated tools (never one with the same capability).
A share of call requests becomes UNSUPPORTED by leaving the right tool out. Those, and near
misses (which keep their tool), get only distractors sharing no word or area with the tool, so
none of them can serve the request either.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from eve.data.datagen.checks import clean_pairs
from eve.data.datagen.shard import GenTool, Row, shards
from eve.data.draw import shuffled
from eve.data.generate.library import ToolSpec
from eve.data.generate.sampler import ToolBank, prompt_size
from eve.data.render import TaskExample

REMOVAL_UNSUPPORTED_SHARE = 0.10  # of call requests, shown without their tool


def load_approved(root: str | Path) -> tuple[list[GenTool], list[Row]]:
    """The tools and requests of every approved shard under ``root``."""
    tools: list[GenTool] = []
    rows: list[Row] = []
    for shard in shards(Path(root)):
        if shard.status() == "approved":
            # tool ids repeat across shards: make them unique, rows follow
            prefix = f"{shard.path.name}-"
            tools += [
                GenTool.from_dict({**t.to_dict(), "id": prefix + t.id}) for t in shard.tools()
            ]
            rows += [
                Row(r.id, prefix + r.tool, r.utterance, r.intent, r.args, r.card, r.writer)
                for r in shard.rows()
            ]
    return tools, rows


def _far(bank: ToolBank, gold: int, count: int, rng: np.random.Generator) -> list[int]:
    """Up to ``count`` tools sharing no word or domain with the gold tool, for a request labelled
    UNSUPPORTED: a near neighbour might serve it after all."""
    picked = bank.distractors([gold], 3 * count, rng, unsupported=True, hard=False)
    return [i for i in picked if bank.similarity[gold, i] == 0][:count]


def datagen_examples(
    tools: Sequence[GenTool],
    rows: Sequence[Row],
    train_specs: Sequence[ToolSpec],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """One training row per request (see the module), tools inline."""
    generated = [ToolSpec(t.tool(), (), capability=t.capability) for t in tools]
    bank = ToolBank(list(train_specs) + generated)
    index = {t.id: len(train_specs) + i for i, t in enumerate(tools)}
    by_id = {t.id: t for t in tools}
    out = []
    for row in rows:
        tool, gold = by_id[row.tool], index[row.tool]
        every = [(i, j) for i in range(len(tool.names)) for j in range(len(tool.descriptions))]
        pairs = (clean_pairs(row, tool) if row.intent != "unsupported" else []) or every
        shown = tool.tool(*pairs[int(rng.integers(len(pairs)))])
        intent, calls = row.intent, row.call(shown)
        if intent == "call" and rng.random() < REMOVAL_UNSUPPORTED_SHARE:
            prompt = [bank.specs[i].tool for i in _far(bank, gold, prompt_size(rng), rng)]
            intent, calls = "unsupported", ()
        elif intent == "unsupported":
            far = _far(bank, gold, prompt_size(rng) - 1, rng)
            prompt = [shown] + [bank.specs[i].tool for i in far]
        else:
            picked = bank.distractors([gold], prompt_size(rng) - 1, rng)
            prompt = [shown] + [bank.specs[i].tool for i in picked]
        if not prompt:
            prompt = [shown]  # nothing else to offer: keep the request as written
            intent, calls = row.intent, row.call(shown)
        example = TaskExample(row.utterance, tuple(shuffled(rng, prompt)), intent, calls)
        out.append({**example.to_json(inline=True), "source": "datagen"})
    return out
