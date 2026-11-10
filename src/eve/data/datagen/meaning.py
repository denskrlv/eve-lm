"""Whether two tools mean the same thing, by sentence embeddings (all-MiniLM-L6-v2).

Word overlap can't do this: the generated tools say things in new words on purpose. Used to keep
the eval tools' capabilities out of training, to stop repeats across shards, and to find a
training-library twin of a new tool.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from eve import defaults
from eve.data.datagen.llm import Embed
from eve.data.generate.library import ToolSpec
from eve.data.generate.novel import load_novel
from eve.data.schema import load_library

# Cosine similarities, set from real distributions (md/2026-11-10-datagen-bakeoff.md).
BLOCK_SIMILARITY = 0.55  # this close to a novel-tools or held-out tool: rejected
REPEAT_SIMILARITY = 0.8  # this close to an accepted generated tool: rejected as a repeat
TWIN_SIMILARITY = 0.75  # this close to a training-library tool: kept, with its capability
COHERENCE_FLOOR = 0.3  # a tool's own descriptions less alike than this describe two things
TOPIC_SIMILARITY = 0.5  # a topic line this close to a blocked tool is skipped


class MeaningIndex:
    """Texts by key; ``nearest`` finds the key whose texts are closest to the given ones."""

    def __init__(self, embed: Embed):
        self.embed = embed
        self.keys: list[str] = []
        self._owner: list[int] = []  # for each stored vector, its key's index
        self._vectors: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.keys)

    def add(self, key: str, texts: Sequence[str]) -> None:
        """Store ``texts`` under ``key``."""
        self.add_many([(key, texts)])

    def add_many(self, items: Sequence[tuple[str, Sequence[str]]]) -> None:
        """Store several keys' texts, embedded in one call."""
        flat = [text for _, texts in items for text in texts]
        if not flat:
            return
        self._vectors.append(self.embed(flat))
        for key, texts in items:
            self.keys.append(key)
            self._owner += [len(self.keys) - 1] * len(texts)

    def spread(self, texts: Sequence[str]) -> float:
        """The lowest cosine between any two of ``texts`` (1 for fewer than two)."""
        if len(texts) < 2:
            return 1.0
        vectors = self.embed(texts)
        return float((vectors @ vectors.T)[np.triu_indices(len(texts), 1)].min())

    def nearest(self, texts: Sequence[str]) -> tuple[str, float] | None:
        """The key with the highest cosine between any of its texts and any of ``texts``, and
        that cosine. None if the index is empty."""
        if not self._vectors:
            return None
        scores = (self.embed(texts) @ np.concatenate(self._vectors).T).max(axis=0)
        best = int(scores.argmax())
        return self.keys[self._owner[best]], float(scores[best])


def blocked_index(
    embed: Embed,
    novel_dir: str | Path = defaults.NOVEL_TOOLS_DIR,
    heldout_dir: str | Path = Path(defaults.TOOLS_DIR) / "heldout",
) -> MeaningIndex:
    """The tools generated data must never cover: novel-tools and the held-out travel tools."""
    index = MeaningIndex(embed)
    tools = list(load_novel(novel_dir)[0].values()) + list(load_library(heldout_dir).values())
    index.add_many([(tool.name, [tool.description]) for tool in tools])
    return index


def library_index(embed: Embed, specs: Sequence[ToolSpec]) -> MeaningIndex:
    """The training tools by name, each by its description."""
    index = MeaningIndex(embed)
    index.add_many([(spec.name, [spec.tool.description]) for spec in specs])
    return index
