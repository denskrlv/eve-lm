"""The registered data modules (kind "data"): a run's train and val sets and how to batch them.

A builder takes ``tokenizer``, ``model_cfg`` and ``seed`` at run time, and its config params as
keywords. The batches' keys are in ``eve.components``.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from eve import defaults
from eve.components import Batch
from eve.config import ModelConfig
from eve.data.corpus import load_tokens
from eve.data.jsonl import read_jsonl
from eve.data.pretrain import SpanCorruptionDataset
from eve.data.render import Rendered, TaskCollator, TaskDataset, TaskExample, collate_pretrain
from eve.data.schema import Tool
from eve.registry import register
from eve.tokenizer.bpe import BPETokenizer

VAL_SEED = 1  # val's tool shuffle: unlike train's (seed 0), and the same at every eval


def batch_tokens(batch: Batch) -> int:
    """The real encoder and decoder tokens of a batch (padding left out)."""
    return int(batch["enc_mask"].sum() + batch["dec_mask"].sum())


@dataclass(frozen=True)
class SpanCorruptionData:
    """Span-corruption windows of a token corpus (box 4). Batches have ``enc_*`` and ``dec_*``
    keys, no tool or question keys."""

    train: SpanCorruptionDataset
    val: SpanCorruptionDataset

    def collate(self, items: Sequence[dict[str, list[int]]]) -> Batch:
        """Pad windows into a batch (``collate_pretrain``)."""
        return collate_pretrain(items)

    def batch_tokens(self, batch: Batch) -> int:
        """See ``batch_tokens``."""
        return batch_tokens(batch)


@register("data", "span_corruption")
def span_corruption(
    tokenizer: BPETokenizer, model_cfg: ModelConfig, seed: int, path: str = defaults.BIN_DIR
) -> SpanCorruptionData:
    """``path``'s ``train.bin`` and ``val.bin`` in windows of the model's ``max_encoder_len``,
    so every encoder position that fine-tuning uses has been trained (learned positions only
    learn where they are used). Train's corruption is seeded by ``seed``, val's is fixed."""
    data_dir = Path(path)
    return SpanCorruptionData(
        train=SpanCorruptionDataset(
            load_tokens(data_dir / "train.bin"), model_cfg.max_encoder_len, seed=seed
        ),
        val=SpanCorruptionDataset(load_tokens(data_dir / "val.bin"), model_cfg.max_encoder_len),
    )


def load_examples(path: str | Path, library: dict[str, Tool]) -> list[TaskExample]:
    """The examples of a task .jsonl file, with tools looked up in ``library``."""
    return [TaskExample.from_json(row, library) for row in read_jsonl(path)]


def in_batches(
    data: TaskDataset, collate: TaskCollator, size: int
) -> list[dict[str, torch.Tensor]]:
    """All of ``data`` in order, in batches of ``size`` (the last one may be smaller)."""
    return [
        collate([data[j] for j in range(i, min(i + size, len(data)))])
        for i in range(0, len(data), size)
    ]


@dataclass(frozen=True)
class TaskData:
    """The task splits (M6) rendered for the question heads and the value decoder: train, val
    and the held-out-tool set, which only the eval report scores. Batches have every key."""

    train: TaskDataset
    val: TaskDataset
    val_examples: list[TaskExample]  # val's examples as written, for the report's samples
    heldout: TaskDataset
    collator: TaskCollator = field(repr=False)

    def collate(self, items: Sequence[Rendered]) -> Batch:
        """Pad rendered examples into a batch (``TaskCollator``)."""
        return self.collator(items)

    def batch_tokens(self, batch: Batch) -> int:
        """See ``batch_tokens``."""
        return batch_tokens(batch)


@register("data", "task")
def task(
    tokenizer: BPETokenizer,
    model_cfg: ModelConfig,
    seed: int,
    path: str = defaults.TASK_DIR,
    library: dict[str, Tool] | None = None,
) -> TaskData:
    """``path``'s ``train.jsonl``, ``val.jsonl`` and ``heldout-templates.jsonl``, with tools
    looked up in ``library`` (default: every known tool, ``Sources.load().library``).

    Train's tool shuffle is seeded 0 whatever ``seed`` is, as before components were
    swappable; val's is ``VAL_SEED``.
    """
    del seed  # see the docstring
    if library is None:
        from eve.data.generate.build import Sources

        library = Sources.load().library
    task_dir = Path(path)
    val_examples = load_examples(task_dir / "val.jsonl", library)
    return TaskData(
        train=TaskDataset(load_examples(task_dir / "train.jsonl", library), tokenizer, model_cfg),
        val=TaskDataset(val_examples, tokenizer, model_cfg, seed=VAL_SEED),
        val_examples=val_examples,
        heldout=TaskDataset(
            load_examples(task_dir / "heldout-templates.jsonl", library), tokenizer, model_cfg
        ),
        collator=TaskCollator(model_cfg),
    )
