"""Training the splitter (``eve.model.splitter``): data, loss, metrics and report.

Items are the task splits' requests whose actions are known (``row_actions``): template parts,
xLAM rows labeled by rule, and single-action rows. Each word's first token is labeled with the
actions the word belongs to. Val is carved out of train, grouped by request text.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from eve import defaults
from eve.components import Batch, DataModule
from eve.config import ModelConfig
from eve.data.generate.actions import row_actions
from eve.data.generate.validate import normalize
from eve.data.jsonl import read_jsonl
from eve.data.render import bucket
from eve.model.checkpoint import SPLIT_TASK
from eve.model.splitter import LENGTHS, Splitter, SplitterConfig
from eve.registry import register
from eve.split.decide import Thresholds, decide
from eve.split.words import Span, action_text, encode_words, word_spans, words_in
from eve.tokenizer.bpe import PAD, BPETokenizer
from eve.train.loop import Column, Report

TASK = SPLIT_TASK  # meta["task"] of the checkpoints a splitter run writes
SHOWN = 4  # val requests whose predicted actions each eval shows
COLUMNS = (
    Column("val loss", "val_loss"),
    Column("val exact split", "val_exact_split"),
    Column("val wrong split", "val_wrong_split"),
)


@dataclass(frozen=True)
class SplitItem:
    """One request: its encoder ids, each word's first-token index, and each action's words."""

    text: str
    ids: list[int]
    first: list[int]
    members: list[list[int]]
    words: list[Span] = field(repr=False)


@dataclass(frozen=True)
class SplitData:
    """Train and val items, batched to one of ``LENGTHS``. Batches have ``ids``, ``mask``
    (B, n); ``word_mask`` (B, n), True at each word's first token; ``exists`` (B, K) 0/1; and
    ``members`` (B, K, n) 0/1 at first tokens."""

    train: list[SplitItem]
    val: list[SplitItem]
    n_actions: int = 8

    def collate(self, items: Sequence[SplitItem]) -> Batch:
        n = bucket(max(len(item.ids) for item in items), LENGTHS)
        b, k = len(items), self.n_actions
        ids = torch.full((b, n), PAD, dtype=torch.long)
        word_mask = torch.zeros(b, n, dtype=torch.bool)
        exists = torch.zeros(b, k)
        members = torch.zeros(b, k, n)
        for row, item in enumerate(items):
            ids[row, : len(item.ids)] = torch.tensor(item.ids)
            word_mask[row, item.first] = True
            exists[row, : len(item.members)] = 1.0
            for slot, words in enumerate(item.members):
                members[row, slot, [item.first[w] for w in words]] = 1.0
        return {
            "enc_ids": ids,
            "enc_mask": ids != PAD,
            "word_mask": word_mask,
            "exists": exists,
            "members": members,
        }

    def batch_tokens(self, batch: Batch) -> int:
        """The real tokens of a batch."""
        return int(batch["enc_mask"].sum())


def _item(row: dict[str, Any], tokenizer: BPETokenizer, max_len: int, k: int) -> SplitItem | None:
    actions = row_actions(row)
    if actions is None or not 1 <= len(actions) <= k:
        return None
    text = row["utterance"]
    try:
        ids, first = encode_words(tokenizer, text, max_len)
    except ValueError:
        return None
    words = word_spans(text)
    members = [words_in(words, action) for action in actions]
    if any(not m for m in members):
        return None
    return SplitItem(text, ids, first, members, words)


@register("data", "split")
def split_data(
    tokenizer: BPETokenizer,
    model_cfg: ModelConfig,
    seed: int,
    path: str = defaults.TASK_DIR,
    single_share: float = 0.5,
    val_share: float = 0.1,
) -> SplitData:
    """``path``'s ``train.jsonl`` rows with known actions, as items. Single-action items are
    cut (at random) to at most ``single_share`` of all items, unless there are no others;
    ``val_share`` of the request texts (and every copy of them) go to val."""
    max_len = min(getattr(model_cfg, "max_split_len", LENGTHS[-1]), LENGTHS[-1])
    k = getattr(model_cfg, "n_actions", SplitterConfig.n_actions)
    items = [
        item
        for row in read_jsonl(Path(path) / "train.jsonl")
        if (item := _item(row, tokenizer, max_len, k)) is not None
    ]
    rng = np.random.default_rng(seed)
    multi = [item for item in items if len(item.members) > 1]
    single = [item for item in items if len(item.members) == 1]
    if multi and single_share < 1:
        keep = int(len(multi) * single_share / (1 - single_share))
        single = [single[i] for i in sorted(rng.permutation(len(single))[:keep])]
    items = multi + single

    groups: defaultdict[str, list[SplitItem]] = defaultdict(list)
    for item in items:
        groups[normalize(item.text)].append(item)
    keys = sorted(groups)
    n_val = round(val_share * len(keys))
    if val_share > 0 and len(keys) >= 2:
        n_val = min(max(1, n_val), len(keys) - 1)
    val_keys = {keys[i] for i in rng.permutation(len(keys))[:n_val]}
    train = [item for key in keys if key not in val_keys for item in groups[key]]
    val = [item for key in keys if key in val_keys for item in groups[key]]
    return SplitData(train, val, k)


def _member_loss(logits: torch.Tensor, batch: Batch) -> torch.Tensor:
    weight = batch["exists"][:, :, None] * batch["word_mask"][:, None, :]
    safe = logits.masked_fill(~batch["enc_mask"][:, None, :], 0.0)  # padded -inf would give NaN
    loss = F.binary_cross_entropy_with_logits(safe, batch["members"], reduction="none")
    return (loss * weight).sum() / weight.sum().clamp(min=1)


@register("loss", "split")
def split_loss(model: Splitter, batch: Batch) -> dict[str, torch.Tensor]:
    """Binary cross-entropy of every slot's existence, and of each existing slot's word
    memberships (at words' first tokens)."""
    out = model(batch["enc_ids"], batch["enc_mask"])
    return {
        "exists": F.binary_cross_entropy_with_logits(out.exists.float(), batch["exists"]),
        "members": _member_loss(out.members.float(), batch),
    }


@register("metrics", "split")
def split_metrics(model: Splitter, batch: Batch) -> dict[str, torch.Tensor]:
    """The losses; ``exact_split``: every slot's existence and every existing slot's words
    right (at P 0.5); ``wrong_split``: single-action requests given a second action."""
    out = model(batch["enc_ids"], batch["enc_mask"])
    exists_pred = out.exists.float() > 0
    exists_ok = (exists_pred == batch["exists"].bool()).all(-1)
    member_pred = out.members.float() > 0
    wrong = (member_pred != batch["members"].bool()) & batch["word_mask"][:, None, :]
    members_ok = ~(wrong & batch["exists"].bool()[:, :, None]).any(-1).any(-1)
    single = batch["exists"].sum(-1) == 1
    split = exists_pred[:, 1] & single
    return {
        "exists": F.binary_cross_entropy_with_logits(out.exists.float(), batch["exists"]),
        "members": _member_loss(out.members.float(), batch),
        "exact_split": (exists_ok & members_ok).float().mean(),
        "wrong_split": split.sum().float() / single.sum().clamp(min=1),
    }


def _describe(model: Splitter, data: SplitData, items: Sequence[SplitItem]) -> list[str]:
    lines = []
    for item in items:
        batch = {k: v.to(next(model.parameters()).device) for k, v in data.collate([item]).items()}
        out = model(batch["enc_ids"], batch["enc_mask"])
        exists = torch.sigmoid(out.exists[0].float()).tolist()
        members = torch.sigmoid(out.members[0, :, item.first].float()).tolist()
        actions = decide(exists, members, item.words, item.text, Thresholds(0.5, 0.5))
        said = (
            [item.text]
            if actions is None
            else [action_text(item.text, item.words, a) for a in actions]
        )
        gold = [action_text(item.text, item.words, a) for a in item.members]
        lines.append(f"{item.text} → {' | '.join(said)} | gold {' | '.join(gold)}")
    return lines


@register("report", "split_scores")
def split_scores(data: DataModule, tokenizer: BPETokenizer, batch_size: int) -> Report:
    """The eval table's val loss, exact and wrong split; at each eval, the predicted actions
    (at P 0.5) of a few multi-action val requests."""
    del tokenizer, batch_size
    if not isinstance(data, SplitData):
        raise TypeError(f"the split_scores report needs split data, got {type(data).__name__}")
    shown = [item for item in data.val if len(item.members) > 1][:SHOWN] or data.val[:SHOWN]

    @torch.no_grad()
    def on_eval(model: Splitter) -> dict[str, Any]:
        was_training = model.training
        model.eval()
        try:
            return {"samples": _describe(model, data, shown)}
        finally:
            model.train(was_training)

    return Report(COLUMNS, on_eval, None, ((r"→ [^|]*", "bold"), (r"\| gold .*", "dim")))
