"""When to split, and the splitter's answer for a request.

The splitter splits a request only when it is sure (``decide``); otherwise the request goes
to Eve whole, as without a splitter. A missed split costs little (Eve gets a longer request),
a wrong one a lot (the request is cut apart), so the thresholds are picked to keep wrong
splits of single-action requests rare (``pick_thresholds``).
"""

import itertools
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from eve.model.checkpoint import Checkpoint
from eve.model.splitter import LENGTHS, Splitter
from eve.runtime import eval_mode
from eve.split.words import Span, action_text, encode_words, word_spans
from eve.tokenizer.bpe import PAD

EXISTS_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
MEMBER_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
MAX_WRONG_SPLIT = 0.02  # of single-action requests (the design's budget)
_FILLER = re.compile(r"(?:and|also|additionally|then|furthermore|moreover|finally|lastly)", re.I)


@dataclass(frozen=True)
class Thresholds:
    """How sure the splitter must be that an action exists, and of every word's membership."""

    exists: float
    member: float

    def to_json(self) -> dict[str, float]:
        return {"exists": self.exists, "member": self.member}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Thresholds":
        return cls(float(data["exists"]), float(data["member"]))


def _real(text: str, words: Sequence[Span], members: Sequence[int]) -> bool:
    """Whether the action has a word that is more than punctuation or a connector."""
    for i in members:
        word = text[words[i][0] : words[i][1]]
        if re.search(r"\w", word) and not _FILLER.fullmatch(word):
            return True
    return False


def decide(
    exists_p: Sequence[float],
    member_p: Sequence[Sequence[float]],
    words: Sequence[Span],
    text: str,
    th: Thresholds,
) -> list[list[int]] | None:
    """The actions (word indices per action) if the splitter is sure of a split into two or
    more, else None ("don't split"). Sure means: the first n slots exist with P ≥ ``th.exists``
    and every later one with P ≤ 1 − ``th.exists``; in each of the n, every word is in or out
    with P ≥ ``th.member``; and every action has a real word."""
    n = 0
    while n < len(exists_p) and exists_p[n] >= th.exists:
        n += 1
    if n < 2 or any(p > 1 - th.exists for p in exists_p[n:]):
        return None
    actions = []
    for slot in range(n):
        probs = member_p[slot][: len(words)]
        if any(max(p, 1 - p) < th.member for p in probs):
            return None
        members = [i for i, p in enumerate(probs) if p >= 0.5]
        if not _real(text, words, members):
            return None
        actions.append(members)
    return actions


@torch.no_grad()
def split_probs(
    ckpt: Checkpoint[Splitter], texts: Sequence[str]
) -> list[tuple[list[float], list[list[float]], list[Span]] | None]:
    """Per text, the splitter's P(exists) per slot, P(member) per slot and word (at each
    word's first token) and the words; None for a text too long to read."""
    model, cfg = ckpt.model, ckpt.model.cfg
    out: list[tuple[list[float], list[list[float]], list[Span]] | None] = []
    with eval_mode(model) as device:
        for text in texts:
            try:
                ids, first = encode_words(ckpt.tokenizer, text, min(cfg.max_split_len, LENGTHS[-1]))
            except ValueError:
                out.append(None)
                continue
            n = next(size for size in LENGTHS if len(ids) <= size)
            padded = torch.full((1, n), PAD, dtype=torch.long)
            padded[0, : len(ids)] = torch.tensor(ids)
            logits = model(padded.to(device), (padded != PAD).to(device))
            exists = torch.sigmoid(logits.exists[0].float()).tolist()
            members = torch.sigmoid(logits.members[0, :, first].float()).tolist()
            out.append((exists, members, word_spans(text)))
    return out


def split_request(ckpt: Checkpoint[Splitter], text: str, th: Thresholds) -> list[str]:
    """The request's actions as texts, in order; ``[text]`` when it isn't split."""
    (probs,) = split_probs(ckpt, [text])
    if probs is None:
        return [text]
    exists, members, words = probs
    actions = decide(exists, members, words, text, th)
    if actions is None:
        return [text]
    return [action_text(text, words, action) for action in actions]


@dataclass(frozen=True)
class SplitCase:
    """The splitter's probabilities for one labeled request, and its gold actions."""

    exists_p: list[float]
    member_p: list[list[float]]
    words: list[Span]
    text: str
    gold: list[list[int]]


def pick_thresholds(
    cases: Sequence[SplitCase], max_wrong_split: float = MAX_WRONG_SPLIT
) -> Thresholds:
    """Of the grid's pairs that split at most ``max_wrong_split`` of the single-action cases,
    the one that splits the most multi-action cases exactly right (ties: the looser pair).
    If no pair keeps the budget, the strictest pair."""
    singles = [c for c in cases if len(c.gold) == 1]
    multis = [c for c in cases if len(c.gold) > 1]
    best: tuple[int, Thresholds] | None = None
    for exists, member in itertools.product(EXISTS_GRID, MEMBER_GRID):
        th = Thresholds(exists, member)
        wrong = sum(
            decide(c.exists_p, c.member_p, c.words, c.text, th) is not None for c in singles
        )
        if wrong > max_wrong_split * len(singles):
            continue
        right = sum(decide(c.exists_p, c.member_p, c.words, c.text, th) == c.gold for c in multis)
        if best is None or right > best[0]:
            best = (right, th)
    return best[1] if best else Thresholds(EXISTS_GRID[-1], MEMBER_GRID[-1])
