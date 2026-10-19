import heapq
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import regex
from tqdm import tqdm

from colibri.config import ModelConfig

N_SENTINELS = 32
N_BYTES = 256  # one base token per byte value
_NO_RANK = 1 << 30  # rank of a pair never merged: above every merge id
SPECIAL_TOKENS: tuple[str, ...] = (
    "<pad>",  # padding, ignored by attention and the loss
    "<bos>",  # start of the encoder input
    "<eos>",  # end of a sequence; also separates documents in the pretraining corpus
    "<utt>",  # the user's utterance follows
    "<tools>",  # the tool schemas follow
    "<tool>",  # one tool: its name, description and arguments follow
    "<arg>",  # one argument of a tool
    "<dec>",  # start of the decoder sequence
    "<call>",  # decoder: one call starts (the tool name is copied in, M8)
    "<val>",  # decoder: an argument value follows
    "<end>",  # decoder: end of an argument value
    "<absent>",  # decoder: an optional argument the user didn't give
    "<ask>",  # decoder: a required argument the user didn't give; we must ask
    *(f"<x{i}>" for i in range(N_SENTINELS)),  # span-corruption sentinels (pretraining only)
)
(PAD, BOS, EOS, UTT, TOOLS, TOOL, ARG, DEC, CALL, VAL, END, ABSENT, ASK) = range(13)
SENTINEL0 = 13
BYTE0 = len(SPECIAL_TOKENS)  # id of byte 0x00; byte b has id BYTE0 + b
SPECIAL_IDS = {token: i for i, token in enumerate(SPECIAL_TOKENS)}


def sentinel(i: int) -> int:
    """The token id of the sentinel ``<x{i}>``."""
    if not 0 <= i < N_SENTINELS:
        raise IndexError(f"there are only {N_SENTINELS} sentinels")
    return SENTINEL0 + i


SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)"""  # contractions: 's 't 'd 'm 'll 've 're
    r"""|[^\r\n\p{L}\p{N}_]?\p{L}+"""  # a word, with at most one leading space or symbol
    r"""|\p{N}"""  # one digit
    r"""| ?[^\s\p{L}\p{N}]+[\r\n]*"""  # a run of punctuation (this is where "_" goes)
    r"""|\s*[\r\n]+"""  # newlines
    r"""|\s+(?!\S)"""  # spaces, except the last one before a word
    r"""|\s+"""
)
_SPLIT = regex.compile(SPLIT_PATTERN)
_SPECIAL_SPLIT = regex.compile("(" + "|".join(regex.escape(t) for t in SPECIAL_TOKENS) + ")")
_CACHE_SIZE = 500_000


def _merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i, (first, second) = [], 0, pair
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == first and ids[i + 1] == second:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-level BPE with special tokens (book ch. 2.5); ``train`` learns it."""

    def __init__(self, merges: Iterable[tuple[int, int]]):
        self.merges: list[tuple[int, int]] = [(a, b) for a, b in merges]
        self.vocab: list[bytes] = [t.encode() for t in SPECIAL_TOKENS] + [
            bytes([b]) for b in range(N_BYTES)
        ]
        self.ranks: dict[tuple[int, int], int] = {}  # pair -> merged id (= its merge order)
        for pair in self.merges:
            self.ranks[pair] = len(self.vocab)
            self.vocab.append(self.vocab[pair[0]] + self.vocab[pair[1]])
        self._cache: dict[str, tuple[int, ...]] = {}

    @property
    def vocab_size(self) -> int:
        """Special tokens + 256 bytes + one token per merge."""
        return len(self.vocab)

    @classmethod
    def train(
        cls,
        texts: Iterable[tuple[str, int]],
        vocab_size: int = ModelConfig.vocab_size,
        min_count: int = 2,
        progress: bool = False,
    ) -> "BPETokenizer":
        """Learn merges from weighted ``texts`` until there are ``vocab_size`` tokens, or
        until no pair is seen ``min_count`` times."""
        n_merges = vocab_size - BYTE0 - N_BYTES
        if n_merges < 0:
            raise ValueError(f"vocab_size must be at least {BYTE0 + N_BYTES}")

        # 1. Word-frequency counts: train on the distinct pieces, not the raw text.
        piece_counts: Counter[str] = Counter()
        for text, weight in texts:
            for part in _SPECIAL_SPLIT.split(text)[::2]:  # odd items are special tokens
                counts = Counter(_SPLIT.findall(part))
                if weight != 1:
                    counts = Counter({piece: n * weight for piece, n in counts.items()})
                piece_counts.update(counts)

        words = [[BYTE0 + b for b in piece.encode("utf-8")] for piece in piece_counts]
        freqs = list(piece_counts.values())

        # 2. Count every adjacent pair, and remember which words contain it.
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for word_id, word in enumerate(words):
            for pair in zip(word, word[1:], strict=False):
                pair_counts[pair] += freqs[word_id]
                where[pair].add(word_id)

        # 3. Merge the most frequent pair, repeatedly. A heap finds the maximum quickly; when a
        #    pair's count changes we push its new count and skip outdated entries when popped.
        #    Ties go to the smaller pair of ids, so training is deterministic.
        heap = [(-n, pair) for pair, n in pair_counts.items()]
        heapq.heapify(heap)
        merges: list[tuple[int, int]] = []
        bar = tqdm(total=n_merges, desc="BPE merges", disable=not progress)
        while len(merges) < n_merges and heap:
            neg_count, pair = heapq.heappop(heap)
            count = pair_counts.get(pair, 0)
            if count != -neg_count:
                continue  # outdated entry
            if count < min_count:
                break
            new_id = BYTE0 + N_BYTES + len(merges)
            merges.append(pair)
            bar.update()

            # Only the words that contain the pair change. Remove their old pairs from the
            # counts and add their new ones.
            changed = set()
            for word_id in where.pop(pair):
                old = words[word_id]
                new = _merge_pair(old, pair, new_id)
                if len(new) == len(old):
                    continue  # this word lost the pair in an earlier merge
                for adjacent in zip(old, old[1:], strict=False):
                    pair_counts[adjacent] -= freqs[word_id]
                    changed.add(adjacent)
                for adjacent in zip(new, new[1:], strict=False):
                    pair_counts[adjacent] += freqs[word_id]
                    where[adjacent].add(word_id)
                    changed.add(adjacent)
                words[word_id] = new
            for adjacent in changed:
                if pair_counts[adjacent] > 0:
                    heapq.heappush(heap, (-pair_counts[adjacent], adjacent))
                else:
                    del pair_counts[adjacent]
        bar.close()
        return cls(merges)

    def encode(self, text: str, specials: bool = False) -> list[int]:
        """Text -> token ids.

        With ``specials=False`` (user text), "<tool>" is just 6 characters. With ``specials=True``
        (text we build ourselves, like a rendered schema) it is the single ``<tool>`` token.
        """
        if not specials:
            return self._encode_ordinary(text)
        ids: list[int] = []
        for i, part in enumerate(_SPECIAL_SPLIT.split(text)):
            if i % 2:
                ids.append(SPECIAL_IDS[part])
            elif part:
                ids.extend(self._encode_ordinary(part))
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in _SPLIT.findall(text):
            cached = self._cache.get(piece)
            if cached is None:
                cached = self._encode_piece(piece)
                if len(self._cache) < _CACHE_SIZE:
                    self._cache[piece] = cached
            ids.extend(cached)
        return ids

    def _encode_piece(self, piece: str) -> tuple[int, ...]:
        ids = [BYTE0 + b for b in piece.encode("utf-8")]
        while len(ids) > 1:
            # The pair that was merged earliest in training goes first.
            pair = min(zip(ids, ids[1:], strict=False), key=lambda p: self.ranks.get(p, _NO_RANK))
            if pair not in self.ranks:
                break
            ids = _merge_pair(ids, pair, self.ranks[pair])
        return tuple(ids)

    def decode(self, ids: Iterable[int]) -> str:
        """Token ids back to text (bad UTF-8 becomes U+FFFD)."""
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def pieces(self, ids: Iterable[int]) -> list[str]:
        """Each token as a string, for looking at how text was split."""
        return [self.vocab[i].decode("utf-8", errors="replace") for i in ids]

    def to_dict(self) -> dict[str, Any]:
        """The merges, plus the split pattern and special tokens they need."""
        return {
            "format": "colibri-bpe-1",
            "split_pattern": SPLIT_PATTERN,
            "special_tokens": list(SPECIAL_TOKENS),
            "merges": [list(m) for m in self.merges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BPETokenizer":
        """A tokenizer from ``to_dict``; refuses other split rules or tokens."""
        if data.get("format") != "colibri-bpe-1":
            raise ValueError("not a Colibri tokenizer")
        if (
            data["split_pattern"] != SPLIT_PATTERN
            or tuple(data["special_tokens"]) != SPECIAL_TOKENS
        ):
            raise ValueError("trained with a different split pattern or special tokens")
        return cls(tuple(m) for m in data["merges"])

    def save(self, path: str | Path) -> None:
        """Write the tokenizer as JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Read a tokenizer written by ``save``."""
        try:
            return cls.from_dict(json.loads(Path(path).read_text()))
        except ValueError as e:
            raise ValueError(f"{path}: {e}") from None
