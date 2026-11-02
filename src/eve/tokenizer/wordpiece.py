"""BERT's uncased WordPiece tokenizer, for a retriever that starts from a pretrained BERT
encoder (MiniLM, ``eve.model.bert``) instead of Eve's own.

Text is cleaned, lowercased, stripped of accents and split at spaces and punctuation (BERT's
"basic" tokenizer); each word is then split into the longest pieces of the vocabulary from the
left, continuations marked ``##`` ("thermostat" -> "the", "##rm", "##osta", "##t"). A word with
no split becomes ``[UNK]``.

Lowercasing loses what Eve's value decoder needs to write "John" rather than "john", so
``encode_cased`` also gives each token the case of the word it came from (``CASES``), which
Eve adds to the encoder's input.

Eve's special tokens keep their BPE ids (``<pad>`` 0, ``<bos>`` 1 …): BERT's vocabulary has
``[PAD]`` at 0 and ~1,000 ``[unusedN]`` slots after it, so the specials take those slots and the
code that builds Eve's prompts (``BOS``, ``TOOLS``, ``PAD``…) works with either tokenizer.
"""

import unicodedata
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from eve.tokenizer.bpe import _SPECIAL_SPLIT, SPECIAL_IDS, SPECIAL_TOKENS

FORMAT = "eve-wordpiece-1"
UNK = "[UNK]"
CONTINUATION = "##"
MAX_WORD_CHARS = 100  # a longer word is [UNK], as in BERT
_CACHE_SIZE = 500_000
LOWER, TITLE, UPPER, MIXED = range(4)  # a word's case; no letters counts as LOWER
CASES = 4


def _is_punctuation(char: str) -> bool:
    """BERT's rule: every non-letter, non-digit ASCII symbol ("$", "^", "`" too), and Unicode
    punctuation."""
    cp = ord(char)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(char).startswith("P")


def _is_cjk(cp: int) -> bool:
    """A CJK ideograph: BERT puts spaces around each, so each is a word of its own."""
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F
        or 0x2B740 <= cp <= 0x2B81F
        or 0x2B820 <= cp <= 0x2CEAF
        or 0xF900 <= cp <= 0xFAFF
        or 0x2F800 <= cp <= 0x2FA1F
    )


def _clean(text: str) -> str:
    """Drop NUL, U+FFFD and control characters; turn every kind of space into " "; space out
    CJK ideographs."""
    out = []
    for char in text:
        cp = ord(char)
        if char in " \t\n\r" or unicodedata.category(char) == "Zs":
            out.append(" ")
        elif cp == 0 or cp == 0xFFFD or unicodedata.category(char).startswith("C"):
            continue
        elif _is_cjk(cp):
            out.append(f" {char} ")
        else:
            out.append(char)
    return "".join(out)


def case_of(word: str) -> int:
    """``LOWER`` ("lights", "50"), ``TITLE`` ("John"), ``UPPER`` ("NYC") or ``MIXED`` ("iPhone")."""
    if word.islower() or not any(c.isupper() for c in word):
        return LOWER
    if word.isupper():
        return UPPER if sum(c.isalpha() for c in word) > 1 else TITLE  # "A", "I": capitalized
    if word[0].isupper() and (word[1:].islower() or not any(c.isalpha() for c in word[1:])):
        return TITLE
    return MIXED


def cased_words(text: str) -> list[str]:
    """``basic_words`` before lowercasing: ``"Set the Café's"`` -> ``Set the Cafe ' s``."""
    words = []
    for token in _clean(text).split():
        token = unicodedata.normalize("NFD", token)
        token = "".join(c for c in token if unicodedata.category(c) != "Mn")
        word = ""
        for char in token:
            if _is_punctuation(char):
                if word:
                    words.append(word)
                words.append(char)
                word = ""
            else:
                word += char
        if word:
            words.append(word)
    return words


def basic_words(text: str) -> list[str]:
    """BERT's basic tokenizer, uncased: lowercase, no accents, split at spaces and around every
    punctuation character. ``"Set the Café's lights!"`` -> ``set the cafe ' s lights !``."""
    return [word.lower() for word in cased_words(text)]


class WordPieceTokenizer:
    """WordPiece over a fixed vocabulary (BERT's ``vocab.txt``, with Eve's specials in front)."""

    def __init__(self, vocab: Sequence[str]):
        self.vocab: list[str] = list(vocab)
        if tuple(self.vocab[: len(SPECIAL_TOKENS)]) != SPECIAL_TOKENS:
            raise ValueError("the vocabulary must start with Eve's special tokens")
        self.ids = {piece: i for i, piece in enumerate(self.vocab)}
        self.unk = self.ids[UNK]
        self._cache: dict[str, tuple[int, ...]] = {}

    @classmethod
    def from_bert_vocab(cls, path: str | Path) -> "WordPieceTokenizer":
        """BERT's ``vocab.txt`` (one piece per line), with ``[PAD]`` and the ``[unusedN]`` slots
        after it given to Eve's special tokens."""
        bert = Path(path).read_text(encoding="utf-8").splitlines()
        slots = bert[: len(SPECIAL_TOKENS)]
        if slots[0] != "[PAD]" or any(s != f"[unused{i}]" for i, s in enumerate(slots[1:])):
            raise ValueError(f"{path}: expected [PAD] and then [unused0]… at the start")
        return cls(list(SPECIAL_TOKENS) + bert[len(SPECIAL_TOKENS) :])

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, specials: bool = False) -> list[int]:
        """Text -> token ids; ``specials`` as in ``BPETokenizer.encode``."""
        return self.encode_cased(text, specials)[0]

    def encode_cased(self, text: str, specials: bool = False) -> tuple[list[int], list[int]]:
        """Token ids, and per token the case of its word (``case_of``; specials are LOWER)."""
        if not specials:
            return self._encode_ordinary(text)
        ids: list[int] = []
        cases: list[int] = []
        for i, part in enumerate(_SPECIAL_SPLIT.split(text)):
            if i % 2:
                ids.append(SPECIAL_IDS[part])
                cases.append(LOWER)
            elif part:
                more, more_cases = self._encode_ordinary(part)
                ids.extend(more)
                cases.extend(more_cases)
        return ids, cases

    def _encode_ordinary(self, text: str) -> tuple[list[int], list[int]]:
        ids: list[int] = []
        cases: list[int] = []
        for word in cased_words(text):
            lower = word.lower()
            cached = self._cache.get(lower)
            if cached is None:
                cached = self._encode_word(lower)
                if len(self._cache) < _CACHE_SIZE:
                    self._cache[lower] = cached
            ids.extend(cached)
            cases.extend([case_of(word)] * len(cached))
        return ids, cases

    def _encode_word(self, word: str) -> tuple[int, ...]:
        """Greedy longest match from the left; ``[UNK]`` for the whole word if a part has no
        piece."""
        if len(word) > MAX_WORD_CHARS:
            return (self.unk,)
        ids = []
        start = 0
        while start < len(word):
            for end in range(len(word), start, -1):
                piece = word[start:end] if start == 0 else CONTINUATION + word[start:end]
                if piece in self.ids:
                    ids.append(self.ids[piece])
                    start = end
                    break
            else:
                return (self.unk,)
        return tuple(ids)

    def decode(self, ids: Iterable[int]) -> str:
        """Token ids back to (lowercased) text: pieces joined, ``##`` continuations glued on."""
        return " ".join(self.pieces(ids)).replace(" " + CONTINUATION, "")

    def pieces(self, ids: Iterable[int]) -> list[str]:
        """Each token as a string, for looking at how text was split."""
        return [self.vocab[i] for i in ids]

    def to_dict(self) -> dict[str, Any]:
        return {"format": FORMAT, "vocab": self.vocab}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WordPieceTokenizer":
        if data.get("format") != FORMAT:
            raise ValueError("not an Eve WordPiece tokenizer")
        return cls(data["vocab"])
