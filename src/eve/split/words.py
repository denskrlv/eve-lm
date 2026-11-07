"""A request's words, as the splitter labels and points at them.

A word is one piece of the tokenizer's pre-split (``split_pieces``) that isn't only
whitespace: "turn", "off", ",", "7", "'s". An action is a set of word indices; its text is
those words in sentence order (``action_text``). Labels are character spans of the request,
turned into word indices by ``words_in``.
"""

from collections.abc import Iterable, Sequence

from eve.tokenizer.bpe import BOS, EOS, UTT, BPETokenizer, split_pieces

Span = tuple[int, int]  # character start and end in the request


def _pieces(text: str) -> list[tuple[str, int]]:
    """The pieces of ``" " + text`` (as Eve's encoder reads a request) with each one's start
    in ``text`` (the added space is at -1)."""
    out, start = [], -1
    for piece in split_pieces(" " + text):
        out.append((piece, start))
        start += len(piece)
    return out


def word_spans(text: str) -> list[Span]:
    """The character span of each word in ``text``, leading whitespace left out."""
    spans = []
    for piece, start in _pieces(text):
        stripped = piece.lstrip()
        if stripped:
            begin = start + len(piece) - len(stripped)
            spans.append((begin, begin + len(stripped.rstrip())))
    return spans


def encode_words(tokenizer: BPETokenizer, text: str, max_len: int) -> tuple[list[int], list[int]]:
    """The encoder ids ``[BOS, UTT, *tokens of " " + text, EOS]`` and, per word, the index of
    its first token in them. Raises ``ValueError`` if the ids don't fit in ``max_len``."""
    ids, first = [BOS, UTT], []
    for piece, _ in _pieces(text):
        tokens = tokenizer.encode(piece)
        if piece.strip():
            first.append(len(ids))
        ids.extend(tokens)
    ids.append(EOS)
    if len(ids) > max_len:
        raise ValueError(f"the request is too long: {len(ids)} tokens, at most {max_len}")
    return ids, first


def words_in(words: Sequence[Span], spans: Sequence[Span]) -> list[int]:
    """The indices of the words that overlap any of ``spans``, ascending."""
    return [
        i for i, (a, b) in enumerate(words) if any(a < end and start < b for start, end in spans)
    ]


def action_text(text: str, words: Sequence[Span], members: Iterable[int]) -> str:
    """The member words in sentence order: a run of consecutive words keeps the request's own
    characters between them, and runs are joined by one space."""
    runs: list[list[int]] = []
    for i in sorted(set(members)):
        if runs and runs[-1][1] == i - 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return " ".join(text[words[a][0] : words[b][1]] for a, b in runs)
