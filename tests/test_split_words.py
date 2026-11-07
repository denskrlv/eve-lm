import pytest

from eve.split.words import action_text, encode_words, word_spans, words_in
from eve.tokenizer.bpe import BOS, EOS, UTT, BPETokenizer


def _texts(text: str) -> list[str]:
    return [text[a:b] for a, b in word_spans(text)]


def test_word_spans_cover_words() -> None:
    assert _texts("turn off the lights, please") == ["turn", "off", "the", "lights", ",", "please"]


def test_encode_words_first_tokens(tokenizer: BPETokenizer) -> None:
    text = "turn off the kitchen lights, please"
    ids, first = encode_words(tokenizer, text, 64)
    assert ids[:2] == [BOS, UTT] and ids[-1] == EOS
    words = _texts(text)
    assert len(first) == len(words)
    for index, word in zip(first, words, strict=True):
        assert word.startswith(tokenizer.decode([ids[index]]).strip())
    assert ids[2:-1] == tokenizer.encode(" " + text)  # what Eve's encoder reads


def test_action_text_runs_and_gaps() -> None:
    text = "turn off the lights in the bathroom and hallway lights"
    assert action_text(text, word_spans(text), [0, 1, 2, 3, 8, 9]) == (
        "turn off the lights hallway lights"
    )


def test_action_text_keeps_punctuation_inside_runs() -> None:
    text = "set it to 7.5, then stop"
    words = word_spans(text)
    assert action_text(text, words, range(6)) == "set it to 7.5"


def test_words_with_quotes_and_accents() -> None:
    text = "peers for 'WMT' and Tom's café"
    words = word_spans(text)
    assert action_text(text, words, range(len(words))) == text
    assert "café" in _texts(text)


def test_words_in() -> None:
    text = "turn off the fan and lock the door"
    words = word_spans(text)
    assert words_in(words, [(0, 16), (21, 34)]) == [0, 1, 2, 3, 5, 6, 7]


def test_encode_words_too_long(tokenizer: BPETokenizer) -> None:
    with pytest.raises(ValueError, match="too long"):
        encode_words(tokenizer, " ".join(["lights"] * 20), 8)
