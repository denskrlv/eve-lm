from pathlib import Path

import pytest

from eve.data.schema import Tool
from eve.model.retriever import query_ids, tool_ids
from eve.tokenizer import tokenizer_from_dict
from eve.tokenizer.bpe import BOS, EOS, SPECIAL_TOKENS, TOOL, TOOLS, UTT, BPETokenizer
from eve.tokenizer.wordpiece import WordPieceTokenizer, basic_words

PIECES = ["[UNK]", "[CLS]", "[SEP]", "[MASK]", "set", "the", "lights", "light", "##s", "cafe"]
PIECES += ["'", "!", "%", "to", "5", "##0", "thermo", "##stat", "東", "京", ":", "_", "room"]


def bert_vocab(tmp_path: Path, slots: int = len(SPECIAL_TOKENS) - 1) -> Path:
    """A BERT-style vocab.txt: [PAD], ``slots`` [unusedN] slots, then real pieces."""
    path = tmp_path / "vocab.txt"
    lines = ["[PAD]", *(f"[unused{i}]" for i in range(slots)), *PIECES]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def wordpiece(tmp_path: Path) -> WordPieceTokenizer:
    return WordPieceTokenizer.from_bert_vocab(bert_vocab(tmp_path))


def test_basic_words_lowercase_strip_accents_and_split_punctuation() -> None:
    assert basic_words("Set the Café's lights to 50%!") == [
        "set",
        "the",
        "cafe",
        "'",
        "s",
        "lights",
        "to",
        "50",
        "%",
        "!",
    ]
    assert basic_words("東京\tok\u200b") == ["東", "京", "ok"]  # CJK spaced, controls dropped
    assert basic_words("set_lights") == ["set", "_", "lights"]  # "_" is punctuation


def test_words_split_into_the_longest_pieces(wordpiece: WordPieceTokenizer) -> None:
    ids = wordpiece.encode("Set the THERMOSTAT, lights 50%")
    assert wordpiece.pieces(ids) == [
        "set",
        "the",
        "thermo",
        "##stat",
        "[UNK]",
        "lights",
        "5",
        "##0",
        "%",
    ]  # "," isn't in the vocab: [UNK]
    assert wordpiece.decode(ids) == "set the thermostat [UNK] lights 50 %"
    assert wordpiece.pieces(wordpiece.encode("lightz")) == ["[UNK]"]  # no split: the whole word


def test_specials_keep_eve_ids(wordpiece: WordPieceTokenizer) -> None:
    assert wordpiece.vocab[: len(SPECIAL_TOKENS)] == list(SPECIAL_TOKENS)
    assert wordpiece.encode("<tool> set_lights", specials=True)[0] == TOOL
    assert TOOL not in wordpiece.encode("<tool> set_lights")  # user text: just characters
    tool = Tool("set_lights", "set the lights", ())
    assert tool_ids(wordpiece, tool, 64)[:3] == [BOS, TOOLS, TOOL]
    assert query_ids(wordpiece, "lights", 64) == [BOS, UTT, wordpiece.ids["lights"], EOS]


def test_saves_and_loads_by_format(wordpiece: WordPieceTokenizer, tokenizer: BPETokenizer) -> None:
    again = tokenizer_from_dict(wordpiece.to_dict())
    assert isinstance(again, WordPieceTokenizer) and again.vocab == wordpiece.vocab
    assert isinstance(tokenizer_from_dict(tokenizer.to_dict()), BPETokenizer)


def test_a_vocab_without_room_for_the_specials_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unused"):
        WordPieceTokenizer.from_bert_vocab(bert_vocab(tmp_path, slots=3))
