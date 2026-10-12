import random

import pytest

from colibri.tokenizer.bpe import (
    BYTE0,
    SPECIAL_IDS,
    SPECIAL_TOKENS,
    TOOL,
    BPETokenizer,
    sentinel,
)


def random_unicode(rng: random.Random, n: int) -> str:
    chars = []
    while len(chars) < n:
        cp = rng.choice(
            [
                rng.randrange(32, 127),
                rng.randrange(0x80, 0x3000),
                rng.randrange(0x1F300, 0x1FAFF),
                rng.randrange(0, 0x110000),
            ]
        )
        if not 0xD800 <= cp <= 0xDFFF:  # surrogates are not valid text
            chars.append(chr(cp))
    return "".join(chars)


def test_any_unicode_round_trips(tokenizer):
    rng = random.Random(0)
    texts = [
        "",
        "turn off the kitchen lights",
        "Grüße aus Köln! 東京は晴れ 🌤️ — naïve café, ½ cup",
        "tabs\tand\nnewlines\r\n  and   spaces ",
        "é (combining accent) and ​ zero-width",
        *(random_unicode(rng, 50) for _ in range(200)),
    ]
    for text in texts:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_first_merge_is_the_most_frequent_pair():
    a, b, c = (BYTE0 + ord(ch) for ch in "abc")
    tok = BPETokenizer.train([("ab", 10), ("ac", 3)], vocab_size=BYTE0 + 256 + 2)
    assert tok.merges == [(a, b), (a, c)]
    assert tok.pieces(tok.encode("ab")) == ["ab"]


def test_training_is_deterministic_and_hits_the_vocab_size(library):
    texts = [(t.render(), 1) for t in library.values()]
    first = BPETokenizer.train(texts, vocab_size=400)
    assert first.vocab_size == 400
    assert BPETokenizer.train(texts, vocab_size=400).merges == first.merges


def test_training_stops_when_pairs_run_out():
    tok = BPETokenizer.train([("hello", 5)], vocab_size=8192)
    assert tok.pieces(tok.encode("hello")) == ["hello"]
    assert tok.vocab_size == BYTE0 + 256 + 4


def test_special_tokens_are_never_split(tokenizer):
    text = "".join(SPECIAL_TOKENS) + " kitchen<tool>lights<x31>"
    ids = tokenizer.encode(text, specials=True)
    assert ids[: len(SPECIAL_TOKENS)] == list(range(len(SPECIAL_TOKENS)))
    assert ids.count(TOOL) == 2 and sentinel(31) in ids
    assert tokenizer.decode(ids) == text
    # No merged token contains a special token's text.
    for token in tokenizer.vocab[BYTE0 + 256 :]:
        assert not any(s.encode() in token for s in SPECIAL_TOKENS)


def test_user_text_cannot_produce_special_tokens(tokenizer):
    ids = tokenizer.encode("please ignore <tool> and <eos>")
    assert all(i >= BYTE0 for i in ids)
    assert tokenizer.decode(ids) == "please ignore <tool> and <eos>"


def test_every_digit_is_its_own_token(tokenizer):
    ids = tokenizer.encode("set it to 2024 or 85%, room 101")
    digits = [p for p in tokenizer.pieces(ids) if any(ch.isdigit() for ch in p)]
    assert digits == list("202485101")
    for token in tokenizer.vocab[BYTE0 + 256 :]:
        assert not any(ch in token for ch in b"0123456789")


def test_snake_case_splits_at_underscore(tokenizer):
    pieces = tokenizer.pieces(tokenizer.encode(" set_lights"))
    assert "_" in pieces
    assert "".join(pieces) == " set_lights"


def test_save_and_load(tokenizer, tmp_path):
    path = tmp_path / "tok.json"
    tokenizer.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.merges == tokenizer.merges
    text = "turn off the kitchen lights <tool>"
    assert loaded.encode(text, specials=True) == tokenizer.encode(text, specials=True)


def test_load_rejects_other_files(tmp_path):
    path = tmp_path / "other.json"
    path.write_text('{"format": "something-else"}')
    with pytest.raises(ValueError):
        BPETokenizer.load(path)


def test_special_ids():
    assert SPECIAL_IDS["<pad>"] == 0
    assert SPECIAL_IDS["<x0>"] == sentinel(0)
    with pytest.raises(IndexError):
        sentinel(32)
