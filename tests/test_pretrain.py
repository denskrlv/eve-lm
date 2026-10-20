import numpy as np
import pytest

from colibri.data.pretrain import (
    SpanCorruptionDataset,
    corrupt,
    noise_mask,
    raw_length,
    span_counts,
    spans_by_sentinel,
    uncorrupt,
)
from colibri.tokenizer.bpe import BYTE0, DEC, EOS, N_SENTINELS, SENTINEL0, is_sentinel, sentinel


def test_window_corrupts_to_exactly_the_input_length():
    window = raw_length(512)
    n_noise, n_spans = span_counts(window, 0.15, 3.0)
    assert window - n_noise + n_spans == 512
    assert n_spans <= N_SENTINELS


def test_noise_mask_has_the_planned_counts():
    rng = np.random.default_rng(0)
    for _ in range(20):
        mask = noise_mask(569, 0.15, 3.0, rng)
        n_noise, n_spans = span_counts(569, 0.15, 3.0)
        starts = mask & ~np.concatenate([[False], mask[:-1]])
        assert mask.sum() == n_noise and starts.sum() == n_spans
        assert not mask[0]  # kept and dropped spans alternate, kept first


def test_corrupt_example():
    tokens = np.array([100, 101, 102, 103, 104, 105])
    mask = np.array([0, 1, 1, 0, 1, 0], dtype=bool)
    inputs, targets = corrupt(tokens, mask)
    assert inputs == [100, sentinel(0), 103, sentinel(1), 105]
    assert targets == [sentinel(0), 101, 102, sentinel(1), 104]
    assert uncorrupt(inputs, targets) == tokens.tolist()


def test_dataset_items():
    tokens = np.arange(BYTE0, BYTE0 + 5000, dtype=np.uint16)
    ds = SpanCorruptionDataset(tokens, input_len=512, seed=1)
    assert len(ds) == 5000 // ds.window
    item = ds[3]
    assert len(item["enc_ids"]) == 512
    assert item["dec_ids"][0] == DEC and item["dec_labels"][-1] == EOS
    assert item["dec_ids"][1:] == item["dec_labels"][:-1]  # shifted by one, as in the book
    assert len(item["dec_ids"]) == len(ds[0]["dec_ids"])  # every sample has the same shape
    window = tokens[3 * ds.window : 4 * ds.window].tolist()
    assert uncorrupt(item["enc_ids"], item["dec_labels"]) == window
    assert ds[3] == item  # each item has a fixed seed
    assert ds[4] != item
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_input_plus_target_rebuild_the_text(tokenizer):
    text = (
        "The kitchen is the room where food is prepared. Turn the lights off when you leave a"
        " room, and set the thermostat to 20 degrees at night to save energy. "
    ) * 8
    tokens = np.array(tokenizer.encode(text), dtype=np.uint16)
    ds = SpanCorruptionDataset(tokens, input_len=64)
    item = ds[0]
    assert any(SENTINEL0 <= t < SENTINEL0 + N_SENTINELS for t in item["enc_ids"])
    original = tokenizer.decode(tokens[: ds.window].tolist())
    assert tokenizer.decode(uncorrupt(item["enc_ids"], item["dec_labels"])) == original


def test_spans_by_sentinel_skips_eos_and_tokens_before_the_first_sentinel() -> None:
    stray, first, second, third = BYTE0 + 1, BYTE0 + 2, BYTE0 + 3, BYTE0 + 4
    targets = [stray, sentinel(0), first, second, sentinel(1), third, EOS]
    assert spans_by_sentinel(targets) == {sentinel(0): [first, second], sentinel(1): [third]}


def test_is_sentinel_covers_exactly_the_sentinel_ids() -> None:
    assert all(is_sentinel(sentinel(i)) for i in range(N_SENTINELS))
    assert not is_sentinel(SENTINEL0 - 1) and not is_sentinel(SENTINEL0 + N_SENTINELS)
