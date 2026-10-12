import numpy as np
import pytest

from colibri.config import INTENTS, preset
from colibri.data.render import (
    ENCODER_BUCKETS,
    IGNORE,
    Call,
    TaskCollator,
    TaskDataset,
    TaskExample,
    bucket,
    collate_pretrain,
    render,
)
from colibri.data.schema import MAX_ARGS, SchemaError
from colibri.tokenizer.bpe import ARG, BOS, DEC, EOS, PAD, TOOL, TOOLS, UTT

CFG = preset("tiny")
STOP = CFG.max_tools


def example(
    library, intent="call", calls=None, tools=("set_lights", "set_thermostat", "play_music")
):
    if calls is None:
        calls = [Call("set_lights", {"room": "kitchen", "on": False})]
    return TaskExample(
        "turn off the kitchen lights", tuple(library[t] for t in tools), intent, tuple(calls)
    )


def test_encoder_input_and_markers(library, tokenizer):
    r = render(example(library), tokenizer, CFG)
    assert r.enc_ids[:2] == [BOS, UTT] and r.enc_ids[-1] == EOS and TOOLS in r.enc_ids
    assert tokenizer.decode(r.enc_ids[2 : r.enc_ids.index(TOOLS)]) == " turn off the kitchen lights"
    assert [r.enc_ids[p] for p in r.tool_pos] == [TOOL] * 3
    for name, tool_pos, arg_pos in zip(r.tool_names, r.tool_pos, r.arg_pos, strict=True):
        assert tokenizer.decode(r.enc_ids[tool_pos + 1 :]).startswith(f" {name} :")
        tool = library[name]
        assert len(arg_pos) == len(tool.args)
        for arg, p in zip(tool.args, arg_pos, strict=True):
            assert r.enc_ids[p] == ARG
            assert tokenizer.decode(r.enc_ids[p + 1 :]).startswith(f" {arg.name} ")


def test_labels_for_a_call(library, tokenizer):
    r = render(example(library), tokenizer, CFG)
    assert INTENTS[r.intent] == "call"
    assert r.slots == [0, STOP, STOP, STOP, STOP]
    assert r.dec_ids[0] == DEC and len(r.dec_ids) == len(r.dec_labels)
    full = tokenizer.decode(r.dec_ids + [EOS])
    assert full == (
        "<dec><call> set_lights<arg> room<val> kitchen<end><arg> on<val> false<end>"
        "<arg> brightness<val><absent><eos>"
    )
    # Only the values, <end> and <absent> are predicted; names and keys are copied.
    predicted = [t for t in r.dec_labels if t != IGNORE]
    assert tokenizer.decode(predicted) == " kitchen<end> false<end><absent>"


def test_shuffled_tools_keep_targets_consistent(library, tokenizer):
    calls = [
        Call("set_thermostat", {"temperature": 21}),
        Call("set_lights", {"room": "kitchen", "on": False}),
        Call("set_lights", {"room": "bedroom", "on": True, "brightness": 30}),
    ]
    ex = example(library, calls=calls)
    orders = set()
    for seed in range(20):
        r = render(ex, tokenizer, CFG, np.random.default_rng(seed))
        orders.add(tuple(r.tool_names))
        for call, slot in zip(calls, r.slots, strict=False):
            assert r.tool_names[slot] == call.tool
            assert tokenizer.decode(r.enc_ids[r.tool_pos[slot] + 1 :]).startswith(f" {call.tool} ")
        assert r.slots[3:] == [STOP, STOP]
        # Calls stay in their own order: the decoder doesn't depend on the tool order.
        assert r.dec_ids == render(ex, tokenizer, CFG).dec_ids
    assert len(orders) > 3


def test_other_intents(library, tokenizer):
    none = render(example(library, "none", []), tokenizer, CFG)
    assert none.slots == [STOP] * 5 and all(t == IGNORE for t in none.dec_labels)
    assert none.dec_ids == [DEC]

    unclear = render(example(library, "ask", []), tokenizer, CFG)
    assert unclear.slots == [IGNORE] * 5

    ask = render(
        example(library, "ask", [Call("set_lights", {"room": "?", "on": True})]), tokenizer, CFG
    )
    assert ask.slots[0] == ask.tool_names.index("set_lights")
    predicted = [t for t in ask.dec_labels if t != IGNORE]
    assert tokenizer.decode(predicted) == "<ask> true<end><absent>"


@pytest.mark.parametrize(
    "intent, calls",
    [
        ("call", []),
        ("none", [Call("set_lights", {"room": "kitchen", "on": False})]),
        ("call", [Call("lock_door", {"door": "front", "locked": True})]),  # not in the prompt
        ("call", [Call("set_lights", {"room": "garden", "on": False})]),
        ("call", [Call("set_lights", {"room": "?", "on": False})]),  # "?" only when asking
        ("maybe", []),
    ],
)
def test_invalid_examples(library, intent, calls):
    with pytest.raises(SchemaError):
        example(library, intent, calls)


def test_too_many_tools(library, tokenizer):
    ex = TaskExample("hi", tuple(list(library.values())[:6]), "none")
    with pytest.raises(ValueError):
        render(ex, tokenizer, CFG)


def test_json_round_trip(library):
    ex = example(library)
    assert TaskExample.from_json(ex.to_json(), library) == ex


def test_dataset_shuffles_per_epoch(library, tokenizer):
    ds = TaskDataset([example(library)] * 4, tokenizer, CFG)
    first = [tuple(ds[i].tool_names) for i in range(4)]
    assert first == [tuple(ds[i].tool_names) for i in range(4)]
    orders = set(first)
    for epoch in range(1, 10):
        ds.set_epoch(epoch)
        orders.add(tuple(ds[0].tool_names))
    assert len(orders) > 1


def test_task_collate(library, tokenizer):
    batch = [
        render(example(library), tokenizer, CFG),
        render(example(library, "none", [], tools=("pause_media",)), tokenizer, CFG),
    ]
    out = TaskCollator(CFG)(batch)
    longest = max(len(b.enc_ids) for b in batch)
    assert out["enc_ids"].shape == (2, bucket(longest, ENCODER_BUCKETS))
    assert out["dec_ids"].shape == (2, 64)
    assert out["enc_mask"].sum(1).tolist() == [len(b.enc_ids) for b in batch]
    assert (out["enc_ids"][~out["enc_mask"]] == PAD).all()
    assert (out["dec_labels"][~out["dec_mask"]] == IGNORE).all()
    assert out["tool_pos"].shape == (2, CFG.max_tools)
    assert out["tool_mask"].tolist() == [[True] * 3 + [False] * 2, [True] + [False] * 4]
    assert out["arg_pos"].shape == (2, CFG.max_tools, MAX_ARGS)
    assert out["arg_mask"][0, 0].sum() == len(library[batch[0].tool_names[0]].args)
    assert (out["enc_ids"].gather(1, out["tool_pos"])[out["tool_mask"]] == TOOL).all()
    assert out["intent"].tolist() == [0, 3]
    assert out["slots"].shape == (2, CFG.n_slots)


def test_pretrain_collate():
    batch = [
        {"enc_ids": [5] * 512, "dec_ids": [DEC] + [6] * 99, "dec_labels": [6] * 99 + [EOS]}
    ] * 3
    out = collate_pretrain(batch)
    assert out["enc_ids"].shape == (3, 512) and out["dec_ids"].shape == (3, 128)
    assert out["enc_mask"].all() and out["dec_mask"].sum() == 300


def test_bucket():
    assert bucket(1, (128, 256)) == 128
    assert bucket(129, (128, 256)) == 256
    with pytest.raises(ValueError):
        bucket(257, (128, 256))


def test_no_lone_space_before_markers(library, tokenizer):
    r = render(example(library), tokenizer, CFG)
    for p in [*r.tool_pos[1:], *(p for args in r.arg_pos for p in args)]:
        assert tokenizer.decode([r.enc_ids[p - 1]]) != " "
