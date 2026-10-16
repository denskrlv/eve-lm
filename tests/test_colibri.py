import pytest
import torch
import torch.nn.functional as F

from colibri.config import preset
from colibri.data.render import IGNORE, Call, TaskCollator, TaskExample, render
from colibri.model.colibri import Colibri

CFG = preset("tiny")
INPUTS = ("enc_ids", "enc_mask", "dec_ids", "dec_mask", "tool_pos", "tool_mask")


def make(cfg=CFG, seed=0) -> Colibri:
    torch.manual_seed(seed)
    return Colibri(cfg).eval()


def examples(library) -> list[TaskExample]:
    """Eight small examples, one of each kind the heads have to tell apart."""

    def ex(utterance, tools, intent, *calls):
        return TaskExample(utterance, tuple(library[t] for t in tools), intent, calls)

    return [
        ex(
            "turn off the kitchen lights",
            ["set_lights", "set_thermostat", "play_music"],
            "call",
            Call("set_lights", {"room": "kitchen", "on": False}),
        ),
        ex(
            "set the thermostat to 21 degrees",
            ["set_blinds", "set_thermostat"],
            "call",
            Call("set_thermostat", {"temperature": 21}),
        ),
        ex(
            "dim the bedroom lights to 30% and play some jazz",
            ["play_music", "lock_door", "set_lights"],
            "call",
            Call("set_lights", {"room": "bedroom", "on": True, "brightness": 30}),
            Call("play_music", {"query": "jazz"}),
        ),
        ex(  # fan-out: one call per door
            "lock all the doors",
            ["lock_door", "set_lights"],
            "call",
            Call("lock_door", {"door": "front", "locked": True}),
            Call("lock_door", {"door": "back", "locked": True}),
            Call("lock_door", {"door": "garage", "locked": True}),
        ),
        ex(
            "turn on the lights",
            ["set_lights", "set_blinds"],
            "ask",
            Call("set_lights", {"room": "?", "on": True}),
        ),
        ex("can you change it", ["set_thermostat", "set_blinds"], "ask"),  # tool unclear
        ex("order me a pizza", ["set_lights", "play_music", "lock_door"], "unsupported"),
        ex("thanks, that's all", ["set_volume"], "none"),
    ]


@pytest.fixture
def batch(library, tokenizer):
    return TaskCollator(CFG)([render(e, tokenizer, CFG) for e in examples(library)])


def forward(model, batch):
    return model(**{k: batch[k] for k in INPUTS})


def task_losses(out, batch) -> dict[str, torch.Tensor]:
    """The fine-tuning loss of M7–M8: one cross-entropy per question family."""
    q = out.questions
    return {
        "intent": F.cross_entropy(q.intent, batch["intent"]),
        "slots": F.cross_entropy(
            q.slots.flatten(0, 1), batch["slots"].flatten(), ignore_index=IGNORE
        ),
        "values": F.cross_entropy(
            out.lm.flatten(0, 1), batch["dec_labels"].flatten(), ignore_index=IGNORE
        ),
    }


def test_output_shapes(batch):
    out = forward(make(), batch)
    b, n_dec = batch["dec_ids"].shape
    assert out.questions.intent.shape == (b, CFG.n_intents)
    assert out.questions.slots.shape == (b, CFG.n_slots, CFG.max_tools + 1)
    assert out.lm.shape == (b, n_dec, CFG.vocab_size)


def test_pretraining_needs_no_tools(batch):
    model = make()
    out = model(batch["enc_ids"], batch["enc_mask"], batch["dec_ids"], batch["dec_mask"])
    assert out.questions is None and out.lm.shape[-1] == CFG.vocab_size


@pytest.mark.parametrize("name", ["tiny", "small", "base"])
def test_parameter_count_matches_config(name):
    cfg = preset(name)
    counts = Colibri(cfg).parameter_counts()
    assert counts["total"] == cfg.approx_params()
    assert counts["total"] == sum(v for k, v in counts.items() if k != "total")


def test_lm_head_is_tied_to_embeddings():
    model = make()
    assert model.lm_head.weight is model.embeddings.tok.weight


def test_padded_tools_get_zero_probability(batch):
    probs = forward(make(), batch).questions.slots.softmax(-1)
    tools = torch.cat([batch["tool_mask"], torch.ones(len(probs), 1, dtype=torch.bool)], dim=1)
    assert (probs[~tools[:, None, :].expand_as(probs)] == 0).all()
    torch.testing.assert_close(probs.sum(-1), torch.ones(probs.shape[:2]))


def test_slot_head_points_at_tool_positions(batch):
    """Listing the same <tool> markers in another order permutes the slot answers the same way:
    the answers are the prompt's tools, not fixed classes."""
    model = make()
    out = forward(model, batch).questions.slots
    order = torch.tensor([2, 0, 1, 4, 3])
    permuted = batch | {
        "tool_pos": batch["tool_pos"][:, order],
        "tool_mask": batch["tool_mask"][:, order],
    }
    out_permuted = forward(model, permuted).questions.slots
    torch.testing.assert_close(out_permuted[..., :-1], out[..., order])
    torch.testing.assert_close(out_permuted[..., -1], out[..., -1])  # STOP


def test_encoder_padding_changes_nothing(library, tokenizer):
    model = make()
    one = TaskCollator(CFG)([render(examples(library)[0], tokenizer, CFG)])
    padded = dict(one)
    extra = 40
    padded["enc_ids"] = F.pad(one["enc_ids"], (0, extra), value=0)
    padded["enc_mask"] = F.pad(one["enc_mask"], (0, extra), value=False)
    a, b = forward(model, one), forward(model, padded)
    torch.testing.assert_close(a.questions.intent, b.questions.intent)
    torch.testing.assert_close(a.questions.slots, b.questions.slots)
    torch.testing.assert_close(a.lm, b.lm)


def test_decoder_is_causal(batch):
    model = make()
    out = forward(model, batch).lm
    changed = batch | {"dec_ids": batch["dec_ids"].clone()}
    changed["dec_ids"][:, 6:] = 7
    torch.testing.assert_close(forward(model, changed).lm[:, :6], out[:, :6])


def test_sdpa_gives_the_same_outputs(batch):
    model = make()
    manual = forward(model, batch)
    model.set_sdpa(True)
    fast = forward(model, batch)
    torch.testing.assert_close(manual.questions.slots, fast.questions.slots)
    torch.testing.assert_close(manual.lm, fast.lm)


def test_initial_logits_are_small(batch):
    """With GPT-2's init the untrained model is unsure: no logit far from 0."""
    out = forward(make(), batch)
    assert out.lm.abs().max() < 1.0
    assert out.questions.intent.abs().max() < 1.0


def test_gradients_reach_every_parameter(batch):
    model = make().train()
    sum(task_losses(forward(model, batch), batch).values()).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None or not p.grad.any()]
    assert not missing


def test_tiny_model_overfits_eight_examples(batch):
    """The end-to-end wiring check of M3: if every part is connected right, a tiny model can
    memorize 8 examples. Book ch. 5 does the same with one short text before real training."""
    model = make().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
    for _ in range(200):
        losses = task_losses(forward(model, batch), batch)
        optimizer.zero_grad()
        sum(losses.values()).backward()
        # Without clipping, some seeds spike back up at lr 2e-3 (App. D clips at 1.0 too).
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = forward(model, batch)
        losses = task_losses(out, batch)
    assert all(loss < 0.05 for loss in losses.values()), losses

    assert (out.questions.intent.argmax(-1) == batch["intent"]).all()
    slots = batch["slots"] != IGNORE
    assert (out.questions.slots.argmax(-1)[slots] == batch["slots"][slots]).all()
    values = batch["dec_labels"] != IGNORE
    assert (out.lm.argmax(-1)[values] == batch["dec_labels"][values]).all()
