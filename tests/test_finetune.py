import json

import pytest
import torch
import torch.nn.functional as F

from colibri.config import INTENTS, preset
from colibri.data.render import IGNORE, Call, TaskCollator, TaskDataset, TaskExample, render
from colibri.model.checkpoint import load_checkpoint, save_checkpoint
from colibri.model.colibri import Colibri, Questions
from colibri.train.finetune import (
    describe,
    finetune,
    freeze_body,
    load_init,
    question_loss,
    question_metrics,
)
from colibri.train.loop import TrainConfig

CFG = preset("tiny")


def examples(library) -> list[TaskExample]:
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
            "dim the bedroom lights to 30% and play some jazz",
            ["play_music", "lock_door", "set_lights"],
            "call",
            Call("set_lights", {"room": "bedroom", "on": True, "brightness": 30}),
            Call("play_music", {"query": "jazz"}),
        ),
        ex(
            "turn on the lights",
            ["set_lights", "set_blinds"],
            "ask",
            Call("set_lights", {"room": "?", "on": True}),
        ),
        ex("can you change it", ["set_thermostat", "set_blinds"], "ask"),  # no slot labels
        ex("order me a pizza", ["set_lights", "play_music", "lock_door"], "unsupported"),
        ex("thanks, that's all", ["set_volume"], "none"),
    ]


def make(cfg=CFG, seed=0) -> Colibri:
    torch.manual_seed(seed)
    return Colibri(cfg).eval()


@pytest.fixture
def batch(library, tokenizer):
    return TaskCollator(CFG)([render(e, tokenizer, CFG) for e in examples(library)])


class Oracle(Colibri):
    """Answers every question with the gold label (or a fixed wrong one), to test the metrics."""

    def __init__(self, batch, wrong_intent=False):
        super().__init__(CFG)
        intent = (batch["intent"] + 1) % len(INTENTS) if wrong_intent else batch["intent"]
        slots = batch["slots"].clamp(min=0)  # an IGNOREd slot answers tool 0
        self.answers = Questions(
            intent=F.one_hot(intent, len(INTENTS)).float() * 10,
            slots=F.one_hot(slots, CFG.max_tools + 1).float() * 10,
        )

    def ask_questions(self, *args) -> Questions:
        return self.answers


def test_loss_is_cross_entropy_over_labelled_answers(batch):
    model = make()
    loss = question_loss(model, batch)
    memory = model.encode(batch["enc_ids"], batch["enc_mask"])
    q = model.ask_questions(memory, batch["enc_mask"], batch["tool_pos"], batch["tool_mask"])
    torch.testing.assert_close(loss["intent"], F.cross_entropy(q.intent, batch["intent"]))
    labelled = batch["slots"] != IGNORE
    expected = F.cross_entropy(q.slots[labelled], batch["slots"][labelled])
    torch.testing.assert_close(loss["slots"], expected)


def test_slot_loss_is_zero_without_slot_labels(library, tokenizer):
    unclear = [e for e in examples(library) if e.intent == "ask" and not e.calls]
    b = TaskCollator(CFG)([render(e, tokenizer, CFG) for e in unclear])
    loss = question_loss(make(), b)
    assert loss["slots"].item() == 0.0 and torch.isfinite(loss["intent"])


def test_metrics_of_perfect_and_wrong_answers(batch):
    perfect = question_metrics(Oracle(batch), batch)
    for name in ("intent_acc", "slot_acc", "tools_acc", "exact_acc"):
        assert perfect[name].item() == 1.0, name
    assert perfect["call_stop_rate"].item() == 0.0

    wrong = question_metrics(Oracle(batch, wrong_intent=True), batch)
    assert wrong["intent_acc"].item() == 0.0 and wrong["exact_acc"].item() == 0.0
    assert wrong["tools_acc"].item() == 1.0  # the slots are still right
    # unsupported -> none, none -> call: one of 6 examples says CALL with STOP in slot 1
    assert wrong["call_stop_rate"].item() == pytest.approx(1 / 6)


def test_freeze_body_trains_only_the_question_block(batch):
    model = make().train()
    trainable = freeze_body(model)
    assert trainable == sum(p.numel() for p in model.questions.parameters())
    sum(question_loss(model, batch).values()).backward()
    assert all(
        p.grad is None for n, p in model.named_parameters() if not n.startswith("questions.")
    )
    assert all(p.grad is not None for p in model.questions.parameters())


def test_describe_shows_answers_and_gold(library, tokenizer):
    exs = examples(library)
    lines = describe(make(), exs, TaskDataset(exs, tokenizer, CFG), [0, 3, 5])
    assert lines[0].startswith("turn off the kitchen lights → ")
    assert lines[0].endswith("| gold call · set_lights")
    assert lines[1].endswith("| gold ask")  # tool unclear: no gold tools
    assert lines[2].endswith("| gold none")


def write_task(path, library) -> None:
    path.mkdir()
    rows = [e.to_json() for e in examples(library)]
    for name in ("train", "val", "heldout-templates"):
        (path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_finetune_frozen_then_full(library, tokenizer, tmp_path):
    torch.manual_seed(0)
    pretrained = Colibri(CFG)
    save_checkpoint(tmp_path / "pre.pt", pretrained, tokenizer, {"step": 1})
    write_task(tmp_path / "task", library)
    train_cfg = TrainConfig(steps=4, batch_size=3, warmup_steps=1, eval_every=2, log_every=1)

    val = finetune(
        train_cfg, tmp_path / "task", tmp_path / "frozen", "cpu", library,
        init=tmp_path / "pre.pt", freeze=True,
    )  # fmt: skip
    assert {"val_intent", "val_slots", "val_intent_acc", "val_tools_acc"} <= set(val)
    stage1 = load_checkpoint(tmp_path / "frozen" / "model.pt")
    assert stage1.meta["task"] == "questions" and stage1.meta["frozen"] is True
    body = {k: v for k, v in pretrained.state_dict().items() if not k.startswith("questions.")}
    for k, v in body.items():  # frozen: the body didn't move
        torch.testing.assert_close(stage1.model.state_dict()[k], v)

    records = [json.loads(line) for line in (tmp_path / "frozen" / "metrics.jsonl").open()]
    samples = [r for r in records if r["kind"] == "samples"]
    assert [r["step"] for r in samples] == [0, 2, 4]
    assert "intent_acc" in samples[-1]["heldout"]

    # Stage 2 starts from stage 1 with its trained question block, then trains everything.
    stage2 = load_init(tmp_path / "frozen" / "model.pt")
    for k, v in stage1.model.state_dict().items():
        torch.testing.assert_close(stage2.model.state_dict()[k], v)
    finetune(
        train_cfg, tmp_path / "task", tmp_path / "full", "cpu", library,
        init=tmp_path / "frozen" / "model.pt",
    )  # fmt: skip
    full = load_checkpoint(tmp_path / "full" / "model.pt").model.state_dict()
    assert not torch.allclose(
        full["encoder.0.attn.W_query.weight"], body["encoder.0.attn.W_query.weight"]
    )


def test_load_init_starts_a_new_question_block_from_pretraining(tokenizer, tmp_path):
    save_checkpoint(tmp_path / "pre.pt", make(seed=0), tokenizer, {"step": 1})
    torch.manual_seed(1)
    fresh = load_init(tmp_path / "pre.pt", dropout=0.0).model
    old = make(seed=0).questions.queries
    assert not torch.allclose(fresh.questions.queries, old)  # re-initialized, not loaded
    with pytest.raises(ValueError):
        load_init(tmp_path / "pre.pt", d_model=32)


def test_tools_are_reshuffled_each_epoch(library, tokenizer):
    exs = examples(library)
    data = TaskDataset(exs, tokenizer, CFG)
    orders = set()
    for epoch in range(6):
        data.set_epoch(epoch)
        orders.add(tuple(data[0].tool_names))
    assert len(orders) > 1
