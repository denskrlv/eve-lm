import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from eve.config import INTENTS, preset
from eve.data.render import IGNORE, Call, TaskCollator, TaskDataset, TaskExample, render
from eve.data.schema import Tool
from eve.model.checkpoint import load_checkpoint, save_checkpoint
from eve.model.eve import Eve, Questions
from eve.tokenizer.bpe import BPETokenizer
from eve.train.config import TrainConfig
from eve.train.finetune import (
    describe,
    finetune,
    freeze_body,
    load_init,
    question_loss,
    question_metrics,
    score,
    score_line,
)

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


def make(cfg=CFG, seed=0) -> Eve:
    torch.manual_seed(seed)
    return Eve(cfg).eval()


@pytest.fixture
def batch(library, tokenizer):
    return TaskCollator(CFG)([render(e, tokenizer, CFG) for e in examples(library)])


class Oracle(Eve):
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


def test_describe_copies_its_batch_blocking(
    library: dict[str, Tool], tokenizer: BPETokenizer, copy_flags: list[bool]
) -> None:
    exs = examples(library)
    describe(make(), exs, TaskDataset(exs, tokenizer, CFG), [0, 3])
    assert copy_flags and not any(copy_flags)


def test_scratch_needs_a_tokenizer_path(library: dict[str, Tool], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scratch needs tokenizer_path"):
        finetune(TrainConfig(steps=1), tmp_path, tmp_path / "out", "cpu", library, scratch=CFG)


def write_task(path, library) -> None:
    path.mkdir()
    rows = [e.to_json() for e in examples(library)]
    for name in ("train", "val", "heldout-templates"):
        (path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_finetune_frozen_then_full(library, tokenizer, tmp_path):
    torch.manual_seed(0)
    pretrained = Eve(CFG)
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
    assert {"macro_recall", "exact_acc"} <= set(samples[-1]["heldout"]) & set(samples[-1]["val"])

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


def test_score_counts_recall_per_intent(batch):
    s = score(Oracle(batch), [batch])
    assert s["macro_recall"] == s["tools_acc"] == s["exact_acc"] == 1.0
    assert set(s) >= {f"recall_{i}" for i in INTENTS}

    # The 6 examples: call, call, ask, ask, unsupported, none. Shifting every intent by one
    # makes each one wrong, but the slots stay right.
    wrong = score(Oracle(batch, wrong_intent=True), [batch, batch])  # counts add up
    assert wrong["macro_recall"] == 0.0 and wrong["exact_acc"] == 0.0
    assert wrong["tools_acc"] == 1.0
    assert "macro recall 0.0% (call 0% · ask 0%" in score_line("val", wrong)
