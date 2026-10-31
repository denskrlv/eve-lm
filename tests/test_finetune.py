import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from eve.config import INTENTS, preset
from eve.data.render import IGNORE, Call, TaskCollator, TaskDataset, TaskExample, render
from eve.data.schema import Tool
from eve.model.build import load_init
from eve.model.checkpoint import load_checkpoint, save_checkpoint
from eve.model.eve import Eve, Questions
from eve.tasks.questions import (
    describe,
    joint_loss,
    joint_metrics,
    outcomes,
    score,
    score_line,
    summary,
    tally,
    value_hits,
)
from eve.tokenizer.bpe import BPETokenizer
from eve.train.config import TrainConfig
from eve.train.finetune import finetune

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


class Writer(Oracle):
    """An Oracle whose decoder also writes every value right, except the ``wrong`` positions
    (example, decoder position)."""

    def __init__(self, batch, wrong=(), **kwargs):
        super().__init__(batch, **kwargs)
        labels = batch["dec_labels"].clone()
        for i, j in wrong:
            labels[i, j] = (labels[i, j] + 1) % CFG.vocab_size
        self.lm = F.one_hot(labels.clamp(min=0), CFG.vocab_size).float() * 10

    def decode(self, *args, **kwargs) -> torch.Tensor:
        return self.lm


def value_positions(batch, example):
    """The labelled decoder positions of one example of ``batch``."""
    return (batch["dec_labels"][example] != IGNORE).nonzero().flatten().tolist()


def test_loss_is_cross_entropy_over_labelled_answers(batch):
    model = make()
    loss = joint_loss(model, batch)
    memory = model.encode(batch["enc_ids"], batch["enc_mask"])
    q = model.ask_questions(memory, batch["enc_mask"], batch["tool_pos"], batch["tool_mask"])
    torch.testing.assert_close(loss["intent"], F.cross_entropy(q.intent, batch["intent"]))
    labelled = batch["slots"] != IGNORE
    expected = F.cross_entropy(q.slots[labelled], batch["slots"][labelled])
    torch.testing.assert_close(loss["slots"], expected)


def test_slot_loss_is_zero_without_slot_labels(library, tokenizer):
    unclear = [e for e in examples(library) if e.intent == "ask" and not e.calls]
    b = TaskCollator(CFG)([render(e, tokenizer, CFG) for e in unclear])
    loss = joint_loss(make(), b)
    assert loss["slots"].item() == 0.0 and torch.isfinite(loss["intent"])


def test_joint_loss_adds_the_value_tokens_cross_entropy(batch):
    model = make()
    loss = joint_loss(model, batch)
    assert set(loss) == {"intent", "slots", "values"}
    out = model(batch["enc_ids"], batch["enc_mask"], batch["dec_ids"], batch["dec_mask"])
    labels = batch["dec_labels"]
    expected = F.cross_entropy(out.lm[labels != IGNORE], labels[labels != IGNORE])
    torch.testing.assert_close(loss["values"], expected)


def test_value_loss_is_zero_without_calls(library, tokenizer):
    no_calls = [e for e in examples(library) if not e.calls]
    b = TaskCollator(CFG)([render(e, tokenizer, CFG) for e in no_calls])
    loss = joint_loss(make(), b)
    assert loss["values"].item() == 0.0 and torch.isfinite(loss["intent"])


def test_value_hits_counts_values_with_every_token_right(batch):
    # set_lights(room=kitchen, on=false, brightness absent): 3 values
    values, right = value_hits(Writer(batch).lm, batch["dec_labels"])
    assert values[0] == 3 and values.tolist() == right.tolist()
    assert values[3:].tolist() == [0, 0, 0]  # unclear ASK, UNSUPPORTED, NONE: no calls

    first, *_, last = value_positions(batch, 0)
    lm = Writer(batch, wrong=[(0, first), (0, last)]).lm  # kitchen's first token; <absent>
    values, right = value_hits(lm, batch["dec_labels"])
    assert values[0] == 3 and right[0] == 1


def test_joint_metrics_score_the_values(batch):
    perfect = joint_metrics(Writer(batch), batch)
    assert perfect["value_acc"].item() == perfect["full_acc"].item() == 1.0
    assert perfect["exact_acc"].item() == 1.0
    loss = perfect["intent"] + perfect["slots"] + perfect["values"]
    torch.testing.assert_close(perfect["loss"], loss)

    one_wrong = joint_metrics(Writer(batch, wrong=[(1, value_positions(batch, 1)[0])]), batch)
    n_values = (value_hits(Writer(batch).lm, batch["dec_labels"])[0]).sum().item()
    assert one_wrong["value_acc"].item() == pytest.approx(1 - 1 / n_values)
    assert one_wrong["exact_acc"].item() == 1.0
    assert one_wrong["full_acc"].item() == pytest.approx(5 / 6)


def test_score_counts_the_values(batch):
    wrong = [(1, value_positions(batch, 1)[0])]
    s = score(Writer(batch, wrong=wrong), [batch])
    assert s["exact_acc"] == 1.0 and s["full_acc"] == pytest.approx(5 / 6)
    assert 0.9 < s["value_acc"] < 1.0
    assert "values" in score_line(s) and "full" in score_line(s)


def test_metrics_of_perfect_and_wrong_answers(batch):
    perfect = joint_metrics(Writer(batch), batch)
    for name in ("intent_acc", "slot_acc", "tools_acc", "exact_acc"):
        assert perfect[name].item() == 1.0, name
    assert perfect["call_stop_rate"].item() == 0.0

    wrong = joint_metrics(Writer(batch, wrong_intent=True), batch)
    assert wrong["intent_acc"].item() == 0.0 and wrong["exact_acc"].item() == 0.0
    assert wrong["tools_acc"].item() == 1.0  # the slots are still right
    # unsupported -> none, none -> call: one of 6 examples says CALL with STOP in slot 1
    assert wrong["call_stop_rate"].item() == pytest.approx(1 / 6)


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
    """train.jsonl and the sets fine-tuning reads, all in ``path`` (its task and evals dir)."""
    path.mkdir()
    rows = [e.to_json() for e in examples(library)]
    for name in ("train", "val", "heldout-templates"):
        (path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_finetune_then_train_on(library, tokenizer, tmp_path):
    torch.manual_seed(0)
    pretrained = Eve(CFG)
    save_checkpoint(tmp_path / "pre.pt", pretrained, tokenizer, {"step": 1})
    write_task(tmp_path / "task", library)
    train_cfg = TrainConfig(steps=4, batch_size=3, warmup_steps=1, eval_every=2, log_every=1)

    val = finetune(
        train_cfg, tmp_path / "task", tmp_path / "first", "cpu", library,
        init=tmp_path / "pre.pt", evals_dir=tmp_path / "task",
    )  # fmt: skip
    assert {"val_values", "val_value_acc", "val_full_acc"} <= set(val)
    assert val["val_loss"] == pytest.approx(
        val["val_intent"] + val["val_slots"] + val["val_values"]
    )
    first = load_checkpoint(tmp_path / "first" / "model.pt")
    assert first.meta["task"] == "questions" and first.meta["joint"] is True
    assert first.meta["init"] == str(tmp_path / "pre.pt")
    body = {k: v for k, v in pretrained.state_dict().items() if not k.startswith("questions.")}
    assert not torch.allclose(
        first.model.state_dict()["encoder.0.attn.W_query.weight"],
        body["encoder.0.attn.W_query.weight"],
    )

    records = [json.loads(line) for line in (tmp_path / "first" / "metrics.jsonl").open()]
    samples = [r for r in records if r["kind"] == "samples"]
    assert [r["step"] for r in samples] == [0, 2, 4]
    assert {"macro_recall", "exact_acc", "value_acc", "full_acc"} <= set(
        samples[-1]["heldout"]
    ) & set(samples[-1]["val"])

    # A question run's checkpoint is loaded whole, its trained question block too.
    again = load_init(tmp_path / "first" / "model.pt")
    for k, v in first.model.state_dict().items():
        torch.testing.assert_close(again.model.state_dict()[k], v)
    finetune(
        train_cfg, tmp_path / "task", tmp_path / "second", "cpu", library,
        init=tmp_path / "first" / "model.pt", evals_dir=tmp_path / "task",
    )  # fmt: skip
    assert load_checkpoint(tmp_path / "second" / "model.pt").meta["joint"] is True


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
    answered = tally(outcomes(Oracle(batch, wrong_intent=True), [batch])).answered
    assert answered["call", "ask"] == 2 and answered["none", "call"] == 1
    assert "macro recall 0.0% (call 0% · ask 0%" in score_line(wrong)
    lines = summary({"val": wrong, "heldout": wrong, "samples": []})
    assert [label for label, _ in lines] == ["val", "held-out"]
