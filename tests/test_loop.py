import json
import math

import numpy as np
import pytest
import torch

from colibri.config import preset
from colibri.data.pretrain import SpanCorruptionDataset
from colibri.data.render import collate_pretrain
from colibri.model.checkpoint import load_checkpoint
from colibri.model.colibri import Colibri
from colibri.tokenizer.bpe import BYTE0
from colibri.train.config import TrainConfig
from colibri.train.loop import batch_indices, lr_at, param_groups, train
from colibri.train.pretrain import span_loss

CFG = preset("tiny", dropout=0.1)  # dropout on, so resuming must restore the RNG too
CPU = torch.device("cpu")


def corpus(n_tokens: int = 4000, seed: int = 0) -> SpanCorruptionDataset:
    tokens = np.random.default_rng(seed).integers(BYTE0, 600, n_tokens).astype(np.uint16)
    return SpanCorruptionDataset(tokens, input_len=64)


def small_run(**overrides) -> TrainConfig:
    kw = dict(steps=6, batch_size=4, warmup_steps=2, eval_every=3, eval_batches=2)
    kw |= dict(save_every=3, log_every=1)
    return TrainConfig(**kw | overrides)


def run(out, tokenizer, cfg=None, train_data=None, **kwargs) -> list[dict]:
    torch.manual_seed(0)
    model = Colibri(CFG)
    train(
        model,
        span_loss,
        train_data or corpus(),
        corpus(seed=1),
        collate_pretrain,
        cfg or small_run(),
        out,
        tokenizer,
        CPU,
        **kwargs,
    )
    return [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]


def train_losses(records: list[dict]) -> dict[int, float]:
    return {r["step"]: r["lm"] for r in records if r["kind"] == "train"}


def test_lr_warms_up_then_decays_to_the_floor():
    cfg = TrainConfig(steps=1000, lr=1e-3, warmup_steps=100, min_lr_ratio=0.1)
    lrs = [lr_at(s, cfg) for s in range(1000)]
    assert lrs[0] == pytest.approx(1e-5)
    assert lrs[99] == pytest.approx(1e-3)
    assert all(a <= b for a, b in zip(lrs[:100], lrs[1:100], strict=False))
    assert all(a >= b for a, b in zip(lrs[99:], lrs[100:], strict=False))
    assert lr_at(1000, cfg) == pytest.approx(1e-4)
    assert lr_at(550, cfg) == pytest.approx((1e-3 + 1e-4) / 2)  # halfway down the cosine


def test_weight_decay_only_on_linear_weights():
    model = Colibri(CFG)
    decay, no_decay = (g["params"] for g in param_groups(model, 0.1))
    names = {id(p): n for n, p in model.named_parameters()}
    assert len(decay) + len(no_decay) == len(list(model.parameters()))
    decayed = {names[id(p)] for p in decay}
    assert "encoder.0.attn.W_query.weight" in decayed
    assert "questions.W_q.weight" in decayed
    for name in names.values():
        if any(s in name for s in ("scale", "shift", "bias", "embeddings.", "queries", "stop")):
            assert name not in decayed, name
    assert model.lm_head.weight in no_decay  # tied to the token embeddings


def test_batches_cover_each_epoch_once_in_a_new_order():
    cfg = TrainConfig(steps=100, batch_size=4, seed=3)
    epoch0 = np.concatenate([batch_indices(s, 10, cfg) for s in range(2)])  # 2 whole batches
    epoch1 = np.concatenate([batch_indices(s, 10, cfg) for s in range(2, 4)])
    assert len(set(epoch0)) == 8 and len(set(epoch1)) == 8
    assert not np.array_equal(epoch0, epoch1)
    assert np.array_equal(batch_indices(5, 10, cfg), batch_indices(5, 10, cfg))


def test_loss_goes_down_when_overfitting_one_batch(tokenizer, tmp_path):
    cfg = small_run(steps=60, lr=3e-3, warmup_steps=5, eval_every=60, save_every=60)
    losses = train_losses(run(tmp_path, tokenizer, cfg, train_data=corpus(300)))
    assert len(corpus(300)) == 4  # exactly one batch, seen every step
    assert losses[60] < losses[1] / 2


def test_first_val_loss_is_about_ln_vocab(tokenizer, tmp_path):
    first_val = next(r for r in run(tmp_path, tokenizer) if r["kind"] == "val")
    assert first_val["step"] == 0
    assert first_val["val_lm"] == pytest.approx(math.log(CFG.vocab_size), abs=0.2)


def test_resumed_run_matches_an_uninterrupted_one(tokenizer, tmp_path):
    straight = run(tmp_path / "a", tokenizer)

    run(tmp_path / "b", tokenizer, stop_after=3)
    assert not (tmp_path / "b" / "model.pt").exists()  # not finished
    with (tmp_path / "b" / "metrics.jsonl").open("a") as f:  # as if it crashed after step 3
        f.write(json.dumps({"kind": "train", "step": 4, "lm": 0.0}) + "\n")
    resumed = run(tmp_path / "b", tokenizer, resume=True)

    assert train_losses(resumed) == train_losses(straight)  # exactly, step by step
    assert [r["step"] for r in resumed if r["kind"] == "train"] == [1, 2, 3, 4, 5, 6]
    a = load_checkpoint(tmp_path / "a" / "model.pt").model.state_dict()
    b = load_checkpoint(tmp_path / "b" / "model.pt").model.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_resume_refuses_a_different_schedule(tokenizer, tmp_path):
    run(tmp_path, tokenizer, stop_after=3)
    with pytest.raises(ValueError, match="lr"):
        run(tmp_path, tokenizer, small_run(lr=5e-4), resume=True)
    run(tmp_path, tokenizer, small_run(log_every=2), resume=True)  # logging may change


def test_last_checkpoint_holds_the_train_state(tokenizer, tmp_path):
    run(tmp_path, tokenizer, stop_after=3)
    ckpt = load_checkpoint(tmp_path / "last.pt")
    assert ckpt.train_state["step"] == 3 and ckpt.meta["step"] == 3
    assert ckpt.meta["train_config"] == small_run().to_dict()
    assert "val_lm" in ckpt.meta  # step 3 was an eval step


def test_datasets_are_told_the_epoch(tokenizer, tmp_path):
    class Epochs(SpanCorruptionDataset):
        def set_epoch(self, epoch: int) -> None:
            self.seen.append(epoch)

    data = corpus()
    data.__class__, data.seen = Epochs, []
    per_epoch = len(data) // 4
    run(tmp_path, tokenizer, small_run(steps=per_epoch + 2), train_data=data)
    assert data.seen == [0] * per_epoch + [1, 1]


def test_compile_runs_the_losses_through_the_compiled_model(tokenizer, tmp_path, monkeypatch):
    import colibri.train.loop as loop

    seen = {"loss": set(), "on_eval": set()}

    class Wrapped(torch.nn.Module):  # stands in for torch.compile's wrapper
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, *args, **kwargs):
            return self.model(*args, **kwargs)

    monkeypatch.setattr(loop.torch, "compile", Wrapped)

    def loss_fn(m, batch):
        seen["loss"].add(type(m).__name__)
        return span_loss(m, batch)

    def on_eval(m):
        seen["on_eval"].add(type(m).__name__)
        return {}

    torch.manual_seed(0)
    model = Colibri(CFG)
    train(
        model, loss_fn, corpus(), corpus(seed=1), collate_pretrain, small_run(compile=True),
        tmp_path, tokenizer, CPU, on_eval=on_eval,
    )  # fmt: skip
    assert seen == {"loss": {"Wrapped"}, "on_eval": {"Colibri"}}
    ckpt = load_checkpoint(tmp_path / "model.pt")  # saved from the plain model: no prefixes
    assert ckpt.model.state_dict().keys() == model.state_dict().keys()
