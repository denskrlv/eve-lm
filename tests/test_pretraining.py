import json

import numpy as np
import pytest
import torch

from colibri.config import preset
from colibri.data.render import collate_pretrain
from colibri.model.checkpoint import load_pretrained
from colibri.model.colibri import Colibri
from colibri.tokenizer.bpe import BYTE0, DEC, EOS, sentinel
from colibri.train.loop import TrainConfig
from colibri.train.pretrain import (
    SAMPLES,
    fill_in,
    pretrain,
    prompt_ids,
    span_loss,
    span_metrics,
    steps_for,
)


def test_fill_in_keeps_the_text_and_fills_each_hole(tokenizer):
    torch.manual_seed(0)
    model = Colibri(preset("tiny")).train()
    out = fill_in(model, tokenizer, "Students who <x0> their homework <x1> better.", max_new=8)
    assert out.startswith("Students who") and out.endswith("] better.")
    assert out.count("[") == 2 and "] their homework" in out
    assert model.training  # put back the way it was


def test_prompt_drops_the_lone_space_before_a_hole(tokenizer):
    ids = prompt_ids(tokenizer, "turn off the <x0> lights")
    assert ids == [*tokenizer.encode("turn off the"), sentinel(0), *tokenizer.encode(" lights")]


def test_content_metrics_skip_sentinels_and_eos():
    torch.manual_seed(0)
    model = Colibri(preset("tiny")).eval()

    def item(enc, target):
        return {"enc_ids": enc, "dec_ids": [DEC, *target], "dec_labels": [*target, EOS]}

    x0, x1 = sentinel(0), sentinel(1)
    batch = collate_pretrain(  # the second is shorter, so there is padding too
        [item([300, x0, 301, x1], [x0, 400, 401, x1, 402]), item([302, x0], [x0, 403])]
    )
    with torch.no_grad():
        m = span_metrics(model, batch)
        logits = model(batch["enc_ids"], batch["enc_mask"], batch["dec_ids"], batch["dec_mask"]).lm
    words = batch["dec_labels"] >= BYTE0
    assert int(words.sum()) == 4  # 400, 401, 402, 403
    expected = torch.nn.functional.cross_entropy(logits[words], batch["dec_labels"][words])
    assert m["content"].item() == pytest.approx(expected.item(), rel=1e-5)
    assert m["lm"].item() == pytest.approx(span_loss(model, batch)["lm"].item(), rel=1e-5)
    assert 0.0 <= m["content_acc"].item() <= 1.0


def test_steps_for_one_pass_over_the_corpus():
    assert steps_for(250_000_000, 32) == 250_000_000 // (32 * 569)


def test_pretrain_end_to_end(tokenizer, tmp_path):
    rng = np.random.default_rng(0)
    for name, n in [("train", 3000), ("val", 1000)]:
        rng.integers(BYTE0, 600, n).astype(np.uint16).tofile(tmp_path / f"{name}.bin")
    tokenizer.save(tmp_path / "tok.json")
    cfg = preset("tiny", max_encoder_len=128)  # the smallest encoder bucket
    train_cfg = TrainConfig(steps=4, batch_size=2, warmup_steps=1, eval_every=2, eval_batches=2)

    val = pretrain(cfg, train_cfg, tmp_path, tmp_path / "tok.json", tmp_path / "run", "cpu")

    records = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").open()]
    samples = [r for r in records if r["kind"] == "samples"]
    assert [r["step"] for r in samples] == [0, 2, 4]
    assert len(samples[-1]["samples"]) == len(SAMPLES)
    ckpt = load_pretrained(tmp_path / "run" / "model.pt")  # box 7 reads what box 4 wrote
    assert ckpt.meta["val_lm"] == val["val_lm"]
    assert {"val_lm", "val_content", "val_content_acc"} <= set(val)
    assert ckpt.model.cfg == cfg
