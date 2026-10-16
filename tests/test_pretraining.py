import json

import numpy as np
import torch

from colibri.config import preset
from colibri.model.checkpoint import load_pretrained
from colibri.model.colibri import Colibri
from colibri.tokenizer.bpe import BYTE0
from colibri.train.loop import TrainConfig
from colibri.train.pretrain import SAMPLES, fill_in, pretrain, steps_for


def test_fill_in_keeps_the_text_and_fills_each_hole(tokenizer):
    torch.manual_seed(0)
    model = Colibri(preset("tiny")).train()
    out = fill_in(model, tokenizer, "Students who <x0> their homework <x1> better.", max_new=8)
    assert out.startswith("Students who [") and out.endswith("] better.")
    assert " their homework [" in out
    assert model.training  # put back the way it was


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
    assert ckpt.model.cfg == cfg
