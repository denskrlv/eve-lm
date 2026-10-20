import numpy as np
import pytest
import torch
import torch.nn.functional as F

from eve.config import preset
from eve.data.pretrain import SpanCorruptionDataset
from eve.data.render import IGNORE, collate_pretrain
from eve.model.checkpoint import load_checkpoint, load_pretrained, save_checkpoint
from eve.model.eve import Eve

CFG = preset("tiny")
TEXT = "Turn the kitchen lights off when you leave, and set the thermostat to 20 degrees. "


@pytest.fixture(scope="module")
def batch(tokenizer):
    tokens = np.array(tokenizer.encode(TEXT * 40), dtype=np.uint16)
    data = SpanCorruptionDataset(tokens, input_len=64)
    return collate_pretrain([data[i] for i in range(4)])


def lm_loss(model: Eve, batch) -> torch.Tensor:
    out = model(batch["enc_ids"], batch["enc_mask"], batch["dec_ids"], batch["dec_mask"])
    return F.cross_entropy(out.lm.flatten(0, 1), batch["dec_labels"].flatten(), ignore_index=IGNORE)


def val_loss(model: Eve, batch) -> float:
    model.eval()
    with torch.no_grad():
        return lm_loss(model, batch).item()


@pytest.fixture(scope="module")
def pretrained(tokenizer, batch, tmp_path_factory):
    """A tiny model trained a few steps, so its weights differ from any fresh init."""
    torch.manual_seed(0)
    model = Eve(CFG)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(5):
        opt.zero_grad()
        lm_loss(model, batch).backward()
        opt.step()
    path = tmp_path_factory.mktemp("ckpt") / "pretrain.pt"
    save_checkpoint(path, model, tokenizer, {"step": 5, "val_loss": val_loss(model, batch)})
    return path, model


def test_reloaded_checkpoint_gives_the_same_val_loss(pretrained, batch):
    path, _ = pretrained
    torch.manual_seed(123)  # a different init, so a load that did nothing would show
    ckpt = load_checkpoint(path)
    assert ckpt.meta["step"] == 5
    assert val_loss(ckpt.model, batch) == pytest.approx(ckpt.meta["val_loss"], rel=1e-6)


def test_config_and_tokenizer_come_back(pretrained, tokenizer):
    ckpt = load_checkpoint(pretrained[0])
    assert ckpt.model.cfg == CFG
    assert ckpt.tokenizer.merges == tokenizer.merges
    assert ckpt.tokenizer.encode(TEXT) == tokenizer.encode(TEXT)
    assert not (pretrained[0].parent / "pretrain.pt.tmp").exists()


@pytest.mark.parametrize("load", [load_checkpoint, load_pretrained])
def test_lm_head_stays_tied(pretrained, load):
    model = load(pretrained[0]).model
    assert model.lm_head.weight is model.embeddings.tok.weight


def test_load_pretrained_keeps_the_body_and_starts_questions_fresh(pretrained, batch):
    path, saved = pretrained
    torch.manual_seed(123)
    model = load_pretrained(path).model
    old, new = saved.state_dict(), model.state_dict()
    assert old.keys() == new.keys()
    for k in old:
        if k.startswith("questions."):
            continue
        assert torch.equal(old[k], new[k]), k
    assert not torch.equal(old["questions.queries"], new["questions.queries"])
    assert val_loss(model, batch) == pytest.approx(val_loss(saved, batch), rel=1e-6)


def test_load_pretrained_repeats_with_the_same_seed(pretrained):
    torch.manual_seed(7)
    a = load_pretrained(pretrained[0]).model.questions.queries
    torch.manual_seed(7)
    b = load_pretrained(pretrained[0]).model.questions.queries
    assert torch.equal(a, b)


def test_load_pretrained_can_reshape_the_question_block(pretrained):
    model = load_pretrained(pretrained[0], n_slots=4, dropout=0.1).model
    assert model.questions.queries.shape == (1 + 4, CFG.d_model)
    assert model.cfg.dropout == 0.1


def test_load_pretrained_refuses_to_change_the_body(pretrained):
    with pytest.raises(ValueError, match="d_model"):
        load_pretrained(pretrained[0], d_model=32)


def test_load_pretrained_refuses_missing_body_weights(pretrained, tmp_path):
    data = torch.load(pretrained[0], weights_only=True)
    key = next(k for k in data["state_dict"] if k.startswith("encoder."))
    del data["state_dict"][key]
    torch.save(data, tmp_path / "broken.pt")
    with pytest.raises(KeyError, match=key):
        load_pretrained(tmp_path / "broken.pt")


def test_load_pretrained_refuses_leftover_weights(pretrained, tmp_path):
    data = torch.load(pretrained[0], weights_only=True)
    data["state_dict"]["extra.weight"] = torch.zeros(3)
    torch.save(data, tmp_path / "extra.pt")
    with pytest.raises(KeyError, match="extra.weight"):
        load_pretrained(tmp_path / "extra.pt")


def test_rejects_files_that_are_not_checkpoints(tmp_path):
    torch.save({"weights": torch.zeros(3)}, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="not an Eve checkpoint"):
        load_checkpoint(tmp_path / "other.pt")


def test_save_refuses_a_tokenizer_bigger_than_the_vocab(tokenizer, tmp_path):
    model = Eve(preset("tiny", vocab_size=100))
    with pytest.raises(ValueError, match="tokenizer"):
        save_checkpoint(tmp_path / "x.pt", model, tokenizer)
