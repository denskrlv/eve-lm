from pathlib import Path

import pytest
import torch
from helpers import save_joint, save_retriever

from eve import Eve
from eve.export import export_model
from eve.tokenizer.bpe import BPETokenizer


def test_export_writes_a_half_precision_model_that_loads(
    tmp_path: Path, tokenizer: BPETokenizer
) -> None:
    source = save_joint(tmp_path / "joint.pt", tokenizer)
    out = export_model(source, tmp_path / "out", "float16", "me/eve")

    saved = torch.load(out / "model.pt", weights_only=True)
    assert {t.dtype for t in saved["state_dict"].values()} == {torch.float16}
    assert saved["meta"]["dtype"] == "float16"
    assert saved["meta"]["exported_from"] == str(source)
    card = (out / "README.md").read_text()
    assert "me/eve" in card and "intent accuracy" in card
    Eve.load(out)


def test_export_ships_a_retriever_beside_the_model(tmp_path: Path, tokenizer: BPETokenizer) -> None:
    source = save_joint(tmp_path / "joint.pt", tokenizer)
    retriever = save_retriever(tmp_path / "retriever-run.pt", tokenizer)
    out = export_model(source, tmp_path / "out", retriever=retriever)

    assert (out / "retriever.pt").is_file()
    assert Eve.load(out).info["scorer"] == "Dense"


def test_export_refuses_what_it_cannot_publish(tmp_path: Path, tokenizer: BPETokenizer) -> None:
    joint = save_joint(tmp_path / "joint.pt", tokenizer)
    not_joint = save_joint(tmp_path / "old.pt", tokenizer, {"task": "questions"})

    with pytest.raises(ValueError, match="dtype must be one of"):
        export_model(joint, tmp_path / "out", "int8")
    with pytest.raises(ValueError, match="isn't a fine-tuned joint model"):
        export_model(not_joint, tmp_path / "out")
    with pytest.raises(ValueError, match="isn't a retriever"):
        export_model(joint, tmp_path / "out", retriever=joint)
