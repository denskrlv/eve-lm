import json
from pathlib import Path

import pytest
import torch

from eve.cli import main
from eve.config import INTENTS, preset
from eve.data.modules import VAL_SEED, in_batches, load_examples
from eve.data.render import Call, TaskCollator, TaskDataset, TaskExample
from eve.data.schema import Tool
from eve.evaluate import by_tool, evaluate_set, load_questions_model, set_files
from eve.model.checkpoint import model_file, save_checkpoint
from eve.model.eve import Eve
from eve.tasks.questions import TASK, Outcome, score
from eve.tokenizer.bpe import BPETokenizer

CFG = preset("tiny")


def _write_run(path: Path, tokenizer: BPETokenizer, task: str | None = TASK) -> Path:
    torch.manual_seed(0)
    meta = {"step": 3, "init": "pre.pt"} | ({"task": task} if task else {})
    save_checkpoint(path / "model.pt", Eve(CFG), tokenizer, meta)
    return path


def _write_task(path: Path, library: dict[str, Tool]) -> Path:
    rows = [
        TaskExample(
            "turn off the kitchen lights",
            (library["set_lights"], library["play_music"]),
            "call",
            (Call("set_lights", {"room": "kitchen", "on": False}),),
        ),
        TaskExample("order me a pizza", (library["set_lights"],), "unsupported"),
        TaskExample("thanks, that's all", (library["set_volume"],), "none"),
    ]
    lines = [r.to_json() | ({"group": "shared word"} if r.calls else {}) for r in rows]
    path.mkdir()
    (path / "val.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def test_a_run_directory_means_its_final_model(tmp_path: Path) -> None:
    assert model_file(tmp_path) == tmp_path / "model.pt"
    assert model_file(tmp_path / "last.pt") == tmp_path / "last.pt"


def test_a_pretraining_checkpoint_is_refused(tmp_path: Path, tokenizer: BPETokenizer) -> None:
    run = _write_run(tmp_path, tokenizer, task=None)
    with pytest.raises(ValueError, match="pretraining"):
        load_questions_model(run)


def test_missing_sets_are_named(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="probe.jsonl"):
        set_files(tmp_path, ["probe"])


def test_scores_are_the_training_evals_and_confusion_counts_every_example(
    tmp_path: Path, library: dict[str, Tool], tokenizer: BPETokenizer
) -> None:
    ckpt = load_questions_model(_write_run(tmp_path / "run", tokenizer))
    path = set_files(_write_task(tmp_path / "task", library), ["val"])["val"]
    result = evaluate_set(ckpt, path, library, batch_size=2)

    data = TaskDataset(load_examples(path, library), tokenizer, CFG, seed=VAL_SEED)
    assert result.scores == score(ckpt.model, in_batches(data, TaskCollator(CFG), 2))
    assert result.examples == sum(sum(said.values()) for said in result.confusion.values()) == 3
    assert list(result.confusion) == list(INTENTS)
    assert sum(result.confusion["call"].values()) == 1
    ((n, scores),) = result.groups.values()  # only the CALL row has a group
    assert list(result.groups) == ["shared word"] and n == 1 and "recall_call" in scores
    assert "recall_none" not in scores


def test_by_tool_counts_an_example_for_each_of_its_tools(library: dict[str, Tool]) -> None:
    lights, music = (
        Call("set_lights", {"room": "kitchen", "on": True}),
        Call("play_music", {"query": "jazz"}),
    )
    tools = (library["set_lights"], library["play_music"])
    examples = [
        TaskExample("lights and jazz", tools, "call", (lights, music)),
        TaskExample("lights on", tools, "call", (lights, lights)),  # one tool, counted once
        TaskExample("lights", tools, "ask", (lights,)),
        TaskExample("order a pizza", tools, "unsupported"),  # no gold tool
    ]
    answers = [
        Outcome("call", "call", has_slots=True, tools_ok=True),
        Outcome("call", "call", has_slots=True, tools_ok=False),
        Outcome("ask", "unsupported", has_slots=True, tools_ok=False),
        Outcome("unsupported", "unsupported", has_slots=True, tools_ok=True),
    ]
    results = {result.tool: result for result in by_tool(examples, answers)}
    assert [result.tool for result in by_tool(examples, answers)] == ["set_lights", "play_music"]
    assert results["play_music"].examples == 1 and results["play_music"].exact_acc == 1.0
    lights_result = results["set_lights"]
    assert lights_result.examples == 3
    assert lights_result.intent_acc == pytest.approx(2 / 3)
    assert lights_result.exact_acc == pytest.approx(1 / 3)
    assert lights_result.answered == {"call": 2, "ask": 0, "unsupported": 1, "none": 0}


def test_evaluate_prints_json(
    tmp_path: Path,
    library: dict[str, Tool],
    tokenizer: BPETokenizer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = _write_run(tmp_path / "run", tokenizer)
    task = _write_task(tmp_path / "task", library)
    argv = ["evaluate", str(run), "--evals", str(task), "--sets", "val", "--device", "cpu"]
    main([*argv, "--json"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert set(out) == {"val"} and out["val"]["examples"] == 3
    assert {"macro_recall", "exact_acc", "value_acc", "full_acc"} <= set(out["val"]["scores"])
    assert "by_tool" not in out["val"]
    assert out["val"]["groups"]["shared word"]["examples"] == 1

    main([*argv, "--json", "--by-tool"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert [row["tool"] for row in out["val"]["by_tool"]] == ["set_lights"]

    main([*argv, "--by-tool"])
    assert "val by tool" in capsys.readouterr().out
