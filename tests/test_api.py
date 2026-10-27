import json
import subprocess
import sys
from pathlib import Path

import huggingface_hub
import pytest
import torch

import eve
from eve import Eve, EveError, Response
from eve.config import INTENTS, preset
from eve.data.schema import Tool
from eve.infer.predict import Answers, Value, WrittenCall
from eve.model.checkpoint import save_checkpoint
from eve.model.eve import Eve as EveModel
from eve.tokenizer.bpe import BPETokenizer


@pytest.fixture
def checkpoint(tmp_path: Path, tokenizer: BPETokenizer) -> Path:
    torch.manual_seed(0)
    path = tmp_path / "joint" / "model.pt"
    meta = {"task": "questions", "joint": True, "val_exact_acc": 0.5}
    save_checkpoint(path, EveModel(preset("tiny")), tokenizer, meta)
    return path


def set_lights(room: str, on: bool, brightness: int | None = None) -> str:
    """Turn the lights in a room on or off."""
    return f"{room} {'on' if on else 'off'}"


def test_load_and_respond(checkpoint: Path, library: dict[str, Tool]) -> None:
    model = Eve.load(checkpoint.parent)  # a directory holding model.pt works too
    assert model.max_tools == 5 and model.info["metrics"] == {"val_exact_acc": 0.5}
    response = model.respond("turn off the kitchen lights", [set_lights, library["set_timer"]])
    assert response.intent in INTENTS and response.confidence == response.intents[response.intent]
    assert sum(response.intents.values()) == pytest.approx(1.0)
    assert response.familiar_tools is False  # an uncalibrated model knows no tools
    if response.intent in ("call", "ask"):
        assert all(c.tool in ("set_lights", "set_timer") for c in response.calls)
    else:
        assert response.calls == []
    json.dumps(response.to_dict())  # plain values only


def test_respond_refuses_what_it_cant_read(checkpoint: Path, library: dict[str, Tool]) -> None:
    model = Eve.load(checkpoint)
    six = list(library.values())[:6]
    with pytest.raises(eve.ToolError, match="6 tools; Eve reads at most 5"):
        model.respond("hi", six)
    with pytest.raises(eve.ToolError, match="give at least one tool"):
        model.respond("hi", [])
    with pytest.raises(eve.ToolError, match="two tools are called 'set_lights'"):
        model.respond("hi", [set_lights, set_lights])
    with pytest.raises(EveError, match="empty"):
        model.respond("  ", [set_lights])
    with pytest.raises(EveError, match="encoder input is"):
        model.respond("lights " * 600, [set_lights])


def test_load_refuses_a_model_that_cant_write_calls(
    tmp_path: Path, tokenizer: BPETokenizer
) -> None:
    save_checkpoint(tmp_path / "pre.pt", EveModel(preset("tiny")), tokenizer, {"task": "span"})
    with pytest.raises(eve.CheckpointError, match="isn't a fine-tuned joint model"):
        Eve.load(tmp_path / "pre.pt")
    (tmp_path / "junk.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(eve.CheckpointError, match="can't load"):
        Eve.load(tmp_path / "junk.pt")
    with pytest.raises(eve.CheckpointError, match="no checkpoint at"):
        Eve.load(tmp_path / "missing" / "model.pt")


def test_a_hub_repo_id_is_downloaded(checkpoint: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asked = []

    def download(repo: str, filename: str, revision: str | None = None) -> str:
        asked.append((repo, filename, revision))
        return str(checkpoint)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert Eve.load("someone/eve-joint-base", revision="v1").source == "someone/eve-joint-base"
    assert asked == [("someone/eve-joint-base", "model.pt", "v1")]

    def offline(*args: object, **kwargs: object) -> str:
        raise OSError("no network")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    with pytest.raises(eve.CheckpointError, match="Hub download failed: no network"):
        Eve.load("someone/eve-joint-base")


def answers(intent: str, values: dict[str, Value], prob: float = 0.9) -> Answers:
    intents = {name: (prob if name == intent else (1 - prob) / 3) for name in INTENTS}
    written = [WrittenCall("set_lights", prob, values)] if intent in ("call", "ask") else []
    slots = [{"set_lights": prob, "STOP": 1 - prob}]
    return Answers(intents, slots, written)


GOOD = {"room": Value("value", "kitchen", 0.99), "on": Value("value", False, 0.98),
        "brightness": Value("absent", prob=0.97)}  # fmt: skip


def test_a_response_reads_the_answers() -> None:
    response = Response.from_answers("lights off", answers("call", GOOD), True, {})
    (call,) = response.calls
    assert call.args == {"room": "kitchen", "on": False} and call.missing == []
    assert call.arg_confidence == {"room": 0.99, "on": 0.98, "brightness": 0.97}
    asked = Response.from_answers("lights", answers("ask", GOOD | {"room": Value("ask")}), True, {})
    assert asked.missing == [("set_lights", "room")]
    unsupported = Response.from_answers("book a flight", answers("unsupported", {}), True, {})
    assert unsupported.calls == [] and unsupported.to_dict()["calls"] == []


def test_run_calls_the_functions_only_when_all_is_in_place() -> None:
    functions = {"set_lights": set_lights}
    good = Response.from_answers("lights off", answers("call", GOOD), True, functions)
    assert good.run() == ["kitchen off"]
    with pytest.raises(EveError, match="confidence 90% < 95%"):
        good.run(min_confidence=0.95)

    cases = [
        (answers("ask", GOOD | {"room": Value("ask")}), functions, "the intent is 'ask'"),
        (answers("call", GOOD | {"room": Value("ask")}), functions,
         r"no valid value for \['room'\]"),
        (answers("call", GOOD | {"on": Value("invalid", "maybe", 0.5, "on: bad")}), functions,
         r"no valid value for \['on'\]"),
        (answers("call", GOOD), {}, "was given as a schema, not a function"),
    ]  # fmt: skip
    for given, funcs, reason in cases:
        with pytest.raises(EveError, match=reason):
            Response.from_answers("lights", given, True, funcs).run()


def test_the_library_never_imports_the_training_dependencies(checkpoint: Path) -> None:
    script = (
        "import sys\n"
        "from eve import Eve\n"
        "def f(room: str):\n"
        "    '''Turn the lights on.'''\n"
        f"Eve.load({str(checkpoint)!r}).respond('lights on', [f])\n"
        "print(sorted({m.split('.')[0] for m in sys.modules} & {'pyarrow', 'rich'}))\n"
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


def test_a_response_reads_as_one_line() -> None:
    call = Response.from_answers("lights off", answers("call", GOOD), True, {})
    assert str(call) == "call 90%: set_lights(room='kitchen', on=False)"
    assert repr(call) == "<Response call 90%: set_lights(room='kitchen', on=False)>"
    bad = GOOD | {"room": Value("ask"), "on": Value("invalid", "maybe", 0.5, "on: bad")}
    asked = Response.from_answers("lights", answers("ask", bad), True, {})
    assert str(asked) == "ask 90%: set_lights() missing room invalid on"
    assert repr(Response.from_answers("thanks", answers("none", {}), True, {})) == (
        "<Response none 90%>"
    )
