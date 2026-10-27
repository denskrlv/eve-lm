import json
from pathlib import Path

import pytest
import torch

import eve
from eve import Eve, ToolBox
from eve.api.toolbox import words
from eve.cli import main
from eve.config import preset
from eve.data.schema import Tool
from eve.model.checkpoint import save_checkpoint
from eve.model.eve import Eve as EveModel
from eve.tokenizer.bpe import BPETokenizer

HOME = ["set_lights", "set_thermostat", "play_music", "lock_door", "set_timer", "add_to_list"]


def set_lights(room: str, on: bool) -> str:
    """Turn the lights in a room on or off."""
    return f"{room} {on}"


def test_words_fold_plurals_and_split_names() -> None:
    assert words("Set_Lights in 2 rooms, batteries & glass") == [
        "set", "light", "in", "2", "room", "battery", "glass",
    ]  # fmt: skip


def test_adding_replacing_and_removing(library: dict[str, Tool]) -> None:
    box = ToolBox([library["set_lights"], [library["set_timer"], library["lock_door"]]])
    assert box.names == ["set_lights", "set_timer", "lock_door"] and len(box) == 3
    box.add(set_lights)  # the same name: the function takes the schema's place
    assert box.names == ["set_lights", "set_timer", "lock_door"]
    assert box.functions == {"set_lights": set_lights}
    assert box["set_lights"].args[0].type == "str"
    box.remove("set_timer")
    assert "set_timer" not in box and list(box) == [box["set_lights"], box["lock_door"]]
    with pytest.raises(KeyError, match="no tool called 'nope'"):
        box.remove("nope")


@pytest.mark.parametrize(
    ("text", "best"),
    [
        ("turn off the kitchen lights", "set_lights"),
        ("lock the back door", "lock_door"),
        ("put milk on my shopping list", "add_to_list"),
        ("set a timer for ten minutes", "set_timer"),
    ],
)
def test_search_ranks_the_matching_tool_first(
    library: dict[str, Tool], text: str, best: str
) -> None:
    box = ToolBox(library[name] for name in HOME)
    found = box.search(text, 3)
    assert found[0].name == best and len(found) == 3


def test_search_keeps_registration_order_on_ties(library: dict[str, Tool]) -> None:
    box = ToolBox(library[name] for name in HOME)
    assert [t.name for t in box.search("zzz", 2)] == HOME[:2]
    assert ToolBox().search("lights", 5) == []


def test_save_and_load_keep_the_schemas(library: dict[str, Tool], tmp_path: Path) -> None:
    box = ToolBox([set_lights, library["lock_door"]])
    box.save(tmp_path / "tools.json")
    loaded = ToolBox.load(tmp_path / "tools.json")
    assert list(loaded) == list(box) and loaded.functions == {}
    loaded.add(set_lights)  # functions are bound again by name
    assert loaded.functions == {"set_lights": set_lights}
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(eve.ToolError, match="can't read tools"):
        ToolBox.load(tmp_path / "bad.json")
    (tmp_path / "list.json").write_text(json.dumps([library["set_timer"].to_dict()]))
    assert ToolBox.load(tmp_path / "list.json").names == ["set_timer"]


@pytest.fixture
def model(tmp_path: Path, tokenizer: BPETokenizer) -> Eve:
    torch.manual_seed(0)
    path = tmp_path / "model.pt"
    save_checkpoint(path, EveModel(preset("tiny")), tokenizer, {"joint": True})
    return Eve.load(path)


def test_respond_picks_from_the_registered_tools(model: Eve, library: dict[str, Tool]) -> None:
    with pytest.raises(eve.ToolError, match="no tools"):
        model.respond("turn off the kitchen lights")
    model.add_tools([library[name] for name in HOME[1:]], set_lights)
    response = model.respond("turn off the kitchen lights")
    assert len(response.considered) == model.max_tools and response.considered[0] == "set_lights"
    assert response.functions == {"set_lights": set_lights}  # only the shown tools' functions
    given = model.respond("turn off the kitchen lights", tools=[library["set_timer"]])
    assert given.considered == ["set_timer"]  # tools= overrides the registry


def test_eve_ask_takes_more_tools_than_one_request_holds(
    tmp_path: Path, tokenizer: BPETokenizer, capsys: pytest.CaptureFixture[str]
) -> None:
    save_checkpoint(tmp_path / "m.pt", EveModel(preset("tiny")), tokenizer, {"joint": True})
    argv = ["ask", "lock the back door", "--tools", ",".join(HOME), "--json"]
    main([*argv, "--model", str(tmp_path / "m.pt"), "--device", "cpu"])
    response = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert len(response["considered"]) == 5 and response["considered"][0] == "lock_door"
