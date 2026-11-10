import subprocess
import sys

import pytest

from eve.data.datagen.shard import (
    GenTool,
    Row,
    Shard,
    claim,
    new_shard,
    release,
    shards,
    write_atomic,
)
from eve.data.schema import Arg, SchemaError

PEST = {
    "id": "t01",
    "topic": "garden > pests",
    "names": ["identify_garden_pest", "name_plant_bug"],
    "descriptions": ["Say which insect is damaging a plant", "Tell what bug eats the leaves"],
    "args": [
        {"name": "symptom", "type": "str"},
        {"name": "plant", "type": "str", "required": False},
    ],
}
TOPICS = [f"area{i} > sub{i}" for i in range(15)]


def test_gen_tool_round_trips_and_builds_every_variant():
    tool = GenTool.from_dict(PEST)
    assert GenTool.from_dict(tool.to_dict()) == tool
    assert tool.capability == "identify_garden_pest"
    variant = tool.tool(1, 1)
    assert variant.name == "name_plant_bug"
    assert variant.description == "Tell what bug eats the leaves"
    assert variant.domain == "garden"
    assert tool.args == (Arg("symptom", "str"), Arg("plant", "str", required=False))
    with pytest.raises(SchemaError):
        GenTool.from_dict({**PEST, "args": [{"name": "Bad Name", "type": "str"}]})
    with pytest.raises(SchemaError):
        GenTool.from_dict({**PEST, "names": []})


def test_row_example_renames_the_call():
    tool = GenTool.from_dict(PEST)
    call = Row("t01-r01", "t01", "my kale has holes", "call", {"symptom": "holes"}, {}, "m")
    assert Row.from_dict(call.to_dict()) == call
    example = call.example(tool, name=1)
    assert example.calls[0].tool == "name_plant_bug"
    assert example.calls[0].args == {"symptom": "holes"}
    assert [t.name for t in example.tools] == ["name_plant_bug"]
    ask = Row("t01-r02", "t01", "what's eating my plant", "ask", {"symptom": "?"}, {}, "m")
    assert ask.example(tool).calls[0].args == {"symptom": "?"}
    near = Row("t01-r03", "t01", "how do I grow kale", "unsupported", None, {}, "m")
    assert near.example(tool).calls == ()


def test_new_shards_get_next_numbers_and_their_own_topics(tmp_path):
    made = [new_shard(tmp_path, TOPICS) for _ in range(3)]
    assert [s.number for s in made] == [1, 2, 3]
    assert [s.status() for s in made] == ["new"] * 3
    seen = [set(s.topics()) for s in made]
    assert all(len(t) == 5 for t in seen)
    assert not (seen[0] & seen[1]) and not (seen[1] & seen[2])
    with pytest.raises(ValueError, match="no unused topics"):
        new_shard(tmp_path, TOPICS)
    assert [s.number for s in shards(tmp_path)] == [1, 2, 3]


def test_new_shard_skips_a_number_taken_meanwhile(tmp_path):
    new_shard(tmp_path, TOPICS)
    (tmp_path / "0002").mkdir()  # another process made it but hasn't written it yet
    assert new_shard(tmp_path, TOPICS).number == 3


def test_claim_is_exclusive_and_stale_locks_are_taken(tmp_path):
    shard = new_shard(tmp_path, TOPICS)
    assert claim(shard)
    assert not claim(Shard(shard.path))
    release(shard)
    done = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True
    )
    (shard.path / ".lock").write_text(done.stdout.strip())  # that process is gone
    assert claim(shard)


def test_write_atomic_leaves_no_temp_file(tmp_path):
    write_atomic(tmp_path / "a.json", "{}")
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]
    assert (tmp_path / "a.json").read_text() == "{}"


def test_shard_saves_tools_rows_and_status(tmp_path):
    shard = new_shard(tmp_path, TOPICS)
    tool = GenTool.from_dict(PEST)
    row = Row(
        "t01-r01", "t01", "my kale has holes", "call", {"symptom": "holes"}, {"form": "x"}, "m"
    )
    shard.save_tools([tool])
    shard.save_rows([row])
    shard.set_status("requests")
    again = Shard(shard.path)
    assert again.tools() == [tool]
    assert again.rows() == [row]
    assert again.status() == "requests"
    with pytest.raises(ValueError):
        shard.set_status("done")
