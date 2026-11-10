import re
from pathlib import Path

import numpy as np
import pytest

from eve.cli import build_parser
from eve.data.datagen.llm import LLMError, Reply
from eve.data.datagen.run import Models, load_context, report_rows, run
from eve.data.datagen.shard import TOOLS_PER_SHARD, Shard, claim

ROOT = Path(__file__).parents[1]
TOPICS = [f"area{i} > sub{i}" for i in range(20)]
CONSTRAINTS = {
    "seeds": ["lid", "sock", "frost", "kettle"],
    "speaker": ["a nurse"],
    "situation": ["driving"],
    "form": ["question"],
    "length": ["3-6 words"],
    "tone": ["casual"],
}
WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike".split()


AXIS: dict[str, int] = {}


def fake_embed(texts):
    """Every distinct text gets its own direction, and texts with the same number in them share
    one more: the descriptions of the made-up tool 7 are alike, different tools are not."""
    rows = np.zeros((len(texts), 4096), dtype=np.float32)
    for i, text in enumerate(texts):
        rows[i, AXIS.setdefault(text, len(AXIS))] = 1.0
        if number := re.search(r"\d+", text):
            rows[i, AXIS.setdefault("#" + number.group(), len(AXIS))] = 1.0
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


class ToolMaker:
    """Invents distinct, valid tools: zz1_a, zz2_a, …"""

    def __init__(self):
        self.n = 0
        self.calls = 0

    def __call__(self, system, prompt, schema):
        self.calls += 1
        wanted = int(prompt.split("Write ")[1].split()[0])
        tools = []
        for _ in range(wanted):
            self.n += 1
            tools.append(
                {
                    "names": [f"zz{self.n}_a", f"zz{self.n}_b"],
                    "descriptions": [f"Quux {self.n} wibble", f"Florp {self.n} zonk"],
                    "args": [{"name": "thing", "type": "str"}],
                }
            )
        return Reply("", {"tools": tools}, 10, 1.0)


class RequestWriter:
    """Answers every card with a clean request whose value is a fresh word."""

    def __init__(self, fail=False):
        self.fail, self.n, self.calls = fail, 0, 0

    def __call__(self, system, prompt, schema):
        self.calls += 1
        if self.fail:
            raise LLMError("ollama is down")
        rows = []
        for line in prompt.splitlines():
            if line.startswith('{"card"'):
                n = int(line.split(":")[1].split(",")[0])
                self.n += 1
                word = f"{WORDS[self.n % len(WORDS)]}{self.n}"
                intent = line.split('"intent": "')[1].split('"')[0]
                args = None if intent == "near" else {"thing": "?" if intent == "ask" else word}
                rows.append({"card": n, "utterance": f"please sort {word} out", "args": args})
        return Reply("", {"rows": rows}, 20, 1.0)


def reviewer(fixes=()):
    def review(system, prompt, schema):
        return Reply(
            "",
            {"fixes": [], "drop_rows": [{"id": i, "why": "x"} for i in fixes], "drop_tools": []},
            cost=0.3,
        )

    return review


@pytest.fixture(scope="module")
def context():
    return load_context(fake_embed, ROOT / "data/tools", ROOT / "data/novel-tools")


def models(tools=None, requests=None, review=None):
    return Models(
        tools or ToolMaker(), requests or RequestWriter(), review or reviewer(), fake_embed, "fake"
    )


def test_run_makes_and_approves_a_shard(tmp_path, context):
    made = run(tmp_path, 1, models(), TOPICS, CONSTRAINTS, context)
    shard = made[0]
    assert shard.status() == "approved"
    assert len(shard.tools()) == TOOLS_PER_SHARD
    rows = shard.rows()
    assert len(rows) == TOOLS_PER_SHARD * 20
    assert {r.intent for r in rows} == {"call", "ask", "unsupported"}
    report = shard.read_json("report.json")
    assert report["approved"] and report["review"]["cost"] == 0.3
    assert report["rows"]["kept"] == len(rows)
    assert not (shard.path / ".lock").exists()
    assert report_rows(tmp_path)[0]["status"] == "approved"


def test_resume_skips_finished_stages(tmp_path, context):
    with pytest.raises(LLMError):
        run(tmp_path, 1, models(review=_failing), TOPICS, CONSTRAINTS, context)
    shard = Shard(tmp_path / "0001")
    assert shard.status() == "requests"
    tools, requests = ToolMaker(), RequestWriter()
    run(tmp_path, 0, models(tools, requests), TOPICS, CONSTRAINTS, context)
    assert tools.calls == 0 and requests.calls == 0
    assert shard.status() == "approved"


def _failing(system, prompt, schema):
    raise LLMError("claude is down")


def test_run_with_a_failing_model_leaves_the_shard_resumable(tmp_path, context):
    with pytest.raises(LLMError):
        run(tmp_path, 1, models(requests=RequestWriter(fail=True)), TOPICS, CONSTRAINTS, context)
    shard = Shard(tmp_path / "0001")
    assert shard.status() == "tools" and not (shard.path / ".lock").exists()
    run(tmp_path, 0, models(), TOPICS, CONSTRAINTS, context)
    assert shard.status() == "approved"


def test_high_fix_rate_flags_the_shard(tmp_path, context):
    drops = [f"t{t:02d}-r{r:02d}" for t in range(1, 6) for r in range(1, 21)]  # 100 of 500
    shard = run(tmp_path, 1, models(review=reviewer(drops)), TOPICS, CONSTRAINTS, context)[0]
    assert shard.status() == "flagged"
    assert shard.read_json("report.json")["review"]["fix_rate"] == 0.2


def test_a_shard_held_by_another_process_is_skipped(tmp_path, context):
    with pytest.raises(LLMError):
        run(tmp_path, 1, models(requests=RequestWriter(fail=True)), TOPICS, CONSTRAINTS, context)
    held = Shard(tmp_path / "0001")
    (held.path / ".lock").write_text("1")  # pid 1 is alive
    made = run(tmp_path, 1, models(), TOPICS, CONSTRAINTS, context)
    assert [s.number for s in made] == [2]
    assert held.status() == "tools" and not claim(held)


def test_new_shard_tools_never_repeat_earlier_names(tmp_path, context):
    maker = ToolMaker()
    run(tmp_path, 2, models(tools=maker), TOPICS, CONSTRAINTS, context)
    names = [n for p in ("0001", "0002") for t in Shard(tmp_path / p).tools() for n in t.names]
    assert len(names) == len(set(names)) == 2 * TOOLS_PER_SHARD * 2


def test_a_topic_too_close_to_an_eval_tool_is_skipped(tmp_path, context):
    blocked = context.blocked_tools[0].description  # the fake embed: same text, same direction
    topics = [blocked] + TOPICS[1:]
    shard = run(tmp_path, 1, models(), topics, CONSTRAINTS, context)[0]
    assert len(shard.tools()) == TOOLS_PER_SHARD - TOOLS_PER_SHARD // 5
    assert shard.read_json("stats.json")["tools"]["blocked_topics"] == 1
    assert report_rows(tmp_path)[0]["tools_dropped"] == 0


def test_cli_parses_datagen_run():
    args = build_parser().parse_args(
        ["datagen", "run", "--shards", "3", "--requests-model", "qwen3.5:4b"]
    )
    assert args.shards == 3 and args.requests_model == "qwen3.5:4b" and args.parallel == 2
    args = build_parser().parse_args(["datagen", "report"])
    assert args.dir.endswith("shards")


def test_report_command_prints_every_shard(tmp_path, context, capsys, monkeypatch):
    from eve.cli import main
    from eve.progress import console

    monkeypatch.setattr(console, "width", 200)
    run(tmp_path, 1, models(), TOPICS, CONSTRAINTS, context)
    main(["datagen", "report", "--dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "0001" in out and "approved" in out and "tools dropped" in out, out
