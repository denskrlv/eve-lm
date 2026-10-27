import json
import sys
from pathlib import Path

import pytest

from eve.cli import main
from eve.data.generate import headless
from eve.data.generate.headless import (
    GenerateError,
    Reply,
    batch_message,
    claude_writer,
    generate,
    parse_documents,
    pending,
    read_documents,
    write_batch,
)
from eve.data.generate.tinytools import Lexicon, PromptSpec
from eve.data.jsonl import read_jsonl, write_jsonl
from eve.data.schema import load_library

ROOT = Path(__file__).parents[1]
SCENE = (
    "Nora came home late and the hallway was dark. She had bought a new lamp last week, so she "
    'said to the speaker by the door, "make the hallway brighter, I can\'t see my keys." The '
    "assistant turned on the hallway lamp and set it to the brightest level. Nora found her keys "
    'under a pile of letters and laughed. "Thanks," she said, "you can dim it again now." The '
    "assistant lowered the light to a soft glow, and Nora went to make a cup of tea."
)


@pytest.fixture(scope="module")
def lexicon() -> Lexicon:
    return Lexicon.load(ROOT / "data" / "tinytools")


@pytest.fixture(scope="module")
def heldout():
    return list(load_library(ROOT / "data" / "tools" / "heldout").values())


def specs(n: int) -> list[PromptSpec]:
    return [
        PromptSpec(f"tt-{i:06d}", "scene", "lights", "lamp", "buy", "bright", "", "", f"prompt {i}")
        for i in range(n)
    ]


def asked(message: str) -> list[dict]:
    return [json.loads(line) for line in message.splitlines() if line.startswith("{")]


class Fake:
    """A generator that writes SCENE for every prompt, except ``bad`` ids on their first try."""

    def __init__(self, bad: set[str] = frozenset(), fail: bool = False) -> None:
        self.bad, self.fail, self.messages = set(bad), fail, []

    def __call__(self, system: str, message: str) -> Reply:
        self.messages.append(message)
        if self.fail:
            raise GenerateError("not logged in")
        rows = []
        for row in asked(message):
            text = "The lamp was bright." if row["id"] in self.bad else SCENE
            self.bad.discard(row["id"])
            rows.append(json.dumps({"id": row["id"], "text": text}))
        return Reply("```json\n" + "\n".join(rows) + "\n```", cost=0.01)


def test_parse_skips_fences_chatter_and_other_ids() -> None:
    reply = (
        '```\nHere you go:\n{"id": "a", "text": "x"},\n{"id": "z", "text": "y"}\n{"id": "b"\n```'
    )
    assert parse_documents(reply, {"a", "b"}) == {"a": "x"}


def test_a_rewrite_carries_the_text_and_the_reasons() -> None:
    message = batch_message(specs(2), {"tt-000001": ("too short", ["12 words"])})
    rows = asked(message)
    assert "because" in message.splitlines()[0]
    assert rows[0] == {"id": "tt-000000", "prompt": "prompt 0"}
    assert rows[1]["rejected"] == "too short" and rows[1]["because"] == ["12 words"]


def test_rejected_documents_go_back_with_their_reasons(lexicon, heldout) -> None:
    fake = Fake(bad={"tt-000001"})
    result = write_batch(specs(3), fake, lexicon, heldout, rounds=3)
    assert result.calls == 2 and result.accepted == 3 and not result.rejected
    assert [row["id"] for row in asked(fake.messages[1])] == ["tt-000001"]
    assert "words; a scene needs" in asked(fake.messages[1])[0]["because"][0]
    assert result.cost == pytest.approx(0.02)


def test_a_document_still_bad_after_the_rounds_is_kept_as_rejected(lexicon, heldout) -> None:
    result = write_batch(specs(2), Fake(bad={"tt-000000"}), lexicon, heldout, rounds=1)
    assert result.accepted == 1 and list(result.rejected) == ["tt-000000"]
    assert result.documents["tt-000000"] == "The lamp was bright."


def test_failed_calls_use_up_the_rounds(lexicon) -> None:
    result = write_batch(specs(2), Fake(fail=True), lexicon, rounds=2)
    assert result.calls == 0 and len(result.errors) == 2 and not result.documents


def test_pending_skips_written_documents_unless_rewriting(lexicon, heldout) -> None:
    documents = {"tt-000000": SCENE, "tt-000001": "The lamp was bright."}
    todo, previous = pending(specs(3), documents, lexicon, heldout)
    assert [s.id for s in todo] == ["tt-000002"] and previous == {}
    todo, previous = pending(specs(3), documents, lexicon, heldout, rewrite=True)
    assert [s.id for s in todo] == ["tt-000001", "tt-000002"]
    assert previous["tt-000001"][0] == "The lamp was bright."


def test_generate_saves_batches_and_resumes(tmp_path: Path, lexicon, heldout) -> None:
    run, all_specs = tmp_path / "run", specs(7)
    write_jsonl(run / "prompts.jsonl", (s.to_json() for s in all_specs))
    seen = []
    results = generate(run, all_specs[:5], Fake(), lexicon, heldout, batch_size=2,
                       on_batch=seen.append)  # fmt: skip
    assert [r.number for r in results] == [0, 1, 2] and len(seen) == 3
    assert len(read_documents(run / "documents.jsonl")) == 5
    assert len(read_jsonl(run / "batches" / "batch-0002.jsonl")) == 1

    documents = read_documents(run / "documents.jsonl")
    todo, _ = pending(all_specs, documents, lexicon, heldout)
    results = generate(run, todo, Fake(), lexicon, heldout, batch_size=2)
    assert [r.number for r in results] == [3]
    ids = [row["id"] for row in read_jsonl(run / "documents.jsonl")]
    assert ids == [s.id for s in all_specs]  # one line each, in prompt order


def test_a_rewrite_replaces_the_old_line(tmp_path: Path, lexicon, heldout) -> None:
    run = tmp_path / "run"
    write_jsonl(run / "prompts.jsonl", (s.to_json() for s in specs(2)))
    write_jsonl(run / "documents.jsonl", [{"id": "tt-000000", "text": "The lamp was bright."}])
    todo, previous = pending(specs(2), read_documents(run / "documents.jsonl"), lexicon,
                             heldout, rewrite=True)  # fmt: skip
    fake = Fake()
    generate(run, todo, fake, lexicon, heldout, previous)
    assert "rejected" in asked(fake.messages[0])[0]
    assert read_jsonl(run / "documents.jsonl") == [
        {"id": "tt-000000", "text": SCENE},
        {"id": "tt-000001", "text": SCENE},
    ]


def test_generate_stops_when_nothing_comes_back(tmp_path: Path, lexicon) -> None:
    run = tmp_path / "run"
    write_jsonl(run / "prompts.jsonl", (s.to_json() for s in specs(20)))
    with pytest.raises(GenerateError, match="not logged in"):
        generate(run, specs(20), Fake(fail=True), lexicon, batch_size=2, workers=1, rounds=1)
    started = len(list((run / "batches").glob("*.prompts.jsonl")))
    assert started < 10  # the batches still queued are cancelled


def fake_claude(tmp_path: Path, body: str) -> str:
    script = tmp_path / "claude"
    script.write_text(f"#!{sys.executable}\nimport json, sys\n{body}\n")
    script.chmod(0o755)
    return str(script)


def test_claude_writer_reads_the_json_result(tmp_path: Path) -> None:
    command = fake_claude(
        tmp_path,
        "args = sys.argv[1:]\n"
        "assert args[args.index('--tools') + 1] == '' and args[args.index('--model') + 1] == 'm'\n"
        "print(json.dumps({'result': sys.stdin.read().upper(), 'total_cost_usd': 0.5,"
        " 'is_error': False}))",
    )
    assert claude_writer(command, "m")("rules", "hi") == Reply("HI", 0.5)


def test_claude_writer_raises_on_errors(tmp_path: Path) -> None:
    failed = fake_claude(tmp_path, "print(json.dumps({'result': 'rate limit', 'is_error': True}))")
    with pytest.raises(GenerateError, match="rate limit"):
        claude_writer(failed)("rules", "hi")
    crashed = fake_claude(tmp_path, "sys.exit('no such model')")
    with pytest.raises(GenerateError, match="no such model"):
        claude_writer(crashed)("rules", "hi")


def test_eve_corpus_generate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = tmp_path / "run"
    main(["corpus", "prompts", "--n", "6", "--out", str(run)])
    monkeypatch.setattr(headless, "claude_writer", lambda *_: Fake())
    main(["corpus", "generate", str(run), "--limit", "4", "--batch", "2"])
    out = capsys.readouterr().out
    assert "eve corpus generate" in out and "eve corpus check" in out
    assert len(read_documents(run / "documents.jsonl")) == 4
    assert (run / "tinytools.parquet").exists()

    with pytest.raises(SystemExit, match="no .*prompts.jsonl"):
        main(["corpus", "generate", str(tmp_path / "empty")])


def test_the_rules_name_the_heldout_tools(heldout) -> None:
    rules = headless.system_prompt(heldout)
    assert "convert_units" in rules and "search_flights" in rules and "{heldout}" not in rules
