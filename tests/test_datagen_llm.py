import io
import json
import subprocess
import urllib.error
from pathlib import Path

import numpy as np
import pytest

from eve.data.datagen import llm
from eve.data.datagen.llm import (
    LLMError,
    claude_generate,
    ollama_embed,
    ollama_generate,
    ollama_unload,
    parse_json,
)

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}}


class FakeHTTP:
    """Stands in for urlopen: records each request body, answers from ``replies`` in turn."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.sent = []

    def __call__(self, request, timeout=None):
        self.sent.append((request.full_url, json.loads(request.data)))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return io.BytesIO(json.dumps(reply).encode())


def test_ollama_generate_sends_the_schema_and_reads_speed(monkeypatch):
    http = FakeHTTP({"response": '{"a": "x"}', "eval_count": 50, "eval_duration": 2_000_000_000})
    monkeypatch.setattr(llm.urllib.request, "urlopen", http)
    reply = ollama_generate("m")("sys", "hi", SCHEMA)
    url, body = http.sent[0]
    assert url.endswith("/api/generate")
    assert body["format"] == SCHEMA and body["think"] is False and body["stream"] is False
    assert body["system"] == "sys" and body["prompt"] == "hi" and body["model"] == "m"
    assert reply.data == {"a": "x"} and reply.tokens == 50 and reply.seconds == 2.0


def test_ollama_embed_normalizes_and_batches(monkeypatch):
    def answer(n):
        return {"embeddings": [[3.0, 4.0]] * n}

    http = FakeHTTP(*[answer(llm.EMBED_BATCH)] * 4, answer(300 - 4 * llm.EMBED_BATCH))
    monkeypatch.setattr(llm.urllib.request, "urlopen", http)
    vectors = ollama_embed("e")([f"t{i}" for i in range(300)])
    assert len(http.sent) == 5 and http.sent[0][0].endswith("/api/embed")
    assert vectors.shape == (300, 2)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert ollama_embed("e")([]).shape[0] == 0


def test_ollama_down_raises_llm_error(monkeypatch):
    http = FakeHTTP(urllib.error.URLError("Connection refused"))
    monkeypatch.setattr(llm.urllib.request, "urlopen", http)
    with pytest.raises(LLMError, match="ollama serve"):
        ollama_generate("m")("s", "p", SCHEMA)


def test_ollama_unload_sends_keep_alive_zero(monkeypatch):
    http = FakeHTTP({})
    monkeypatch.setattr(llm.urllib.request, "urlopen", http)
    ollama_unload("m")
    assert http.sent[0][1] == {"model": "m", "keep_alive": 0}


def test_claude_generate_runs_without_tools_or_project(monkeypatch):
    seen = {}

    def run(args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        out = {"result": "{}", "structured_output": {"a": "y"}, "total_cost_usd": 0.5}
        return subprocess.CompletedProcess(args, 0, json.dumps(out), "")

    monkeypatch.setattr(llm.subprocess, "run", run)
    reply = claude_generate("opus", budget=3.0)("sys", "msg", SCHEMA)
    args = seen["args"]
    assert args[:2] == ["claude", "-p"]
    assert args[args.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in args and "--no-session-persistence" in args
    assert json.loads(args[args.index("--json-schema") + 1]) == SCHEMA
    assert args[args.index("--max-budget-usd") + 1] == "3.0"
    assert args[args.index("--model") + 1] == "opus"
    assert seen["kwargs"]["input"] == "msg"
    assert Path(seen["kwargs"]["cwd"]).resolve() != Path.cwd().resolve()
    assert reply.data == {"a": "y"} and reply.cost == 0.5


def test_claude_failure_raises_llm_error(monkeypatch):
    def run(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, "", "not logged in")

    monkeypatch.setattr(llm.subprocess, "run", run)
    with pytest.raises(LLMError, match="not logged in"):
        claude_generate()("s", "m", SCHEMA)


def test_parse_json():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! Here it is: {"a": {"b": 2}} hope that helps') == {"a": {"b": 2}}
    assert parse_json("no json here") is None
    assert parse_json('{"a": ') is None


def test_ollama_generate_falls_back_when_structured_output_is_unavailable(monkeypatch):
    unavailable = urllib.error.HTTPError(
        "u", 501, "no", {}, io.BytesIO(b'{"error":"structured output is unavailable"}')
    )
    ok = {"response": 'Here: {"a": "x"}', "eval_count": 5, "eval_duration": 1_000_000_000}
    http = FakeHTTP(unavailable, ok, ok)
    monkeypatch.setattr(llm.urllib.request, "urlopen", http)
    generate = ollama_generate("m")
    assert generate("sys", "hi", SCHEMA).data == {"a": "x"}
    assert generate("sys", "hi", SCHEMA).data == {"a": "x"}
    first, second, third = (body for _, body in http.sent)
    assert "format" in first and "format" not in second and "format" not in third
    assert json.dumps(SCHEMA) in second["system"]
