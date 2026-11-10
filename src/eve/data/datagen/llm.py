"""The models datagen talks to: Ollama (local generation and embeddings) and headless Claude.

Each is wrapped as a plain callable (``Generate``, ``Embed``), so the stages can be tested with
fakes and never need a model running.
"""

import json
import re
import subprocess
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from eve import defaults

EMBED_BATCH = 64  # Ollama 0.40 fails some batches of 256 real descriptions ("tokenize: EOF")


class LLMError(RuntimeError):
    """A model call failed: the server is down, it timed out, or it answered with an error."""


@dataclass(frozen=True)
class Reply:
    """What a model wrote, parsed when it is JSON, and what it cost."""

    text: str
    data: Any = None  # the parsed JSON, or None
    tokens: int = 0  # tokens written
    seconds: float = 0.0  # time spent writing them
    cost: float = 0.0  # US dollars (Claude only)


Generate = Callable[[str, str, dict[str, Any]], Reply]  # (system, prompt, JSON schema) -> Reply
Embed = Callable[[Sequence[str]], np.ndarray]  # texts -> unit vectors, one row each


def parse_json(text: str) -> Any:
    """The JSON in ``text``: code fences and chatter around the outermost ``{...}`` are
    ignored. None if there is none that parses."""
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _post(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data: dict[str, Any] = json.loads(response.read())
            return data
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:300]
        raise LLMError(f"{url} answered {error.code}: {detail}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise LLMError(
            f"can't reach Ollama at {url} ({error}): is `ollama serve` running?"
        ) from None


def ollama_generate(
    model: str,
    url: str = defaults.OLLAMA_URL,
    temperature: float = 0.9,
    timeout: float = 600,
) -> Generate:
    """A ``Generate`` on an Ollama model, with thinking off and the reply held to the schema."""

    def generate(system: str, prompt: str, schema: dict[str, Any]) -> Reply:
        body = {
            "model": model,
            "system": system,
            "prompt": prompt,
            "format": schema,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature},
        }
        out = _post(f"{url}/api/generate", body, timeout)
        if "error" in out:
            raise LLMError(f"ollama {model}: {out['error']}")
        text = str(out.get("response", ""))
        tokens = int(out.get("eval_count", 0))
        return Reply(text, parse_json(text), tokens, int(out.get("eval_duration", 0)) / 1e9)

    return generate


def ollama_embed(model: str = defaults.EMBED_MODEL, url: str = defaults.OLLAMA_URL) -> Embed:
    """An ``Embed`` on an Ollama embedding model; vectors are scaled to length 1."""

    def embed(texts: Sequence[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            out = _post(
                f"{url}/api/embed", {"model": model, "input": list(texts[i : i + EMBED_BATCH])}, 120
            )
            if "error" in out:
                raise LLMError(f"ollama {model}: {out['error']}")
            rows += out["embeddings"]
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    return embed


def ollama_unload(model: str, url: str = defaults.OLLAMA_URL) -> None:
    """Drop ``model`` from memory now, so the next model fits."""
    _post(f"{url}/api/generate", {"model": model, "keep_alive": 0}, 60)


def claude_generate(
    model: str | None = None, budget: float = 2.0, timeout: float = 900, command: str = "claude"
) -> Generate:
    """A ``Generate`` on headless Claude Code (``claude -p``) with no tools, MCP servers or saved
    session, run in an empty directory so no CLAUDE.md or project memory reaches it. ``budget``
    caps one call's cost in US dollars."""

    def generate(system: str, prompt: str, schema: dict[str, Any]) -> Reply:
        args = [command, "-p", "--output-format", "json", "--tools", "", "--strict-mcp-config"]
        args += ["--no-session-persistence", "--system-prompt", system]
        args += ["--json-schema", json.dumps(schema), "--max-budget-usd", str(budget)]
        if model:
            args += ["--model", model]
        with tempfile.TemporaryDirectory() as cwd:
            try:
                done = subprocess.run(
                    args, input=prompt, capture_output=True, text=True, timeout=timeout, cwd=cwd
                )
            except subprocess.TimeoutExpired:
                raise LLMError(f"{command} took over {timeout:.0f}s") from None
            except FileNotFoundError:
                raise LLMError(f"{command} is not installed or not on PATH") from None
        try:
            out = json.loads(done.stdout)
        except json.JSONDecodeError:
            detail = (done.stderr or done.stdout).strip()[:300]
            raise LLMError(f"{command} exited {done.returncode}: {detail}") from None
        if done.returncode or out.get("is_error"):
            raise LLMError(f"{command} failed: {str(out.get('result'))[:300]}")
        text = str(out.get("result") or "")
        data = out.get("structured_output")
        return Reply(
            text,
            data if data is not None else parse_json(text),
            cost=float(out.get("total_cost_usd") or 0.0),
        )

    return generate
