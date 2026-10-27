"""Write TinyTools documents with headless Claude Code (``claude -p``).

``eve corpus generate`` sends a run's prompts in batches of about 50, as the pilots were written,
and checks every document as it comes back: one that breaks its prompt goes back with the
checker's reasons, up to ``rounds`` times. Each finished batch is kept as
``batches/batch-NNNN.jsonl`` beside its prompts and appended to ``documents.jsonl``, so a stopped
run picks up where it was.
Near-duplicates across the corpus are left to ``eve corpus check``.
"""

import json
import re
import subprocess
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from eve.data.generate.tinytools import Lexicon, PromptSpec, check_corpus, check_document
from eve.data.jsonl import read_jsonl, write_jsonl
from eve.data.schema import Tool

SYSTEM = """You write short documents for TinyTools, a pretraining corpus that teaches a small \
model what people ask assistants for and what tools do.

You get prompts, one JSON object per line: {{"id": ..., "prompt": ...}}. For each one, write the \
document its prompt asks for. Follow it exactly: the length, the three words (any form is fine: \
"bought" for "buy"), the feature, and the format for tool cards and dialogues.

Answer with one JSON object per line and nothing else, in the order of the prompts:
{{"id": "tt-000042", "text": "Mia asked the kitchen speaker ..."}}
Keep line breaks inside a text as \\n.

Rules that matter more than they look:
- Write only the document: no "Sure, here is...", no title, no notes afterwards.
- Spell the three words correctly; a misspelt word doesn't count as used.
- In cards and ways lists, the three words may go outside the requests (in does:, inputs: or the \
first line). Every request must be something a real person would say, so don't bend one to fit \
a word.
- Simple, everyday English. Vary the people, places, numbers and sentence shapes from one \
document to the next, within this batch too.
- Never write angle brackets.
- Never name one of these as a tool: {heldout}. Saying "add the two numbers" in a story is fine; \
a tool called add is not."""

BATCH = re.compile(r"batch-(\d+)\.prompts\.jsonl$")


class GenerateError(RuntimeError):
    """The generator failed: it exited with an error or its output wasn't its JSON."""


@dataclass(frozen=True)
class Reply:
    text: str
    cost: float = 0.0  # US dollars, as Claude Code reports it


Write = Callable[[str, str], Reply]  # (system prompt, message) -> the generator's reply


def claude_writer(command: str = "claude", model: str | None = None, timeout: float = 900) -> Write:
    """A ``Write`` that runs ``claude -p`` with no tools, MCP servers or saved session. It runs
    in an empty directory, so no CLAUDE.md or project memory reaches the generator."""

    def write(system: str, message: str) -> Reply:
        args = [command, "-p", "--output-format", "json", "--tools", "", "--strict-mcp-config"]
        args += ["--no-session-persistence", "--system-prompt", system]
        if model:
            args += ["--model", model]
        with tempfile.TemporaryDirectory() as cwd:
            try:
                done = subprocess.run(
                    args, input=message, capture_output=True, text=True, timeout=timeout, cwd=cwd
                )
            except subprocess.TimeoutExpired as error:
                raise GenerateError(f"{command} took over {timeout:.0f}s") from error
        try:
            out = json.loads(done.stdout)
        except json.JSONDecodeError:
            detail = (done.stderr or done.stdout).strip()[:300]
            raise GenerateError(f"{command} exited {done.returncode}: {detail}") from None
        if done.returncode or out.get("is_error"):
            raise GenerateError(f"{command} failed: {str(out.get('result'))[:300]}")
        return Reply(str(out.get("result") or ""), float(out.get("total_cost_usd") or 0.0))

    return write


def system_prompt(heldout: Sequence[Tool]) -> str:
    """The generator's rules, naming the held-out tools it must never invent."""
    names = ", ".join(sorted(tool.name for tool in heldout)) or "(none)"
    return SYSTEM.format(heldout=names)


def batch_message(
    specs: Sequence[PromptSpec], rewrites: Mapping[str, tuple[str, list[str]]]
) -> str:
    """The prompts of one batch as JSON lines. A prompt in ``rewrites`` carries its rejected
    text and the checker's reasons."""
    lines = []
    for spec in specs:
        row: dict[str, object] = {"id": spec.id, "prompt": spec.prompt}
        if spec.id in rewrites:
            text, reasons = rewrites[spec.id]
            row |= {"rejected": text, "because": reasons}
        lines.append(json.dumps(row, ensure_ascii=False))
    head = "Write one document for each prompt."
    if rewrites:
        head += (
            ' The ones with "rejected" were written before and broke their prompt: write them '
            'again from the prompt, fixing what "because" says.'
        )
    return head + "\n\n" + "\n".join(lines)


def parse_documents(reply: str, ids: Mapping[str, object] | set[str]) -> dict[str, str]:
    """The ``{"id", "text"}`` lines of a reply whose id is in ``ids``; code fences, chatter and
    broken lines are skipped."""
    out = {}
    for line in reply.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("id") in ids and isinstance(row.get("text"), str):
            out[row["id"]] = row["text"]
    return out


@dataclass
class BatchResult:
    """What one batch gave: the last text per prompt, accepted or not."""

    number: int
    size: int  # prompts sent
    documents: dict[str, str] = field(default_factory=dict)
    rejected: dict[str, list[str]] = field(default_factory=dict)  # still failing after the rounds
    calls: int = 0
    cost: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> int:
        return len(self.documents) - len(self.rejected)


def write_batch(
    specs: Sequence[PromptSpec],
    write: Write,
    lexicon: Lexicon,
    heldout: Sequence[Tool] = (),
    rounds: int = 3,
    previous: Mapping[str, tuple[str, list[str]]] | None = None,
    number: int = 0,
) -> BatchResult:
    """Write ``specs`` in up to ``rounds`` calls: each call after the first sends only the
    prompts still missing or rejected, the rejected ones with their reasons. ``previous`` are
    rejected texts from an earlier run to rewrite. A failed call uses up a round."""
    result = BatchResult(number, len(specs))
    system = system_prompt(heldout)
    todo = {spec.id: spec for spec in specs}
    rewrites = dict(previous or {})  # id -> (rejected text, reasons)
    for _ in range(rounds):
        if not todo:
            break
        message = batch_message(list(todo.values()), rewrites)
        try:
            reply = write(system, message)
        except GenerateError as error:
            result.errors.append(str(error))
            continue
        result.calls += 1
        result.cost += reply.cost
        for doc_id, text in parse_documents(reply.text, todo).items():
            result.documents[doc_id] = text
            if errors := check_document(todo[doc_id], text, lexicon, heldout):
                rewrites[doc_id] = (text, errors)
            else:
                rewrites.pop(doc_id, None)
                del todo[doc_id]
    result.rejected = {doc_id: rewrites[doc_id][1] for doc_id in todo if doc_id in result.documents}
    return result


def pending(
    specs: Sequence[PromptSpec],
    documents: Mapping[str, str],
    lexicon: Lexicon,
    heldout: Sequence[Tool] = (),
    rewrite: bool = False,
) -> tuple[list[PromptSpec], dict[str, tuple[str, list[str]]]]:
    """The prompts to send: those without a document and, with ``rewrite``, those whose
    document ``check_corpus`` rejects (near-duplicates too), with that text and its reasons."""
    rejected = check_corpus(specs, documents, lexicon, heldout).rejected if rewrite else {}
    todo = [spec for spec in specs if spec.id not in documents or spec.id in rejected]
    return todo, {doc_id: (documents[doc_id], reasons) for doc_id, reasons in rejected.items()}


def read_documents(path: Path) -> dict[str, str]:
    """``documents.jsonl`` as id -> text; a later line for an id replaces an earlier one."""
    return {str(row["id"]): str(row["text"]) for row in read_jsonl(path)} if path.is_file() else {}


def _next_batch(batches: Path) -> int:
    numbers = [int(m[1]) for p in batches.glob("*.jsonl") if (m := BATCH.search(p.name))]
    return max(numbers, default=-1) + 1


def generate(
    run: str | Path,
    todo: Sequence[PromptSpec],
    write: Write,
    lexicon: Lexicon,
    heldout: Sequence[Tool] = (),
    previous: Mapping[str, tuple[str, list[str]]] | None = None,
    batch_size: int = 50,
    workers: int = 4,
    rounds: int = 3,
    on_batch: Callable[[BatchResult], None] = lambda _: None,
) -> list[BatchResult]:
    """Write ``todo`` into the run ``run``, ``workers`` batches at a time. Each batch is saved as
    soon as it finishes; at the end ``documents.jsonl`` is rewritten with one line per id, in
    prompt order. Stops, after saving what finished, when a batch gets nothing back but errors:
    a run that can't reach the generator shouldn't burn through every batch."""
    run = Path(run)
    batches, documents = run / "batches", run / "documents.jsonl"
    first = _next_batch(batches)
    chunks = [todo[i : i + batch_size] for i in range(0, len(todo), batch_size)]
    previous = previous or {}
    lock = threading.Lock()
    results: list[BatchResult] = []

    def one(number: int, specs: Sequence[PromptSpec]) -> BatchResult:
        name = f"batch-{number:04d}"
        write_jsonl(batches / f"{name}.prompts.jsonl", (spec.to_json() for spec in specs))
        rewrites = {spec.id: previous[spec.id] for spec in specs if spec.id in previous}
        result = write_batch(specs, write, lexicon, heldout, rounds, rewrites, number)
        rows = [{"id": doc_id, "text": text} for doc_id, text in result.documents.items()]
        write_jsonl(batches / f"{name}.jsonl", rows)
        with lock, open(documents, "a") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return result

    failed: BatchResult | None = None
    with ThreadPoolExecutor(max(1, workers)) as pool:
        futures = [pool.submit(one, first + i, specs) for i, specs in enumerate(chunks)]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            on_batch(result)
            if result.errors and not result.documents:
                failed = result
                pool.shutdown(cancel_futures=True)
                break
    _compact(run, documents)
    if failed:
        raise GenerateError(f"batch {failed.number} got nothing back: {failed.errors[-1]}")
    return sorted(results, key=lambda r: r.number)


def _compact(run: Path, documents: Path) -> None:
    """One line per id in ``documents.jsonl``, the latest text, in prompt order."""
    texts = read_documents(documents)
    if not texts:
        return
    order = {str(row["id"]): i for i, row in enumerate(read_jsonl(run / "prompts.jsonl"))}
    ids = sorted(texts, key=lambda doc_id: order.get(doc_id, len(order)))
    write_jsonl(documents, ({"id": doc_id, "text": texts[doc_id]} for doc_id in ids))
