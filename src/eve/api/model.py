from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch

from eve.api.errors import CheckpointError, EveError, ToolError
from eve.api.hub import companion, resolve
from eve.api.response import Response
from eve.api.toolbox import BM25, Dense, Scorer, ToolBox
from eve.api.tools import ToolLike, to_tool
from eve.data.schema import SchemaError, Tool
from eve.infer.predict import Calibrations, answer_questions
from eve.model.checkpoint import FORMAT, RETRIEVAL_TASK, Checkpoint, load_checkpoint
from eve.runtime import pick_device


class Eve:
    """A fine-tuned Eve model, ready to answer requests::

    eve = Eve.load("artifacts/runs/eve-32m/model.pt")
    eve.add_tools(set_lights, set_thermostat, play_music)  # once
    response = eve.respond("turn off the kitchen lights")
    if response.intent == "call":
        response.run()

    ``respond`` shows the model at most ``max_tools`` of the registered tools: all of them if
    there are that few, else the ones ``tools.search`` ranks best for the request: by meaning
    with a retriever (``retriever.pt`` beside the model), else by shared words (BM25).
    Passing ``tools=`` to ``respond`` uses exactly those instead.
    """

    def __init__(
        self,
        checkpoint: Checkpoint,
        source: str,
        retriever: Checkpoint | None = None,
    ) -> None:
        if not checkpoint.meta.get("joint"):
            raise CheckpointError(
                f"{source} isn't a fine-tuned joint model: it can't write argument values"
            )
        self._model = checkpoint.model.eval().requires_grad_(False)
        self._tokenizer = checkpoint.tokenizer
        self._meta = checkpoint.meta
        self._calibrations = Calibrations.from_meta(checkpoint.meta)
        self.source = source
        self.tools = ToolBox(scorer=_scorer(retriever))  # the registered tools

    @classmethod
    def load(
        cls,
        source: str | Path,
        *,
        device: str = "cpu",
        revision: str | None = None,
        retriever: str | Path | None = "auto",
    ) -> "Eve":
        """The model in ``source``: a checkpoint file, a directory holding ``model.pt``, or a
        Hugging Face Hub repo id (``"user/eve-32m"``, cached after the first download).

        ``device`` is ``"cpu"`` (the default), ``"cuda"``, ``"mps"`` or ``"auto"`` (the
        fastest there is). ``revision`` pins a Hub branch, tag or commit.

        ``retriever`` picks the tools a request is shown when more are registered than it
        holds: ``"auto"`` loads ``retriever.pt`` from beside the model if there is one, a path
        loads that file, ``None`` uses none: tools are then ranked by shared words (BM25).
        """
        torch_device = pick_device(None if device == "auto" else device)
        path = resolve(source, revision)
        checkpoint = _load(path, torch_device)
        found = companion(source, revision) if retriever == "auto" else retriever
        extra = None if found is None else _load(resolve(found), torch_device)
        if extra is not None and extra.meta.get("task") != RETRIEVAL_TASK:
            raise CheckpointError(f"{found} isn't a retriever")
        return cls(checkpoint, str(source), extra)

    @property
    def max_tools(self) -> int:
        """The most tools one request may have."""
        return self._model.cfg.max_tools

    @property
    def info(self) -> dict[str, Any]:
        """What this model is: its format, size, calibration and validation scores."""
        return {
            "source": self.source,
            "format": FORMAT,
            "parameters": sum(p.numel() for p in self._model.parameters()),
            "max_tools": self.max_tools,
            "calibrated": "calibration" in self._meta,
            "known_tools": len(self._calibrations.known),
            "scorer": type(self.tools.scorer).__name__,
            "metrics": {k: v for k, v in self._meta.items() if k.startswith("val_")},
        }

    def add_tools(self, *tools: ToolLike | Iterable[ToolLike]) -> None:
        """Register tools for every later request: functions (``@eve.tool``), OpenAI- or
        MCP-style JSON Schema dicts, Eve schema dicts, or lists of them. A tool with the name
        of a registered one replaces it. ``eve.tools`` is the registry (``save``, ``load``,
        ``remove``, ``search``)."""
        self.tools.add(*tools)

    def respond(self, text: str, tools: Sequence[ToolLike] | None = None) -> Response:
        """What to do about ``text``: call some tools, ask the user for a missing argument, or
        none of them can do it.

        Without ``tools``, Eve picks up to ``max_tools`` of the registered ones (see
        ``add_tools``); ``Response.considered`` names them. With ``tools`` (1 to
        ``max_tools``), it reads exactly those.
        """
        if not isinstance(text, str) or not text.strip():
            raise EveError("the request is empty")
        if tools is None:
            if not self.tools:
                raise ToolError("no tools: register them with add_tools, or pass tools=")
            schemas = self.tools.search(text, self.max_tools)
            registered = self.tools.functions
            functions = {t.name: registered[t.name] for t in schemas if t.name in registered}
        else:
            schemas, functions = [], {}
            for source in tools:
                schema, function = to_tool(source)
                schemas.append(schema)
                if function is not None:
                    functions[schema.name] = function
        _check(schemas, self.max_tools)
        familiar = self._calibrations.is_familiar(schemas)
        calibration = self._calibrations.for_tools(schemas)
        try:
            answers = answer_questions(
                self._model, self._tokenizer, text.strip(), schemas, calibration
            )
        except (SchemaError, ValueError) as error:  # e.g. too long for the encoder
            raise EveError(str(error)) from error
        return Response.from_answers(text, answers, familiar, functions, [t.name for t in schemas])

    def __repr__(self) -> str:
        return f"Eve({self.source!r})"


def _load(path: Path, device: torch.device) -> Checkpoint:
    try:
        return load_checkpoint(path, device)
    except Exception as error:  # torch.load raises many kinds for a file that isn't one
        raise CheckpointError(f"can't load {path}: {error}") from error


def _scorer(retriever: Checkpoint | None) -> Scorer:
    """What ``tools.search`` ranks by: the retriever (``Dense``), else BM25."""
    if retriever is None:
        return BM25()
    model = retriever.model.eval().requires_grad_(False)
    return Dense(model, retriever.tokenizer)  # type: ignore[arg-type]


def _check(tools: Sequence[Tool], max_tools: int) -> None:
    if not tools:
        raise ToolError("give at least one tool")
    if len(tools) > max_tools:
        raise ToolError(f"{len(tools)} tools; Eve reads at most {max_tools} per request")
    names = [t.name for t in tools]
    if repeated := sorted({n for n in names if names.count(n) > 1}):
        raise ToolError(f"two tools are called {repeated[0]!r}")
