from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from eve.api.errors import CheckpointError, EveError, ToolError
from eve.api.hub import resolve
from eve.api.response import Response
from eve.api.toolbox import ToolBox
from eve.api.tools import ToolLike, to_tool
from eve.data.schema import SchemaError, Tool
from eve.infer.predict import Calibrations, answer_questions
from eve.model.checkpoint import FORMAT, Checkpoint, load_checkpoint
from eve.runtime import pick_device


class Eve:
    """A fine-tuned Eve model, ready to answer requests::

    eve = Eve.load("artifacts/runs/joint-base/model.pt")
    eve.add_tools(set_lights, set_thermostat, play_music)  # once
    response = eve.respond("turn off the kitchen lights")
    if response.intent == "call":
        response.run()

    ``respond`` shows the model at most ``max_tools`` of the registered tools: all of them if
    there are that few, else the ones ``tools.search`` ranks best for the request. Passing
    ``tools=`` to ``respond`` uses exactly those instead.
    """

    def __init__(self, checkpoint: Checkpoint, source: str) -> None:
        if not checkpoint.meta.get("joint"):
            raise CheckpointError(
                f"{source} isn't a fine-tuned joint model: it can't write argument values"
            )
        self._model = checkpoint.model.eval().requires_grad_(False)
        self._tokenizer = checkpoint.tokenizer
        self._meta = checkpoint.meta
        self._calibrations = Calibrations.from_meta(checkpoint.meta)
        self.source = source
        self.tools = ToolBox()  # the registered tools

    @classmethod
    def load(cls, source: str | Path, *, device: str = "cpu", revision: str | None = None) -> "Eve":
        """The model in ``source``: a checkpoint file, a directory holding ``model.pt``, or a
        Hugging Face Hub repo id (``"user/eve-joint-base"``, cached after the first download).

        ``device`` is ``"cpu"`` (the default), ``"cuda"``, ``"mps"`` or ``"auto"`` (the
        fastest there is). ``revision`` pins a Hub branch, tag or commit.
        """
        path = resolve(source, revision)
        try:
            checkpoint = load_checkpoint(path, pick_device(None if device == "auto" else device))
        except Exception as error:  # torch.load raises many kinds for a file that isn't one
            raise CheckpointError(f"can't load {path}: {error}") from error
        return cls(checkpoint, str(source))

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
        calibration = self._calibrations.familiar if familiar else self._calibrations.new
        try:
            answers = answer_questions(
                self._model, self._tokenizer, text.strip(), schemas, True, calibration
            )
        except (SchemaError, ValueError) as error:  # e.g. too long for the encoder
            raise EveError(str(error)) from error
        return Response.from_answers(text, answers, familiar, functions, [t.name for t in schemas])

    def __repr__(self) -> str:
        return f"Eve({self.source!r})"


def _check(tools: Sequence[Tool], max_tools: int) -> None:
    if not tools:
        raise ToolError("give at least one tool")
    if len(tools) > max_tools:
        raise ToolError(f"{len(tools)} tools; Eve reads at most {max_tools} per request")
    names = [t.name for t in tools]
    if repeated := sorted({n for n in names if names.count(n) > 1}):
        raise ToolError(f"two tools are called {repeated[0]!r}")
