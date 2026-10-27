from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from eve.api.errors import EveError
from eve.infer.predict import Answers, WrittenCall

Intent = Literal["call", "ask", "unsupported", "none"]
ACTS = ("call", "ask")  # the intents that come with calls


@dataclass(frozen=True)
class Call:
    """One call Eve chose: the tool, its arguments, and how sure it is of each part."""

    tool: str
    args: dict[str, Any]  # the arguments the user gave: the call to make
    confidence: float  # that this call is to this tool
    arg_confidence: dict[str, float]  # per argument: of what was written (a value, or none)
    missing: list[str]  # required arguments the user didn't give: what to ask them for
    invalid: dict[str, str]  # arguments whose written value doesn't fit the schema -> why

    @classmethod
    def from_written(cls, call: WrittenCall) -> "Call":
        return cls(
            tool=call.tool,
            args=call.args,
            confidence=call.prob,
            arg_confidence={name: v.prob for name, v in call.values.items()},
            missing=call.missing,
            invalid={name: v.error or "" for name, v in call.values.items() if v.kind == "invalid"},
        )

    def __str__(self) -> str:
        """``set_lights(room='kitchen', on=False)``, then what is missing or invalid."""
        text = f"{self.tool}({', '.join(f'{k}={v!r}' for k, v in self.args.items())})"
        if self.missing:
            text += f" missing {', '.join(self.missing)}"
        if self.invalid:
            text += f" invalid {', '.join(self.invalid)}"
        return text

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "args": self.args,
            "confidence": self.confidence,
            "arg_confidence": self.arg_confidence,
            "missing": self.missing,
            "invalid": self.invalid,
        }


@dataclass(frozen=True)
class Response:
    """What Eve decided for one request.

    Act on ``intent`` first: ``"call"`` (make ``calls``), ``"ask"`` (ask the user for
    ``missing``), ``"unsupported"`` (no tool given can do it) or ``"none"`` (not a request,
    e.g. "thanks"). ``calls`` is empty unless the intent is call or ask. Confidences are
    probabilities, calibrated if the model was (``familiar_tools`` says which calibration:
    the one for tools it trained on, or the one for new tools).
    """

    text: str
    intent: Intent
    confidence: float  # of the intent
    intents: dict[str, float]  # all four intents' probabilities
    calls: list[Call]
    familiar_tools: bool
    slots: list[dict[str, float]] = field(repr=False)  # per call slot: tool (or STOP) -> prob
    considered: list[str] = field(default_factory=list)  # the tools the model read, in order
    functions: Mapping[str, Callable[..., Any]] = field(default_factory=dict, repr=False)

    @classmethod
    def from_answers(
        cls,
        text: str,
        answers: Answers,
        familiar: bool,
        functions: Mapping[str, Callable[..., Any]],
        considered: list[str] | None = None,
    ) -> "Response":
        intent = cast(Intent, max(answers.intent, key=answers.intent.__getitem__))
        calls = [Call.from_written(c) for c in answers.written] if intent in ACTS else []
        return cls(
            text=text,
            intent=intent,
            confidence=answers.intent[intent],
            intents=dict(answers.intent),
            calls=calls,
            familiar_tools=familiar,
            slots=answers.slots,
            considered=considered or [],
            functions=dict(functions),
        )

    def __str__(self) -> str:
        """``call 96%: set_lights(room='kitchen', on=False)``: the intent, how sure Eve is,
        and the calls."""
        text = f"{self.intent} {self.confidence:.0%}"
        return text + (": " + ", ".join(map(str, self.calls)) if self.calls else "")

    def __repr__(self) -> str:
        return f"<Response {self}>"

    @property
    def missing(self) -> list[tuple[str, str]]:
        """``(tool, argument)`` for each required argument the user still has to give."""
        return [(call.tool, name) for call in self.calls for name in call.missing]

    def run(self, min_confidence: float = 0.0) -> list[Any]:
        """Call the Python function of each call, in order, and return their results.

        Nothing runs unless all of it can: the intent is ``"call"``, every call is to a tool
        given as a function and has every argument it needs, and the intent and every tool
        choice are at least ``min_confidence`` likely. Otherwise this raises ``EveError``.
        """
        if self.intent != "call":
            raise EveError(f"the intent is {self.intent!r}, so there is nothing to run")
        if self.confidence < min_confidence:
            raise EveError(f"intent confidence {self.confidence:.0%} < {min_confidence:.0%}")
        for call in self.calls:
            if call.tool not in self.functions:
                raise EveError(f"{call.tool} was given as a schema, not a function")
            if call.missing or call.invalid:
                bad = call.missing + list(call.invalid)
                raise EveError(f"{call.tool} can't run: no valid value for {bad}")
            if call.confidence < min_confidence:
                raise EveError(
                    f"{call.tool} confidence {call.confidence:.0%} < {min_confidence:.0%}"
                )
        return [self.functions[call.tool](**call.args) for call in self.calls]

    def to_dict(self) -> dict[str, Any]:
        """The response as plain values, ready for ``json.dumps``."""
        return {
            "text": self.text,
            "intent": self.intent,
            "confidence": self.confidence,
            "intents": self.intents,
            "calls": [call.to_dict() for call in self.calls],
            "missing": [list(pair) for pair in self.missing],
            "familiar_tools": self.familiar_tools,
            "considered": self.considered,
            "slots": self.slots,
        }
