"""Split, then call: the splitter cuts a request into actions and Eve answers each one.

``respond`` is the pipeline for one example; ``splitter=None`` is Eve alone, the baseline.
Both modes are scored by ``outcome_of`` from the calls Eve actually writes (free decoding),
so their numbers compare with each other, not with ``eve evaluate``'s teacher-forced values.
"""

import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eve.data.generate.actions import row_actions
from eve.data.modules import load_examples
from eve.data.render import TaskExample
from eve.data.schema import ASK_VALUE, Tool
from eve.evaluate import calls_group
from eve.infer.predict import WrittenCall, answer_questions
from eve.model.checkpoint import Checkpoint
from eve.model.eve import Eve
from eve.model.splitter import Splitter
from eve.split.decide import Thresholds, decide, split_probs, split_request
from eve.split.words import words_in
from eve.tasks.questions import Outcome, Tally

MODES = ("alone", "split")


@dataclass(frozen=True)
class Predicted:
    """What the pipeline answered for one request: the combined intent, every action's calls
    in action order, and the actions' texts."""

    intent: str
    calls: list[WrittenCall]
    actions: list[str]


def combine_intent(intents: Sequence[str]) -> str:
    """CALL if every action is CALL, else the first action's intent that isn't."""
    return next((intent for intent in intents if intent != "call"), "call")


def respond(
    eve: Checkpoint[Eve],
    splitter: Checkpoint[Splitter] | None,
    th: Thresholds | None,
    example: TaskExample,
) -> Predicted:
    """Eve's answer to each of the example's actions, with the example's own tools, joined.
    Without a splitter (or thresholds) the request is one action: Eve alone."""
    text = example.utterance
    actions = [text] if splitter is None or th is None else split_request(splitter, text, th)
    intents, calls = [], []
    for action in actions:
        answers = answer_questions(eve.model, eve.tokenizer, action, example.tools)
        intents.append(max(answers.intent, key=answers.intent.__getitem__))
        calls += answers.written
    return Predicted(combine_intent(intents), calls, actions)


def outcome_of(example: TaskExample, predicted: Predicted) -> Outcome:
    """The example scored like ``eve.tasks.questions.Outcome``: tools right when the called
    tools are the gold ones in order; per gold argument value, right when the call at the
    same place is to the same tool and wrote that value (or asked for a "?" one)."""
    gold_tools = [call.tool for call in example.calls]
    tools_ok = [call.tool for call in predicted.calls] == gold_tools
    values = sum(len(call.args) for call in example.calls)
    right = 0
    for gold, said in zip(example.calls, predicted.calls, strict=False):
        if gold.tool != said.tool:
            continue
        for name, value in gold.args.items():
            written = said.values.get(name)
            if written is None:
                continue
            if value == ASK_VALUE:
                right += written.kind == "ask"
            else:
                right += written.kind == "value" and written.value == value
    has_slots = not (example.intent == "ask" and not example.calls)
    return Outcome(example.intent, predicted.intent, has_slots, tools_ok, values, right)


def evaluate_pipeline(
    eve: Checkpoint[Eve],
    splitter: Checkpoint[Splitter] | None,
    th: Thresholds | None,
    path: str | Path,
    library: dict[str, Tool],
    on_example: Callable[[], None] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """``{mode: {group: scores}}`` on one eval set: Eve alone and, with a splitter, split
    then call; groups are the number of gold calls (``calls_group``) and "all". A request the
    splitter doesn't split reuses Eve alone's answer (same input, same answer)."""
    tallies: dict[str, defaultdict[str, Tally]] = {mode: defaultdict(Tally) for mode in MODES}
    for example in load_examples(Path(path), library):
        alone = respond(eve, None, None, example)
        answers = {"alone": alone}
        if splitter is not None and th is not None:
            actions = split_request(splitter, example.utterance, th)
            answers["split"] = alone if len(actions) == 1 else respond(eve, splitter, th, example)
        for mode, predicted in answers.items():
            outcome = outcome_of(example, predicted)
            for group in ("all", calls_group(len(example.calls))):
                tallies[mode][group].add(outcome)
        if on_example is not None:
            on_example()
    return {
        mode: {group: tally.scores() for group, tally in sorted(groups.items())}
        for mode, groups in tallies.items()
        if groups
    }


def split_metrics(
    splitter: Checkpoint[Splitter], th: Thresholds, rows: Iterable[dict[str, Any]]
) -> dict[str, float]:
    """The splitter on rows with known actions: ``exact_split`` (multi-action rows split
    exactly right), ``unsure`` (multi-action rows left whole) and ``wrong_split``
    (single-action rows split), with the counts of each kind."""
    labeled = [(row["utterance"], a) for row in rows if (a := row_actions(row)) is not None]
    probs = split_probs(splitter, [text for text, _ in labeled])
    multi = right = unsure = single = wrong = 0
    for (text, actions), p in zip(labeled, probs, strict=True):
        decided = None if p is None else decide(p[0], p[1], p[2], text, th)
        if len(actions) == 1:
            single += 1
            wrong += decided is not None
            continue
        multi += 1
        unsure += decided is None
        if p is not None and decided == [words_in(p[2], action) for action in actions]:
            right += 1
    return {
        "exact_split": right / max(1, multi),
        "unsure": unsure / max(1, multi),
        "wrong_split": wrong / max(1, single),
        "multi": float(multi),
        "single": float(single),
    }


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


def hard_set_scores(
    splitter: Checkpoint[Splitter], th: Thresholds, rows: Iterable[dict[str, Any]]
) -> dict[str, float]:
    """Per ``kind`` of the hard set (and "all"), the share of requests whose action texts
    are the expected ones, compared without case and punctuation."""
    seen: defaultdict[str, int] = defaultdict(int)
    right: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        said = [_norm(t) for t in split_request(splitter, row["utterance"], th)]
        ok = said == [_norm(t) for t in row["actions"]]
        for kind in (row["kind"], "all"):
            seen[kind] += 1
            right[kind] += ok
    return {kind: right[kind] / seen[kind] for kind in seen}
