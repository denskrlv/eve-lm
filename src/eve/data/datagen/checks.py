"""The code checks every generated tool and request must pass (empty list = fine). A problem
is a short reason, sent back to the model that wrote the tool or request."""

import re
from collections.abc import Collection, Iterable
from typing import Any

from eve.data.datagen.shard import GenTool, Row
from eve.data.generate.novel import stem
from eve.data.generate.sampler import STOPWORDS as SAMPLER_STOPWORDS
from eve.data.generate.templates import number_words
from eve.data.generate.validate import check_example, mentions, normalize
from eve.data.schema import ASK_VALUE, SchemaError

STOPWORDS = frozenset(
    SAMPLER_STOPWORDS | {"i", "my", "me", "you", "your", "can", "please", "what", "how"}
)
MIN_NAMES = MIN_DESCRIPTIONS = 2
MAX_NAMES = MAX_DESCRIPTIONS = 4
DESCRIPTION_OVERLAP = 0.5  # descriptions sharing more of their words than this count as one
MAX_DESCRIPTION_WORDS = 12
ASK_TYPES = ("str", "int", "float")  # a value an ASK request can leave out


def content_stems(text: str) -> set[str]:
    """The text's words without stopwords, each without a common ending."""
    return {stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def _value_stems(values: Iterable[Any]) -> set[str]:
    out: set[str] = set()
    for value in values:
        if isinstance(value, bool):
            continue
        out |= content_stems(str(value))
        if isinstance(value, int | float) and float(value).is_integer() and 0 <= value <= 999:
            out |= content_stems(number_words(int(value)).replace("-", " "))
    return out


def shared_words(utterance: str, texts: Iterable[str], values: Iterable[Any] = ()) -> set[str]:
    """Words (stemmed) the request shares with ``texts``, leaving out the words of its own
    values: "kale" in "my kale has holes" is the user's word, not the tool's."""
    said = content_stems(utterance) - _value_stems(values)
    return said & set().union(*(content_stems(t) for t in texts))


def _overlap(a: str, b: str) -> float:
    sa, sb = content_stems(a), content_stems(b)
    return len(sa & sb) / max(1, len(sa | sb))


def tool_problems(tool: GenTool, taken: Collection[str]) -> list[str]:
    """What's wrong with an invented tool: too few or repeated names and descriptions,
    descriptions that say it the same way, or a name already in use."""
    problems = []
    if not MIN_NAMES <= len(set(tool.names)) == len(tool.names) <= MAX_NAMES:
        problems.append(f"needs {MIN_NAMES}-{MAX_NAMES} different names, got {list(tool.names)}")
    if not MIN_DESCRIPTIONS <= len(tool.descriptions) <= MAX_DESCRIPTIONS:
        problems.append(f"needs {MIN_DESCRIPTIONS}-{MAX_DESCRIPTIONS} descriptions")
    else:
        distinct: list[str] = []
        for text in tool.descriptions:
            if all(_overlap(text, d) <= DESCRIPTION_OVERLAP for d in distinct):
                distinct.append(text)
        if len(distinct) < MIN_DESCRIPTIONS:
            problems.append("the descriptions must differ in wording, not just a word or two")
    problems += [
        f"the description {text!r} has over {MAX_DESCRIPTION_WORDS} words: say it shorter"
        for text in tool.descriptions
        if len(text.split()) > MAX_DESCRIPTION_WORDS
    ]
    problems += [f"the name {name} is already taken" for name in tool.names if name in taken]
    return problems


def _pairs(tool: GenTool) -> list[tuple[str, str]]:
    return [(n.replace("_", " "), d) for n in tool.names for d in tool.descriptions]


def clean_pairs(row: Row, tool: GenTool) -> list[tuple[int, int]]:
    """The (name, description) variants of ``tool`` the request shares no word with: the
    ones the build may show it with."""
    values = (row.args or {}).values()
    return [
        (i, j)
        for i, name in enumerate(tool.names)
        for j, description in enumerate(tool.descriptions)
        if not shared_words(row.utterance, [name.replace("_", " "), description], values)
    ]


def row_problems(row: Row, tool: GenTool, seen: Collection[str]) -> list[str]:
    """What's wrong with a generated request: a bad label, a value not in the text, no name and
    description it shares no word with (CALL and ASK only), an ASK the tool can't have or that
    leaves out an optional value, or a repeat."""
    try:
        example = row.example(tool)
    except SchemaError as error:
        return [str(error)]
    problems = check_example(example)
    if row.intent in ("call", "ask") and not clean_pairs(row, tool):
        values = (row.args or {}).values()
        fewest = min((shared_words(row.utterance, pair, values) for pair in _pairs(tool)), key=len)
        words = ", ".join(f'"{w}"' for w in sorted(fewest))
        problems.append(
            f"shares {words} with every name and description pair: avoid the words of at "
            "least one name and one description"
        )
    if row.intent == "ask" and not any(a.required and a.type in ASK_TYPES for a in tool.args):
        problems.append("an ask request needs a required text or number argument to leave out")
    known = {a.name: a for a in tool.args}
    for name, value in (row.args or {}).items() if row.intent == "ask" else ():
        if value == ASK_VALUE and name in known and not known[name].required:
            problems.append(f'only a required value can be left out ("?"), not {name}')
    for name, value in (row.args or {}).items():  # the decoder copies: case must match too
        if name in known and known[name].type == "str" and isinstance(value, str):
            said = re.search(rf"(?<!\w){re.escape(value)}(?!\w)", row.utterance)
            if value != ASK_VALUE and not said and mentions(row.utterance, value):
                problems.append(f"{name}={value!r} must be written with the request's case")
    if row.intent == "unsupported" and row.args is not None:
        problems.append("an unsupported request has no call")
    if normalize(row.utterance) in seen:
        problems.append("repeats an earlier request")
    return problems
