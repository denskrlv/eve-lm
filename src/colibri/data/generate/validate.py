import re
from collections.abc import Collection
from typing import Any

from colibri.data.generate.templates import MAX_CALLS, number_words
from colibri.data.render import TaskExample
from colibri.data.schema import ASK_VALUE, Arg

MAX_UTTERANCE_CHARS = 300


def value_forms(arg: Arg, value: Any) -> list[str]:
    """The ways a value may be written in the utterance: 7 -> "7", "seven"."""
    if arg.type in ("int", "float"):
        forms = [arg.format(value)]
        if float(value).is_integer() and 0 <= value <= 999:
            words = number_words(int(value))
            forms += [words, words.replace("-", " ")]
        return forms
    return [str(value)]


def mentions(text: str, form: str) -> bool:
    """``form`` appears in ``text`` as whole words, ignoring case."""
    return re.search(rf"(?<!\w){re.escape(form)}(?!\w)", text, re.IGNORECASE) is not None


def check_example(
    example: TaskExample, implied: Collection[tuple[int, str]] = frozenset()
) -> list[str]:
    """Label problems the TaskExample constructor doesn't catch (empty list = fine).

    The main rule: every str and number value appears in the utterance, because the value
    decoder copies it from the text. Exempt: bools ("turn off" means on=false), enums (the
    model picks one of the listed options) and ``implied`` values, given as (call index, argument)
    pairs, which a template sets on purpose ("mute it" means level=0).
    """
    errors = []
    text = example.utterance
    if not text.strip() or len(text) > MAX_UTTERANCE_CHARS:
        errors.append(f"utterance is empty or longer than {MAX_UTTERANCE_CHARS} characters")
    if not 1 <= len(example.tools) <= 5:
        errors.append(f"{len(example.tools)} tools in the prompt; 1 to 5 allowed")
    if len(example.calls) > MAX_CALLS:
        errors.append(f"{len(example.calls)} calls; at most {MAX_CALLS}")
    tools = {t.name: t for t in example.tools}
    for i, call in enumerate(example.calls):
        for name, value in call.args.items():
            arg = tools[call.tool].arg(name)
            if arg.type in ("bool", "enum") or (i, name) in implied:
                continue
            if value == ASK_VALUE and example.intent == "ask":
                continue
            if not any(mentions(text, form) for form in value_forms(arg, value)):
                errors.append(f"{call.tool}.{name}={value!r} is not in the utterance")
    if example.intent == "ask" and example.calls:
        if not any(v == ASK_VALUE for c in example.calls for v in c.args.values()):
            errors.append("intent 'ask' with calls needs a missing value ('?')")
    return errors


def normalize(text: str) -> str:
    """For finding the same utterance in two splits: case and punctuation don't count."""
    return " ".join(re.findall(r"\w+", text.lower()))
