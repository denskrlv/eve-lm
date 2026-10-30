"""Word problems (kinds S, W): GSM8K's train split and SVAMP, each with its worked answer.
A share of the documents writes its numbers in words ("twenty-five")."""

import re
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

from eve.data.tinytools.common import Context, Doc, clean, spell_numbers

SPELLED_SHARE = 0.3  # documents whose numbers are written in words
_CALCULATION = re.compile(r"<<[^>]*>>")  # GSM8K's calculator annotations
_FINAL = re.compile(r"\n?####\s*(.+)$")


def _number(text: str) -> str:
    """A number without trailing zeros or a bare point: 145.0 -> 145, .5 -> 0.5."""
    text = str(text).strip()
    text = re.sub(r"(?<![\d.])\.(\d)", r"0.\1", text)
    return re.sub(r"(\d+)\.0+\b", r"\1", text)


def _tidy(text: str) -> str:
    """Spaces before punctuation removed and floats like 70.0 written as 70."""
    text = re.sub(r"\s+([.,?!;:])", r"\1", clean(text))
    return re.sub(r"\b(\d+)\.0+\b", r"\1", text)


def _problems(rows: Iterator[tuple[str, str]], ctx: Context) -> Iterator[Doc]:
    """(question, worked answer) pairs as documents, without travel or eval-set questions."""
    seen = set()
    for question, answer in rows:
        if question in seen:
            ctx.drop("duplicate question")
            continue
        seen.add(question)
        if ctx.keep(question):
            text = f"{question}\n{answer}"
            yield Doc(spell_numbers(text) if ctx.rng.random() < SPELLED_SHARE else text, "scene")


def _columns(paths: list[Path], columns: list[str]) -> Iterator[tuple[str, ...]]:
    for path in paths:
        table = pq.read_table(path, columns=columns)
        yield from zip(*(table.column(c).to_pylist() for c in columns), strict=True)


def gsm8k(raw: Path, ctx: Context) -> Iterator[Doc]:
    """GSM8K's train problems, the reasoning without the calculator annotations."""

    def rows() -> Iterator[tuple[str, str]]:
        for question, answer in _columns([raw / "gsm8k/train.parquet"], ["question", "answer"]):
            answer = _CALCULATION.sub("", answer).strip()
            answer = _FINAL.sub(lambda m: f"\nThe answer is {_number(m.group(1))}.", answer)
            yield clean(question), answer

    return _problems(rows(), ctx)


def svamp(raw: Path, ctx: Context) -> Iterator[Doc]:
    """SVAMP's problems with their equation and answer."""

    def rows() -> Iterator[tuple[str, str]]:
        paths = sorted((raw / "svamp").glob("*.parquet"))
        for body, question, equation, answer in _columns(
            paths, ["Body", "Question", "Equation", "Answer"]
        ):
            equation = _tidy(re.sub(r"^\(\s*(.*?)\s*\)$", r"\1", equation.strip()))
            answer = _number(answer)
            yield (
                _tidy(f"{body.rstrip('. ')}. {question}"),
                f"{equation} = {answer}. The answer is {answer}.",
            )

    return _problems(rows(), ctx)
