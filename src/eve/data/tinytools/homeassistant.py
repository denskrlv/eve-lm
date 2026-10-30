"""Home Assistant intents (kinds W, T, S): the English test sentences with their responses, plus a
few random sentences from each template.

Templates use hassil's syntax: ``(a|b)`` picks one, ``[a]`` is optional, ``(a;b)`` is any order,
``<rule>`` expands a rule from ``rules/en`` and ``{list}`` takes a value from ``lists/``. Lists
that Home Assistant fills at runtime (areas, floors, device names) take the names the test files
declare."""

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from eve.data.draw import pick, shuffled
from eve.data.tinytools.common import Context, Doc, Intent, chunks, clean, humanize, intent_docs

SAMPLES_PER_TEMPLATE = 5  # the templates are very regular: a few sentences each is plenty
MAX_DEPTH = 20  # rules that expand into themselves stop here
WILDCARD_FILLS = {"area": "areas", "floor": "floors", "name": "entities"}  # list -> test key

# A parsed template: ("text", str) | ("seq", [node]) | ("alt", [node]) | ("perm", [node]) |
# ("rule", name) | ("list", name)
Node = tuple[str, Any]


class UnknownName(Exception):
    """A template uses a list or rule we have no values for."""


def parse(template: str) -> Node:
    """A hassil template as a tree of nodes."""
    node, _ = _group(template, 0, None)
    return node


def _group(text: str, i: int, end: str | None) -> tuple[Node, int]:
    """The sequences from ``i`` up to ``end`` (or the end of ``text``), split by | or ;."""
    seqs: list[list[Node]] = [[]]
    kind, buffer = "alt", ""
    while i < len(text) and text[i] != end:
        char = text[i]
        if char in "([<{|;" and buffer:
            seqs[-1].append(("text", buffer))
            buffer = ""
        if char in "([":
            sub, i = _group(text, i + 1, ")" if char == "(" else "]")
            seqs[-1].append(sub if char == "(" else ("alt", [sub, ("text", "")]))
            continue
        if char in "<{":
            close = text.index(">" if char == "<" else "}", i)
            name = text[i + 1 : close].split(":")[0].strip()
            seqs[-1].append(("rule" if char == "<" else "list", name))
            i = close + 1
            continue
        if char in "|;":
            kind = "perm" if char == ";" else "alt"
            seqs.append([])
        else:
            buffer += char
        i += 1
    if buffer:
        seqs[-1].append(("text", buffer))
    return (kind, [("seq", seq) for seq in seqs]), i + 1


@dataclass
class Grammar:
    """Expansion rules and slot lists, by name; rules and list values are parsed when used."""

    rules: dict[str, str] = field(default_factory=dict)
    lists: dict[str, dict[str, Any]] = field(default_factory=dict)
    fills: dict[str, list[str]] = field(default_factory=dict)
    _parsed: dict[str, Node] = field(default_factory=dict)

    def with_local(self, doc: dict[str, Any]) -> "Grammar":
        """This grammar plus a sentence file's own rules and lists."""
        rules = {**self.rules, **(doc.get("expansion_rules") or {})}
        lists = {**self.lists, **(doc.get("lists") or {})}
        return Grammar(rules, lists, self.fills)

    def node(self, template: str) -> Node:
        if template not in self._parsed:
            self._parsed[template] = parse(template)
        return self._parsed[template]

    def sample(self, node: Node, rng: np.random.Generator, depth: int = 0) -> str:
        """A random sentence from ``node``. Raises UnknownName for a list or rule we lack."""
        kind, value = node
        if depth > MAX_DEPTH:
            raise UnknownName("too deep")
        if kind == "text":
            return str(value)
        if kind == "seq":
            return "".join(self.sample(n, rng, depth + 1) for n in value)
        if kind == "alt":
            return self.sample(pick(rng, value), rng, depth + 1)
        if kind == "perm":
            return " ".join(self.sample(n, rng, depth + 1) for n in shuffled(rng, value))
        if kind == "rule":
            if value not in self.rules:
                raise UnknownName(value)
            return self.sample(self.node(self.rules[value]), rng, depth + 1)
        return self._list_value(value, rng, depth)

    def _list_value(self, name: str, rng: np.random.Generator, depth: int) -> str:
        spec = self.lists.get(name)
        if spec is None or spec.get("wildcard"):
            if not self.fills.get(name):
                raise UnknownName(name)
            return pick(rng, self.fills[name])
        if "range" in spec:
            r = spec["range"]
            return str(int(rng.choice(np.arange(r["from"], r["to"] + 1, r.get("step", 1)))))
        values = spec.get("values") or []
        if not values:
            raise UnknownName(name)
        value = pick(rng, values)
        text = value.get("in", "") if isinstance(value, dict) else str(value)
        return self.sample(self.node(text), rng, depth + 1)


def _yaml_files(folder: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    for path in sorted(folder.glob("**/*.yaml")):
        doc = yaml.safe_load(path.read_text())
        if isinstance(doc, dict):
            yield path, doc


def load_grammar(root: Path) -> Grammar:
    """The English rules and the lists (shared and English), with wildcard fills from the
    names the test files declare."""
    grammar = Grammar()
    for _, doc in _yaml_files(root / "rules/en"):
        grammar.rules.update(doc.get("expansion_rules") or {})
    for folder in (root / "lists", root / "lists/en"):
        for path in sorted(folder.glob("*.yaml")):
            doc = yaml.safe_load(path.read_text()) or {}
            grammar.lists.update(doc.get("lists") or {})
    for list_name, key in WILDCARD_FILLS.items():
        names = {
            clean(str(item["name"]))
            for _, doc in _yaml_files(root / "tests/en")
            for item in doc.get(key) or []
            if isinstance(item, dict) and item.get("name")
        }
        grammar.fills[list_name] = sorted(names)
    return grammar


def _sentence_blocks(path: Path, doc: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """(intent, data block) pairs of a sentence file, in either layout: "data:" in a folder
    named after the intent, or the older "intents: {name: {data: ...}}"."""
    for block in doc.get("data") or []:
        yield path.parent.name, block
    for intent, body in (doc.get("intents") or {}).items():
        for block in body.get("data") or []:
            yield intent, block


def sampled_sentences(root: Path, grammar: Grammar, ctx: Context) -> dict[str, list[str]]:
    """A few random sentences from every English template, by intent."""
    out: dict[str, list[str]] = {}
    for path, doc in _yaml_files(root / "sentences/en"):
        local = grammar.with_local(doc)
        for intent, block in _sentence_blocks(path, doc):
            for template in block.get("sentences") or []:
                node = local.node(template)
                for _ in range(SAMPLES_PER_TEMPLATE):
                    try:
                        sentence = clean(local.sample(node, ctx.rng))
                    except UnknownName:
                        ctx.drop("template with an unknown list or rule")
                        break
                    out.setdefault(intent, []).append(sentence)
    return out


def tested_sentences(root: Path) -> Iterator[tuple[str, list[str], str]]:
    """(intent, sentences, response) for every English test case."""
    for path, doc in _yaml_files(root / "tests/en"):
        for case in doc.get("tests") or []:
            intent = (case.get("intent") or {}).get("name") or path.parent.name
            response = case.get("response")
            response = response[0] if isinstance(response, list) and response else response
            sentences = [clean(s) for s in case.get("sentences") or [] if clean(s)]
            yield intent, sentences, clean(str(response or ""))


def _exchange(request: str, response: str, ctx: Context) -> str:
    name = ctx.name()
    request = request[:1].upper() + request[1:]
    return pick(
        ctx.rng,
        (
            f"“{request},” said {name}. The assistant answered, “{response}.”",
            f"{name}: {request}\nAssistant: {response}.",
        ),
    )


def home_assistant(raw: Path, ctx: Context) -> Iterator[Doc]:
    """Requests by intent as "ways" lists and tool cards, and test cases with their responses as
    short exchanges."""
    root = raw / "home-assistant"
    grammar = load_grammar(root)
    requests = sampled_sentences(root, grammar, ctx)
    exchanges: list[tuple[str, str]] = []
    for intent, sentences, response in tested_sentences(root):
        requests.setdefault(intent, []).extend(sentences)
        if response and not re.search(r"[{}<>]", response):
            exchanges.extend((s, response) for s in sentences)
    intents = (
        Intent(humanize(k).replace(" ", "_"), humanize(k), tuple(v))
        for k, v in sorted(requests.items())
    )
    yield from intent_docs(intents, ctx)
    kept = [e for e in shuffled(ctx.rng, exchanges) if ctx.keep(e[0])]
    for part in chunks(kept, ctx.rng, 3, 6):
        yield Doc("\n\n".join(_exchange(r, a, ctx) for r, a in part), "scene")
