"""Rank tools for a request, to pick the few Eve reads when more are registered.

- ``BM25``: word overlap with each tool's name, description, argument names, options and
  argument descriptions. It misses paraphrases: "20 plus 10" shares no word with "Add two
  numbers".
- ``Dense``: a trained retriever (``eve.model.retriever``) embeds the request and each tool, and
  scores them by cosine similarity, so meaning counts, not shared words. Fusing it with BM25
  ranked worse than ``Dense`` alone: shared words pulled in tools with another meaning.
"""

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import torch

from eve.data.schema import Tool

if TYPE_CHECKING:
    from eve.model.retriever import Retriever
    from eve.tokenizer.bpe import BPETokenizer

K1, B = 1.2, 0.75  # BM25's usual term-saturation and length-normalization constants
_WORD = re.compile(r"[a-z0-9]+")  # snake_case names split at "_"


def words(text: str) -> list[str]:
    """Lowercase words, plurals folded: "lights" and "light" match."""
    out = []
    for word in _WORD.findall(text.lower()):
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        out.append(word)
    return out


def _document(tool: Tool) -> list[str]:
    """What a request is matched against: the name counts twice, being the most telling."""
    parts = [tool.name, tool.name, tool.description]
    for arg in tool.args:
        parts += [arg.name, arg.description, *arg.options]
    return words(" ".join(parts))


class Scorer(Protocol):
    """Ranks tools for a request: one score per tool, higher is a better match."""

    def scores(self, text: str, tools: Sequence[Tool]) -> list[float]: ...


class BM25:
    """Word overlap, weighted by how rare each word is among ``tools`` (Okapi BM25)."""

    def __init__(self) -> None:
        self._docs: dict[Tool, Counter[str]] = {}  # tools are frozen dataclasses: hashable

    def _doc(self, tool: Tool) -> Counter[str]:
        if tool not in self._docs:
            self._docs[tool] = Counter(_document(tool))
        return self._docs[tool]

    def scores(self, text: str, tools: Sequence[Tool]) -> list[float]:
        if not tools:
            return []
        query = set(words(text))
        docs = [self._doc(tool) for tool in tools]
        n = len(docs)
        mean_length = sum(sum(d.values()) for d in docs) / n
        df = Counter(word for doc in docs for word in doc.keys() & query)
        out = []
        for doc in docs:
            length, score = sum(doc.values()), 0.0
            for word in query & doc.keys():
                idf = math.log(1 + (n - df[word] + 0.5) / (df[word] + 0.5))
                tf = doc[word]
                score += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / mean_length))
            out.append(score)
        return out


class Dense:
    """Cosine similarity of the request's and each tool's embedding from a trained
    ``Retriever``. Tool embeddings are cached, so a registry is embedded once."""

    def __init__(self, retriever: "Retriever", tokenizer: "BPETokenizer") -> None:
        self.retriever = retriever  # embeds with dropout off and no gradients
        self.tokenizer = tokenizer
        self._tools: dict[Tool, torch.Tensor] = {}
        self._queries: dict[str, torch.Tensor] = {}  # filled by ``prime``, used up by ``scores``

    def prime(self, texts: Sequence[str] = (), tools: Sequence[Tool] = ()) -> None:
        """Embed many requests and tools in batches ahead of ``scores`` (evaluation)."""
        if new := [t for t in dict.fromkeys(tools) if t not in self._tools]:
            vectors = self.retriever.embed_tools(new, self.tokenizer)
            self._tools.update(zip(new, vectors, strict=True))
        if texts:
            texts = list(dict.fromkeys(texts))
            vectors = self.retriever.embed_queries(texts, self.tokenizer)
            self._queries.update(zip(texts, vectors, strict=True))

    def scores(self, text: str, tools: Sequence[Tool]) -> list[float]:
        if not tools:
            return []
        self.prime(tools=tools)
        query = self._queries.get(text)
        if query is None:
            query = self.retriever.embed_queries([text], self.tokenizer)[0]
        matrix = torch.stack([self._tools[tool] for tool in tools])
        return (matrix @ query).tolist()
