from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from colibri.config import INTENTS, ModelConfig
from colibri.data.draw import shuffled
from colibri.data.schema import ASK_VALUE, MAX_ARGS, Arg, SchemaError, Tool
from colibri.tokenizer.bpe import (
    ABSENT,
    ARG,
    ASK,
    BOS,
    CALL,
    DEC,
    END,
    EOS,
    PAD,
    TOOL,
    TOOLS,
    UTT,
    VAL,
    BPETokenizer,
)

IGNORE = -100
ENCODER_BUCKETS = (128, 256, 512)
DECODER_BUCKETS = (64, 128, 256)


@dataclass(frozen=True)
class Call:
    tool: str
    args: dict[str, Any]


@dataclass(frozen=True)
class TaskExample:
    """One line of ``data/task/*.jsonl`` (written in M6), with the tools resolved to schemas."""

    utterance: str
    tools: tuple[Tool, ...]
    intent: str
    calls: tuple[Call, ...] = ()

    def __post_init__(self) -> None:
        if self.intent not in INTENTS:
            raise SchemaError(f"intent must be one of {INTENTS}, got {self.intent!r}")
        if not self.tools:
            raise SchemaError("an example needs at least one tool")
        names = [t.name for t in self.tools]
        if len(set(names)) != len(names):
            raise SchemaError(f"a tool appears twice in the prompt: {names}")
        if self.intent == "call" and not self.calls:
            raise SchemaError("intent 'call' needs at least one call")
        if self.intent in ("unsupported", "none") and self.calls:
            raise SchemaError(f"intent {self.intent!r} has no calls")
        by_name = {t.name: t for t in self.tools}
        for call in self.calls:
            if call.tool not in by_name:
                raise SchemaError(f"call to {call.tool!r}, which is not in the prompt")
            errors = by_name[call.tool].check_call(call.args, allow_ask=self.intent == "ask")
            if errors:
                raise SchemaError(f"call to {call.tool}: {'; '.join(errors)}")

    @classmethod
    def from_json(
        cls, data: dict[str, Any], library: dict[str, Tool] | None = None
    ) -> "TaskExample":
        """``{"utterance", "tools", "intent", "calls": [{"tool", "args"}]}``.

        A tool is a name looked up in ``library``, or a whole schema (a dict, as in
        ``Tool.to_dict``) for tools from outside the library, like external datasets.
        """
        return cls(
            utterance=data["utterance"],
            tools=tuple(
                Tool.from_dict(t) if isinstance(t, dict) else _named(t, library)
                for t in data["tools"]
            ),
            intent=data["intent"],
            calls=tuple(Call(c["tool"], dict(c.get("args", {}))) for c in data.get("calls", ())),
        )

    def to_json(self, inline: bool = False) -> dict[str, Any]:
        """With ``inline``, tools are written as whole schemas instead of names."""
        return {
            "utterance": self.utterance,
            "tools": [t.to_dict() if inline else t.name for t in self.tools],
            "intent": self.intent,
            "calls": [{"tool": c.tool, "args": c.args} for c in self.calls],
        }


def _named(name: str, library: dict[str, Tool] | None) -> Tool:
    if library is None:
        raise TypeError(f"tool {name!r} is a name, but no library was given to look it up")
    return library[name]


@dataclass
class Rendered:
    enc_ids: list[int]
    tool_pos: list[int]  # position of each <tool> marker, in prompt order
    arg_pos: list[list[int]]  # position of each <arg> marker, per tool
    intent: int  # index into INTENTS
    slots: list[int]  # per slot: index of the tool in the prompt, STOP (= max_tools) or IGNORE
    dec_ids: list[int]  # decoder input
    dec_labels: list[int]  # next decoder token where it gets loss, else IGNORE
    tool_names: list[str]  # prompt order, for debugging


def _encoder(
    utterance: str, tools: Sequence[Tool], tokenizer: BPETokenizer, cfg: ModelConfig
) -> tuple[list[int], list[int], list[list[int]]]:
    """Encoder ids, the position of each ``<tool>`` marker, and each tool's ``<arg>`` positions.

    User text never gets special tokens. Every piece of text starts with a space, so a word is
    the same token in the utterance, the schema and a decoder value.
    """
    enc = [BOS, UTT, *tokenizer.encode(" " + utterance), TOOLS]
    tool_pos, arg_pos = [], []
    for tool in tools:
        ids = [i for part in tool.render_parts() for i in tokenizer.encode(part, specials=True)]
        tool_pos.append(len(enc))
        arg_pos.append([len(enc) + j for j, t in enumerate(ids) if t == ARG])
        assert ids[0] == TOOL and len(arg_pos[-1]) == len(tool.args)
        enc.extend(ids)
    enc.append(EOS)
    if len(enc) > cfg.max_encoder_len:
        raise ValueError(f"encoder input is {len(enc)} tokens, max {cfg.max_encoder_len}")
    return enc, tool_pos, arg_pos


def _slot_labels(example: TaskExample, tools: Sequence[Tool], cfg: ModelConfig) -> list[int]:
    """Question labels (M7): per slot, the index of its call's tool in the prompt, then STOP."""
    if example.intent == "ask" and not example.calls:
        return [IGNORE] * cfg.n_slots  # the tool itself is unclear: no slot loss
    index = {t.name: k for k, t in enumerate(tools)}
    slots = [index[c.tool] for c in example.calls]
    return slots + [cfg.max_tools] * (cfg.n_slots - len(slots))  # max_tools = STOP


def _value_ids(arg: Arg, call: Call, intent: str, tokenizer: BPETokenizer) -> list[int]:
    """What the decoder writes for one argument: <absent>, <ask>, or the value then <end>."""
    if arg.name not in call.args:
        return [ABSENT]
    if call.args[arg.name] == ASK_VALUE and intent == "ask" and arg.required:
        return [ASK]
    return [*tokenizer.encode(" " + arg.format(call.args[arg.name])), END]


def _decoder(
    example: TaskExample, tools: Sequence[Tool], tokenizer: BPETokenizer, cfg: ModelConfig
) -> tuple[list[int], list[int]]:
    """Decoder input and labels (M8): structure copied from the schema (no loss), values
    predicted (loss)."""
    seq, loss = [DEC], [False]

    def add(ids: Sequence[int], predicted: bool) -> None:
        seq.extend(ids)
        loss.extend([predicted] * len(ids))

    by_name = {t.name: t for t in tools}
    for call in example.calls:
        tool = by_name[call.tool]
        add([CALL, *tokenizer.encode(" " + tool.name)], False)
        for arg in tool.args:
            add([ARG, *tokenizer.encode(" " + arg.name), VAL], False)
            add(_value_ids(arg, call, example.intent, tokenizer), True)
    add([EOS], False)
    dec_ids = seq[:-1]
    dec_labels = [
        t if predicted else IGNORE for t, predicted in zip(seq[1:], loss[1:], strict=True)
    ]
    if len(dec_ids) > cfg.max_decoder_len:
        raise ValueError(f"decoder input is {len(dec_ids)} tokens, max {cfg.max_decoder_len}")
    return dec_ids, dec_labels


def render(
    example: TaskExample,
    tokenizer: BPETokenizer,
    cfg: ModelConfig,
    rng: np.random.Generator | None = None,
) -> Rendered:
    """Tokenize an example and build its labels. With ``rng``, the tools are shuffled first."""
    tools = list(example.tools)
    if rng is not None:
        tools = shuffled(rng, tools)
    if len(tools) > cfg.max_tools:
        raise ValueError(f"{len(tools)} tools, but the model takes at most {cfg.max_tools}")
    if len(example.calls) > cfg.n_slots:
        raise ValueError(f"{len(example.calls)} calls, but the model has {cfg.n_slots} slots")
    enc, tool_pos, arg_pos = _encoder(example.utterance, tools, tokenizer, cfg)
    slots = _slot_labels(example, tools, cfg)
    dec_ids, dec_labels = _decoder(example, tools, tokenizer, cfg)
    return Rendered(
        enc_ids=enc,
        tool_pos=tool_pos,
        arg_pos=arg_pos,
        intent=INTENTS.index(example.intent),
        slots=slots,
        dec_ids=dec_ids,
        dec_labels=dec_labels,
        tool_names=[t.name for t in tools],
    )


class TaskDataset(Dataset[Rendered]):
    """Renders examples on the fly, with the tool order shuffled differently each epoch.

    Call ``set_epoch(e)`` before each epoch (with ``num_workers > 0``, before creating the
    DataLoader iterator, so the workers get the new value).
    """

    def __init__(
        self,
        examples: Sequence[TaskExample],
        tokenizer: BPETokenizer,
        cfg: ModelConfig,
        seed: int = 0,
        shuffle_tools: bool = True,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.seed = seed
        self.shuffle_tools = shuffle_tools
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Rendered:
        rng = np.random.default_rng((self.seed, self.epoch, i)) if self.shuffle_tools else None
        return render(self.examples[i], self.tokenizer, self.cfg, rng)


# ---- batching -------------------------------------------------------------------------------


def bucket(length: int, buckets: Sequence[int]) -> int:
    """The smallest bucket that fits ``length``."""
    for size in buckets:
        if length <= size:
            return size
    raise ValueError(f"length {length} is longer than the largest bucket {buckets[-1]}")


def pad(seqs: Sequence[Sequence[int]], length: int, value: int) -> torch.Tensor:
    out = torch.full((len(seqs), length), value, dtype=torch.long)
    for i, seq in enumerate(seqs):
        out[i, : len(seq)] = torch.as_tensor(seq, dtype=torch.long)
    return out


def _sequences(
    batch: Sequence[Any], get: Callable[[Any, str], list[int]]
) -> dict[str, torch.Tensor]:
    """Pad the encoder and decoder sequences of a batch to their buckets."""
    enc = [get(b, "enc_ids") for b in batch]
    dec = [get(b, "dec_ids") for b in batch]
    labels = [get(b, "dec_labels") for b in batch]
    le = bucket(max(map(len, enc)), ENCODER_BUCKETS)
    ld = bucket(max(map(len, dec)), DECODER_BUCKETS)
    enc_ids, dec_ids = pad(enc, le, PAD), pad(dec, ld, PAD)
    return {
        "enc_ids": enc_ids,
        "enc_mask": enc_ids != PAD,  # True = a real token
        "dec_ids": dec_ids,
        "dec_mask": pad([[1] * len(d) for d in dec], ld, 0).bool(),
        "dec_labels": pad(labels, ld, IGNORE),
    }


def collate_pretrain(batch: Sequence[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
    return _sequences(batch, lambda b, k: b[k])


class TaskCollator:
    """Pads rendered examples into a batch. Tools are always padded to ``max_tools`` and
    arguments to ``MAX_ARGS``, so the question block sees one shape."""

    def __init__(self, cfg: ModelConfig):
        self.max_tools = cfg.max_tools

    def __call__(self, batch: Sequence[Rendered]) -> dict[str, torch.Tensor]:
        out = _sequences(batch, getattr)
        n_examples, n_tools = len(batch), self.max_tools
        tool_pos = torch.zeros(n_examples, n_tools, dtype=torch.long)
        tool_mask = torch.zeros(n_examples, n_tools, dtype=torch.bool)
        arg_pos = torch.zeros(n_examples, n_tools, MAX_ARGS, dtype=torch.long)
        arg_mask = torch.zeros(n_examples, n_tools, MAX_ARGS, dtype=torch.bool)
        for i, item in enumerate(batch):
            tool_pos[i, : len(item.tool_pos)] = torch.tensor(item.tool_pos)
            tool_mask[i, : len(item.tool_pos)] = True
            for tool_index, positions in enumerate(item.arg_pos):
                arg_pos[i, tool_index, : len(positions)] = torch.tensor(positions, dtype=torch.long)
                arg_mask[i, tool_index, : len(positions)] = True
        out |= {
            "tool_pos": tool_pos,
            "tool_mask": tool_mask,
            "arg_pos": arg_pos,
            "arg_mask": arg_mask,
            "intent": torch.tensor([item.intent for item in batch]),
            "slots": torch.tensor([item.slots for item in batch]),
        }
        return out
