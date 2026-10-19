import numpy as np
from torch.utils.data import Dataset

from colibri.tokenizer.bpe import DEC, EOS, N_SENTINELS, SENTINEL0, sentinel


def span_counts(length: int, noise_density: float, mean_span: float) -> tuple[int, int]:
    """How many tokens to drop from a window of ``length`` tokens, and in how many spans."""
    n_noise = min(max(round(length * noise_density), 1), length - 1)
    n_spans = max(round(n_noise / mean_span), 1)
    return n_noise, n_spans


def raw_length(input_len: int, noise_density: float = 0.15, mean_span: float = 3.0) -> int:
    """The window length whose corrupted version is exactly ``input_len`` tokens long."""
    length = input_len
    while True:
        n_noise, n_spans = span_counts(length, noise_density, mean_span)
        corrupted = length - n_noise + n_spans
        if corrupted == input_len:
            return length
        if corrupted > input_len:
            raise ValueError(f"no window corrupts to exactly {input_len} tokens")
        length += 1


def _split(total: int, parts: int, rng: np.random.Generator) -> np.ndarray:
    """Split ``total`` into ``parts`` random positive lengths."""
    cuts = np.sort(rng.choice(np.arange(1, total), parts - 1, replace=False))
    return np.diff(np.concatenate([[0], cuts, [total]]))


def noise_mask(
    length: int, noise_density: float, mean_span: float, rng: np.random.Generator
) -> np.ndarray:
    """Boolean mask of the tokens to drop. Kept and dropped spans alternate, kept first.

    This is T5's ``random_spans_noise_mask``: the number of dropped tokens and spans is fixed, so
    every window has the same encoder and decoder length (one shape for MPS to compile).
    """
    n_noise, n_spans = span_counts(length, noise_density, mean_span)
    noise_lengths = _split(n_noise, n_spans, rng)
    keep_lengths = _split(length - n_noise, n_spans, rng)
    is_noise = np.stack([np.zeros(n_spans, bool), np.ones(n_spans, bool)], axis=1).ravel()
    lengths = np.stack([keep_lengths, noise_lengths], axis=1).ravel()
    return np.repeat(is_noise, lengths)


def corrupt(tokens: np.ndarray, mask: np.ndarray) -> tuple[list[int], list[int]]:
    """Encoder input (spans replaced by sentinels) and target (sentinel, span, sentinel, ...)."""
    inputs: list[int] = []
    targets: list[int] = []
    span = -1
    for i, (token, dropped) in enumerate(zip(tokens.tolist(), mask.tolist(), strict=True)):
        if dropped:
            if i == 0 or not mask[i - 1]:  # a new span starts
                span += 1
                inputs.append(sentinel(span))
                targets.append(sentinel(span))
            targets.append(token)
        else:
            inputs.append(token)
    return inputs, targets


def uncorrupt(inputs: list[int], targets: list[int]) -> list[int]:
    """Put the spans from ``targets`` back into ``inputs``. The inverse of ``corrupt``."""
    spans: dict[int, list[int]] = {}
    current: list[int] = []
    for token in targets:
        if SENTINEL0 <= token < SENTINEL0 + N_SENTINELS:
            current = spans.setdefault(token, [])
        elif token != EOS:
            current.append(token)
    out: list[int] = []
    for token in inputs:
        out.extend(spans.get(token, [token]))
    return out


class SpanCorruptionDataset(Dataset[dict[str, list[int]]]):
    """Non-overlapping windows over the token stream, each corrupted with its own fixed seed.

    Item i is always the same sample (seeded by ``(seed, i)``), so runs can be repeated and
    resumed. Returns ``enc_ids`` (input_len tokens), ``dec_ids`` = <dec> + target and
    ``dec_labels`` = target + <eos>: the decoder input shifted by one, as in book ch. 2.6.
    """

    def __init__(
        self,
        tokens: np.ndarray,
        input_len: int = 512,
        noise_density: float = 0.15,
        mean_span: float = 3.0,
        seed: int = 0,
    ):
        self.tokens = tokens
        self.noise_density = noise_density
        self.mean_span = mean_span
        self.seed = seed
        self.window = raw_length(input_len, noise_density, mean_span)
        n_spans = span_counts(self.window, noise_density, mean_span)[1]
        if n_spans > N_SENTINELS:
            raise ValueError(f"{n_spans} spans per window but only {N_SENTINELS} sentinels")

    def __len__(self) -> int:
        return len(self.tokens) // self.window

    def __getitem__(self, i: int) -> dict[str, list[int]]:
        if not 0 <= i < len(self):
            raise IndexError(i)
        window = np.asarray(self.tokens[i * self.window : (i + 1) * self.window], dtype=np.int64)
        rng = np.random.default_rng((self.seed, i))
        inputs, targets = corrupt(
            window, noise_mask(self.window, self.noise_density, self.mean_span, rng)
        )
        return {"enc_ids": inputs, "dec_ids": [DEC, *targets], "dec_labels": [*targets, EOS]}
