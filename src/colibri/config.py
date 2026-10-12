"""Model configuration and size presets.

One ``ModelConfig`` describes the whole model: the encoder, the value decoder and the
question block (the Jev-style heads). See the plan, milestone M0 / M3.
"""

from dataclasses import asdict, dataclass, replace

# Answers to the intent question (Q1). The order is the class index used by the intent head.
INTENTS = ("call", "ask", "unsupported", "none")


@dataclass(frozen=True)
class ModelConfig:
    # Box 1: tokens and positions
    vocab_size: int = 8192
    max_encoder_len: int = 512  # utterance + up to 5 tool schemas
    max_decoder_len: int = 256  # argument values of up to 5 calls

    # Box 3: transformer body
    d_model: int = 256
    n_heads: int = 4
    n_encoder_layers: int = 6
    n_decoder_layers: int = 3
    ffn_mult: int = 4  # hidden size of the feed-forward layer = ffn_mult * d_model
    dropout: float = 0.1

    # Question block (Jev part): 1 intent question + n_slots "what is call #i?" questions
    n_question_layers: int = 2
    n_slots: int = 5
    max_tools: int = 5
    pointer_dim: int = 64  # size of the query/key vectors the slot head compares

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model={self.d_model} is not divisible by n_heads={self.n_heads}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_intents(self) -> int:
        return len(INTENTS)

    def to_dict(self) -> dict:
        return asdict(self)

    def approx_params(self) -> int:
        """Parameter count from the planned architecture (pre-LN blocks, tied LM head).

        Book ch. 4 counts GPT-2's parameters the same way. M3 checks the real model against it.
        """
        d, ffn = self.d_model, self.ffn_mult * self.d_model
        attention = 4 * d * d + 4 * d  # W_q, W_k, W_v, W_out with biases
        feed_forward = 2 * d * ffn + ffn + d
        norm = 2 * d  # LayerNorm scale and shift

        encoder_layer = attention + feed_forward + 2 * norm
        decoder_layer = 2 * attention + feed_forward + 3 * norm  # self + cross attention
        question_layer = decoder_layer  # a decoder block with the causal mask off

        embeddings = self.vocab_size * d + (self.max_encoder_len + self.max_decoder_len) * d
        queries = (1 + self.n_slots) * d  # learned query vectors
        intent_head = d * self.n_intents + self.n_intents
        slot_head = 2 * d * self.pointer_dim + self.pointer_dim  # W_q, W_k and the STOP key
        final_norms = 3 * norm  # encoder, decoder, question block

        return (
            embeddings
            + self.n_encoder_layers * encoder_layer
            + self.n_decoder_layers * decoder_layer
            + self.n_question_layers * question_layer
            + queries
            + intent_head
            + slot_head
            + final_norms
        )


PRESETS: dict[str, ModelConfig] = {
    # For tests only: fast on CPU, same structure as the real models.
    "tiny": ModelConfig(
        d_model=64,
        n_heads=2,
        n_encoder_layers=2,
        n_decoder_layers=1,
        n_question_layers=1,
        pointer_dim=16,
        dropout=0.0,
    ),
    # First real model, ~12M parameters.
    "small": ModelConfig(),
    # If the data supports it, ~32M parameters.
    "base": ModelConfig(d_model=384, n_heads=6, n_encoder_layers=8, n_decoder_layers=4),
}


def preset(name: str, **overrides) -> ModelConfig:
    """Return a preset, optionally with some fields changed: ``preset("small", dropout=0.0)``."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return replace(PRESETS[name], **overrides)
