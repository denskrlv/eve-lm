"""Box 1: the byte-level BPE tokenizer; BERT's WordPiece, for an encoder that starts from MiniLM
(``eve.model.bert``); and the pair of them an Eve with that encoder reads and writes with."""

from dataclasses import dataclass
from typing import Any

from eve.tokenizer.bpe import BPETokenizer
from eve.tokenizer.wordpiece import FORMAT as WORDPIECE_FORMAT
from eve.tokenizer.wordpiece import WordPieceTokenizer

PAIR_FORMAT = "eve-pair-1"


@dataclass(frozen=True)
class PairTokenizer:
    """An Eve whose encoder is MiniLM: the request and schemas in MiniLM's WordPiece, the
    decoder's values in Eve's BPE, which keeps their case and every character ("50%", not
    "50 %"). Both put Eve's special tokens at the same ids."""

    encoder: WordPieceTokenizer
    decoder: BPETokenizer

    @property
    def vocab_size(self) -> int:
        """The decoder's: the size ``ModelConfig.vocab_size`` counts."""
        return self.decoder.vocab_size

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": PAIR_FORMAT,
            "encoder": self.encoder.to_dict(),
            "decoder": self.decoder.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PairTokenizer":
        if data.get("format") != PAIR_FORMAT:
            raise ValueError("not an Eve tokenizer pair")
        return cls(
            WordPieceTokenizer.from_dict(data["encoder"]), BPETokenizer.from_dict(data["decoder"])
        )


TextTokenizer = BPETokenizer | WordPieceTokenizer  # turns text into ids
TaskTokenizer = BPETokenizer | PairTokenizer  # renders Eve's prompts and values
Tokenizer = BPETokenizer | WordPieceTokenizer | PairTokenizer  # what a checkpoint holds


def encoder_side(tokenizer: TaskTokenizer) -> TextTokenizer:
    """The tokenizer of the encoder's input."""
    return tokenizer.encoder if isinstance(tokenizer, PairTokenizer) else tokenizer


def decoder_side(tokenizer: TaskTokenizer) -> BPETokenizer:
    """The tokenizer of the decoder's sequence."""
    return tokenizer.decoder if isinstance(tokenizer, PairTokenizer) else tokenizer


def tokenizer_from_dict(data: dict[str, Any]) -> Tokenizer:
    """The tokenizer a checkpoint saved (``to_dict``), whichever kind it is."""
    if data.get("format") == WORDPIECE_FORMAT:
        return WordPieceTokenizer.from_dict(data)
    if data.get("format") == PAIR_FORMAT:
        return PairTokenizer.from_dict(data)
    return BPETokenizer.from_dict(data)
