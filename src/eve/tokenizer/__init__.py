"""Box 1: the byte-level BPE tokenizer; and BERT's WordPiece, for a retriever that starts from
MiniLM (``eve.model.bert``)."""

from typing import Any

from eve.tokenizer.bpe import BPETokenizer
from eve.tokenizer.wordpiece import FORMAT as WORDPIECE_FORMAT
from eve.tokenizer.wordpiece import WordPieceTokenizer

Tokenizer = BPETokenizer | WordPieceTokenizer


def tokenizer_from_dict(data: dict[str, Any]) -> Tokenizer:
    """The tokenizer a checkpoint saved (``to_dict``), whichever kind it is."""
    if data.get("format") == WORDPIECE_FORMAT:
        return WordPieceTokenizer.from_dict(data)
    return BPETokenizer.from_dict(data)
