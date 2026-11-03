"""What several command groups share: count parsing and the arguments many commands take."""

import argparse

from eve import defaults
from eve.train.config import TrainConfig

TRAIN_DEFAULTS = TrainConfig(steps=1)  # the CLI's defaults are TrainConfig's own

type Subcommands = argparse._SubParsersAction[argparse.ArgumentParser]


def count(text: str) -> int:
    """'250M' -> 250_000_000, '2k' -> 2000, '1000' -> 1000."""
    scale = {"k": 10**3, "m": 10**6, "b": 10**9}.get(text[-1].lower(), 1)
    return int(float(text[:-1] if scale > 1 else text) * scale)


def add_device_arg(parser: argparse.ArgumentParser) -> None:
    """``--device``: a torch device name; by default the fastest there is."""
    parser.add_argument("--device", default=None)


def add_json_arg(parser: argparse.ArgumentParser, help_: str) -> None:
    """``--json``: print machine-readable output instead of tables."""
    parser.add_argument("--json", action="store_true", help=help_)


def add_model_arg(parser: argparse.ArgumentParser) -> None:
    """The optional ``model`` positional of the commands that score a fine-tuned model."""
    parser.add_argument(
        "model", nargs="?", default=defaults.FINETUNED_MODEL, help="a model.pt or a run directory"
    )
