"""The ``eve`` command line. Each command group lives in its own module: ``train`` (run,
compare), ``inspect`` (ask, evaluate, calibrate, fill), ``data`` (tokenizer, corpus, data) and
``publish`` (info, export, retriever); each module registers its commands and holds their
handlers and tables."""

import argparse
import logging
import sys

from eve.cli import data, datagen, inspect, publish, train


class _StdoutHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Writes plain lines to whatever ``sys.stdout`` is when a record arrives, like ``print``
    (pytest and redirects swap it). Warnings and errors start with their level: "warning: …"."""

    def emit(self, record: logging.LogRecord) -> None:
        """Write to the current ``sys.stdout``, not the one there was at setup."""
        self.stream = sys.stdout
        super().emit(record)

    def format(self, record: logging.LogRecord) -> str:
        """The message alone, prefixed with the level for warnings and errors."""
        message = record.getMessage()
        if record.levelno >= logging.WARNING:
            return f"{record.levelname.lower()}: {message}"
        return message


def setup_logging() -> None:
    """Show the library's status lines (the ``eve`` loggers, INFO and up) on stdout."""
    logger = logging.getLogger("eve")
    logger.handlers = [_StdoutHandler()]  # replaced, not added: main() may run more than once
    logger.setLevel(logging.INFO)
    logger.propagate = False


def build_parser() -> argparse.ArgumentParser:
    """The ``eve`` command line: one subcommand per stage, each with a ``run`` function."""
    parser = argparse.ArgumentParser(prog="eve")
    commands = parser.add_subparsers(dest="command", required=True)
    publish.add_info(commands)
    data.add_tokenizer(commands)
    data.add_corpus(commands)
    train.add_run(commands)
    inspect.add_ask(commands)
    inspect.add_evaluate(commands)
    inspect.add_calibrate(commands)
    train.add_compare(commands)
    publish.add_export(commands)
    publish.add_retriever(commands)
    inspect.add_fill(commands)
    data.add_data(commands)
    datagen.add_datagen(commands)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the chosen subcommand."""
    args = build_parser().parse_args(argv)
    setup_logging()
    try:
        args.run(args)
    except ModuleNotFoundError as error:
        if error.name != "pyarrow":
            raise
        hint = "this command needs the training extras: pip install 'eve-lm[train]'"
        raise SystemExit(hint) from None
