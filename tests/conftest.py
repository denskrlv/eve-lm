import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

from eve.data.schema import load_library
from eve.tokenizer.bpe import BPETokenizer

ROOT = Path(__file__).parents[1]

TEXT = """\
The kitchen is the room where food is prepared. Most kitchens have a stove, an oven and a
sink. In 2024, about 85% of homes had a microwave. Turn the lights off when you leave a room,
and set the thermostat to 20 degrees at night to save energy. Music sounds better in the living
room; podcasts are nice in the bedroom. Remind yourself to call your friends, and don't forget
to lock the front door before you go to sleep. The weather tomorrow will be sunny in Berlin.
"""


@pytest.fixture(autouse=True)
def restore_eve_logger() -> Iterator[None]:
    """Undo ``setup_logging`` (called by ``main`` too) after each test, so no test sees another
    test's handler, level or ``propagate=False`` (which would hide records from caplog)."""
    logger = logging.getLogger("eve")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    yield
    logger.handlers = handlers
    logger.setLevel(level)
    logger.propagate = propagate


@pytest.fixture(scope="session")
def library():
    return load_library(ROOT / "data" / "tools")


@pytest.fixture(scope="session")
def tokenizer(library):
    texts = [(TEXT, 5)] + [(" " + t.render(), 5) for t in library.values()]
    return BPETokenizer.train(texts, vocab_size=600)


@pytest.fixture
def copy_flags(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """The ``non_blocking`` flag of every ``Tensor.to`` call made during the test."""
    flags: list[bool] = []
    original = torch.Tensor.to

    def spy(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        flags.append(bool(kwargs.get("non_blocking", False)))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", spy)
    return flags
