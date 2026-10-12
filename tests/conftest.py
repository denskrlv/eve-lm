from pathlib import Path

import pytest

from colibri.data.schema import load_library
from colibri.tokenizer.bpe import BPETokenizer

ROOT = Path(__file__).parents[1]

TEXT = """\
The kitchen is the room where food is prepared. Most kitchens have a stove, an oven and a
sink. In 2024, about 85% of homes had a microwave. Turn the lights off when you leave a room,
and set the thermostat to 20 degrees at night to save energy. Music sounds better in the living
room; podcasts are nice in the bedroom. Remind yourself to call your friends, and don't forget
to lock the front door before you go to sleep. The weather tomorrow will be sunny in Berlin.
"""


@pytest.fixture(scope="session")
def library():
    return load_library(ROOT / "data" / "tools")


@pytest.fixture(scope="session")
def tokenizer(library):
    texts = [(TEXT, 5)] + [(" " + t.render(), 5) for t in library.values()]
    return BPETokenizer.train(texts, vocab_size=600)
