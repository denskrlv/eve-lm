from collections.abc import Sequence

import numpy as np


def pick[T](rng: np.random.Generator, options: Sequence[T]) -> T:
    """One of ``options``, uniformly."""
    return options[int(rng.integers(len(options)))]


def shuffled[T](rng: np.random.Generator, items: Sequence[T]) -> list[T]:
    """``items`` in a random order."""
    return [items[i] for i in rng.permutation(len(items))]
