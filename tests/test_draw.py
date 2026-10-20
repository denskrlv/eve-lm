import numpy as np

from eve.data.draw import pick, shuffled


def test_pick_draws_like_indexing_with_integers() -> None:
    options = ("a", "b", "c", "d")
    rng = np.random.default_rng(7)
    picked = [pick(rng, options) for _ in range(5)]
    reference = np.random.default_rng(7)
    assert picked == [options[reference.integers(len(options))] for _ in range(5)]


def test_shuffled_is_a_permutation_in_the_rng_order() -> None:
    items = list(range(10))
    out = shuffled(np.random.default_rng(3), items)
    assert sorted(out) == items
    assert out == [items[i] for i in np.random.default_rng(3).permutation(len(items))]
