import numpy as np

from colibri.data.draw import pick, shuffled


def test_pick_draws_like_indexing_with_integers():
    options = ("a", "b", "c", "d")
    picked = [pick(np.random.default_rng(7), options) for _ in range(3)]
    rng = np.random.default_rng(7)
    assert picked == [options[rng.integers(len(options))]] * 3


def test_shuffled_is_a_permutation_in_the_rng_order():
    items = list(range(10))
    out = shuffled(np.random.default_rng(3), items)
    assert sorted(out) == items
    assert out == [items[i] for i in np.random.default_rng(3).permutation(len(items))]
