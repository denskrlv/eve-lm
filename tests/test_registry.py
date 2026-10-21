import pytest

from eve import registry


@pytest.fixture
def clean(monkeypatch):
    """An empty registry, so test components don't leak into other tests."""
    monkeypatch.setattr(registry, "_components", {kind: {} for kind in registry.KINDS})
    monkeypatch.setattr(registry, "_loaded", True)


def test_function_kinds_get_their_params_bound(clean):
    @registry.register("loss", "scaled")
    def scaled(model, batch, scale=1.0):
        return {"loss": batch * scale}

    assert registry.build("loss", "scaled") is scaled
    assert registry.build("loss", "scaled", scale=3.0)(None, 2.0) == {"loss": 6.0}


def test_other_kinds_are_called_with_their_params(clean):
    @registry.register("optimizer", "fake")
    class Fake:
        def __init__(self, lr=0.1):
            self.lr = lr

    assert registry.build("optimizer", "fake", lr=0.5).lr == 0.5
    assert registry.names("optimizer") == ["fake"]
    assert registry.name_of("optimizer", Fake) == "fake"


def test_unknown_names_and_kinds_say_what_exists(clean):
    registry.register("loss", "a")(lambda m, b: {})
    with pytest.raises(KeyError, match=r"no loss called 'b'; registered: \['a'\]"):
        registry.get("loss", "b")
    with pytest.raises(KeyError, match="unknown component kind"):
        registry.register("lossy", "a")


def test_a_name_is_registered_once(clean):
    registry.register("loss", "a")(lambda m, b: {})
    with pytest.raises(ValueError, match="already registered"):
        registry.register("loss", "a")(lambda m, b: {})
