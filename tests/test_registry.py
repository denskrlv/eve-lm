from types import SimpleNamespace

import pytest

from eve import registry


@pytest.fixture
def clean(monkeypatch):
    """An empty registry, so test components don't leak into other tests."""
    monkeypatch.setattr(registry, "_components", {kind: {} for kind in registry.KINDS})
    monkeypatch.setattr(registry, "_loaded", set(registry.KINDS))


def test_build_binds_the_params(clean):
    @registry.register("loss", "scaled")
    def scaled(model, batch, scale=1.0):
        return {"loss": batch * scale}

    assert registry.build("loss", "scaled") is scaled
    assert registry.build("loss", "scaled", scale=3.0)(None, 2.0) == {"loss": 6.0}


def test_a_class_is_bound_too(clean):
    @registry.register("optimizer", "fake")
    class Fake:
        def __init__(self, model, lr=0.1):
            self.model, self.lr = model, lr

    assert registry.build("optimizer", "fake", lr=0.5)("model").lr == 0.5
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


def test_loading_a_model_kind_skips_the_training_modules(monkeypatch):
    imported = []

    def fake_import(name):
        imported.append(name)
        return SimpleNamespace(__path__=[])  # an empty package for eve.tasks

    monkeypatch.setattr(registry, "_loaded", set())
    monkeypatch.setattr(registry.importlib, "import_module", fake_import)
    registry.load_builtins("model")
    assert imported == ["eve.model.eve"]
    registry.load_builtins("loss")
    assert "eve.tasks" in imported
