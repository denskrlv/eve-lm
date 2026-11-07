from typing import Any

from eve.data.generate.actions import part_actions, row_actions
from eve.split.words import Span


def _texts(text: str, actions: list[list[Span]]) -> list[str]:
    return [" ".join(text[a:b] for a, b in action) for action in actions]


def test_part_actions_skip_connectors_and_decoration() -> None:
    text = "Please turn off the fan, and then lock the door thanks"
    actions = part_actions(text, ["turn off the fan", "lock the door"])
    assert actions is not None
    assert _texts(text, actions) == ["turn off the fan", "lock the door"]


def test_part_actions_find_a_capitalized_first_part() -> None:
    text = "Turn off the fan and lock the door"
    actions = part_actions(text, ["turn off the fan", "lock the door"])
    assert actions is not None and _texts(text, actions) == ["Turn off the fan", "lock the door"]


def test_part_actions_missing_part() -> None:
    assert part_actions("turn off the fan", ["turn off the fan", "lock the door"]) is None


def _row(source: str, n_calls: int, **extra: Any) -> dict[str, Any]:
    call = {"tool": "set_lights", "args": {"room": "kitchen", "on": False}}
    text = "lights off"
    return {"utterance": text, "source": source, "calls": [call] * n_calls, **extra}


def test_row_actions_rules() -> None:
    assert row_actions(_row("massive", 1)) == [[(0, 10)]]
    assert row_actions(_row("template", 0)) == [[(0, 10)]]
    assert row_actions(_row("xlam", 1)) is None  # xLAM rows are labeled by the converter
    assert row_actions(_row("chat", 2)) is None  # no parts to cut at
    assert row_actions(_row("chat", 2, actions=[[[0, 6]], [[7, 10]]])) == [[(0, 6)], [(7, 10)]]


# ---- xLAM rules -----------------------------------------------------------------------------

from eve.data.generate.actions import xlam_actions  # noqa: E402
from eve.data.render import Call, TaskExample  # noqa: E402
from eve.data.schema import Arg, Tool  # noqa: E402


def _tool(name: str, *args: str) -> Tool:
    return Tool(name, f"The {name} tool.", tuple(Arg(a, "str") for a in args))


def _xlam(text: str, *calls: tuple[str, dict[str, Any]]) -> TaskExample:
    tools = {name: _tool(name, *args) for name, args in calls}
    return TaskExample(text, tuple(tools.values()), "call", tuple(Call(n, a) for n, a in calls))


def _xlam_texts(example: TaskExample) -> list[str] | None:
    from eve.split.words import action_text, word_spans, words_in

    actions = xlam_actions(example)
    if actions is None:
        return None
    words = word_spans(example.utterance)
    # A word is a tokenizer piece, and a punctuation run like "'?" is one piece: an action
    # may end with punctuation its spans left out. Eve doesn't mind; the tests ignore it.
    texts = [action_text(example.utterance, words, words_in(words, a)) for a in actions]
    return [t.rstrip("?.!") for t in texts]


def test_xlam_two_sentences() -> None:
    text = (
        "What are the gas fee funding wallet details for the given ID "
        "'dfe02338-43ae-453f-bd51-6f0ea98e4df9'? Also, what are the most important resources "
        "that sustain the economy of Algeria?"
    )
    ex = _xlam(
        text,
        ("get_gas_fee_funding_wallets", {"is_id": "dfe02338-43ae-453f-bd51-6f0ea98e4df9"}),
        ("resources_per_country", {"country_name": "Algeria"}),
    )
    assert _xlam_texts(ex) == [
        "What are the gas fee funding wallet details for the given ID "
        "'dfe02338-43ae-453f-bd51-6f0ea98e4df9'",
        "what are the most important resources that sustain the economy of Algeria",
    ]


def test_xlam_shared_verb_is_copied() -> None:
    text = (
        "Fetch the cat breeds from Russia and the latest Under-58kg Taekwondo rankings for all "
        "countries."
    )
    ex = _xlam(text, ("origin", {"origin": "Russia"}), ("get_u_58_athlete_ranking", {}))
    assert _xlam_texts(ex) == [
        "Fetch the cat breeds from Russia",
        "Fetch the latest Under-58kg Taekwondo rankings for all countries",
    ]


def test_xlam_bare_value_gets_the_previous_head() -> None:
    ex = _xlam(
        "What is the weather in Oslo and Bergen?",
        ("weather", {"city": "Oslo"}),
        ("weather", {"city": "Bergen"}),
    )
    assert _xlam_texts(ex) == ["What is the weather in Oslo", "What is the weather in Bergen"]


def test_xlam_and_inside_one_call() -> None:
    text = (
        "Can you tell me the peers for 'WMT' and 'TGT'? Also, what about the peers for 'MCD' "
        "and 'YUM'?"
    )
    ex = _xlam(
        text,
        ("symbols_get_peers", {"symbol": "WMT,TGT"}),
        ("symbols_get_peers", {"symbol": "MCD,YUM"}),
    )
    texts = _xlam_texts(ex)
    assert texts is not None and len(texts) == 2
    assert "'WMT' and 'TGT'" in texts[0] and "MCD" not in texts[0]


def test_xlam_three_calls() -> None:
    text = (
        "What are the main resources that sustain Ethiopia's economy? Also, find out the "
        "resources for Tanzania. Additionally, search for 'python programming' videos on YouTube."
    )
    ex = _xlam(
        text,
        ("resources_per_country", {"country_name": "Ethiopia"}),
        ("resources_per_country", {"country_name": "Tanzania"}),
        ("search", {"q": "python programming"}),
    )
    assert _xlam_texts(ex) == [
        "What are the main resources that sustain Ethiopia's economy",
        "find out the resources for Tanzania",
        "search for 'python programming' videos on YouTube",
    ]


def test_xlam_two_connectors_between_anchors_is_dropped() -> None:
    ex = _xlam(
        "Get the scores for Tom and the wind and rain forecast in Oslo",
        ("get_scores", {"player": "Tom"}),
        ("forecast", {"city": "Oslo"}),
    )
    assert xlam_actions(ex) is None


def test_xlam_single_call_is_one_action() -> None:
    ex = _xlam("Weather in Oslo and Bergen, please", ("weather", {"city": "Oslo and Bergen"}))
    assert xlam_actions(ex) == [[(0, len(ex.utterance))]]


def test_xlam_value_not_found_is_dropped() -> None:
    ex = _xlam(
        "Weather in Oslo and then Bergen", ("weather", {"city": "Oslo"}), ("w", {"c": "Rome"})
    )
    assert xlam_actions(ex) is None


def test_xlam_preposition_and_value_gets_the_previous_head() -> None:
    ex = _xlam(
        "Weather in Oslo and also in Bergen?",
        ("weather", {"city": "Oslo"}),
        ("weather", {"city": "Bergen"}),
    )
    assert _xlam_texts(ex) == ["Weather in Oslo", "Weather in Bergen"]


# Patterns from the hand check of 100 xLAM labels (md/2026-11-07-splitter-results.md).


def test_xlam_number_at_start_is_not_a_bare_value() -> None:
    ex = _xlam(
        "I need the agent profile for 'AGT987654' and also 10 WikiHow steps on how to tie a tie.",
        ("agent", {"agent_id": "AGT987654"}),
        ("steps", {"count": "10"}),
    )
    assert _xlam_texts(ex) == [
        "I need the agent profile for 'AGT987654'",
        "I need 10 WikiHow steps on how to tie a tie",
    ]


def test_xlam_quoted_bare_value_keeps_its_quote() -> None:
    ex = _xlam(
        "I need details about the book '1984' and 'To Kill a Mockingbird' from the Open Library.",
        ("book", {"title": "1984"}),
        ("book", {"title": "To Kill a Mockingbird"}),
    )
    assert _xlam_texts(ex) == [
        "I need details about the book '1984'",
        "I need details about the book 'To Kill a Mockingbird' from the Open Library",
    ]


def test_xlam_head_without_a_determiner_is_the_first_word() -> None:
    ex = _xlam(
        "Find all songs by The Weeknd and the playlists where he was first discovered.",
        ("songs", {"artist": "The Weeknd"}),
        ("playlists", {}),
    )
    assert _xlam_texts(ex) == [
        "Find all songs by The Weeknd",
        "Find the playlists where he was first discovered",
    ]


def test_xlam_unanchored_name_is_dropped() -> None:
    country = Tool("details", "Country details.", (Arg("code", "enum", options=("AU", "NZ")),))
    calls = (Call("details", {"code": "AU"}), Call("details", {"code": "NZ"}))
    ex = TaskExample(
        "Could you provide the details for Australia and New Zealand?", (country,), "call", calls
    )
    assert xlam_actions(ex) is None


def test_xlam_preposition_phrase_is_dropped() -> None:
    ex = _xlam(
        "What are the latest science news in French for France and in German for Germany?",
        ("news", {"country": "France"}),
        ("news", {"country": "Germany"}),
    )
    assert xlam_actions(ex) is None


def test_xlam_if_clause_is_dropped() -> None:
    ex = _xlam(
        "Check if user@example.com is a valid email address and if the domain is disposable.",
        ("check_email", {"email": "user@example.com"}),
        ("disposable", {}),
    )
    assert xlam_actions(ex) is None


def test_xlam_more_connectors() -> None:
    ex = _xlam(
        "Find the area of a triangle with a base of 15. Furthermore, count the set bits of 100.",
        ("area", {"base": "15"}),
        ("bits", {"n": "100"}),
    )
    assert _xlam_texts(ex) == [
        "Find the area of a triangle with a base of 15",
        "count the set bits of 100",
    ]
