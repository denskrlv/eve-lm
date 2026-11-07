import pytest
import torch

from eve.config import preset
from eve.model.checkpoint import Checkpoint
from eve.model.splitter import Splitter, SplitterConfig
from eve.split.decide import SplitCase, Thresholds, decide, pick_thresholds, split_request
from eve.split.words import word_spans
from eve.tokenizer.bpe import BPETokenizer

TEXT = "turn off the fan and lock the door"
WORDS = word_spans(TEXT)  # turn off the fan | and | lock the door
TH = Thresholds(exists=0.9, member=0.8)
FIRST, SECOND = [0, 1, 2, 3], [5, 6, 7]


def _members(*actions: list[int], p: float = 0.99) -> list[list[float]]:
    return [[p if i in action else 1 - p for i in range(len(WORDS))] for action in actions]


def _exists(n: int, k: int = 8) -> list[float]:
    return [0.99] * n + [0.01] * (k - n)


def test_decide_two_confident_actions() -> None:
    assert decide(_exists(2), _members(FIRST, SECOND), WORDS, TEXT, TH) == [FIRST, SECOND]


def test_decide_unsure_member_returns_none() -> None:
    members = _members(FIRST, SECOND)
    members[1][6] = 0.6  # "the" in action 2: not sure either way
    assert decide(_exists(2), members, WORDS, TEXT, TH) is None


def test_decide_one_action_returns_none() -> None:
    assert decide(_exists(1), _members(FIRST + SECOND), WORDS, TEXT, TH) is None


def test_decide_gap_in_slots_returns_none() -> None:
    exists = _exists(2)
    exists[3] = 0.99  # slot 3 exists, slot 2 doesn't
    assert decide(exists, _members(FIRST, SECOND), WORDS, TEXT, TH) is None


def test_decide_unsure_existence_returns_none() -> None:
    exists = _exists(2)
    exists[2] = 0.5
    assert decide(exists, _members(FIRST, SECOND), WORDS, TEXT, TH) is None


def test_punctuation_only_action_is_unsure() -> None:
    text = "turn off the fan , and"
    words = word_spans(text)
    members = [
        [0.99 if i < 4 else 0.01 for i in range(6)],
        [0.99 if i >= 4 else 0.01 for i in range(6)],
    ]
    assert decide(_exists(2), members, words, text, TH) is None


def _ckpt(tokenizer: BPETokenizer) -> Checkpoint[Splitter]:
    torch.manual_seed(0)
    return Checkpoint(Splitter(SplitterConfig(**preset("tiny").to_dict())), tokenizer, {})


def test_long_request_is_not_split(tokenizer: BPETokenizer) -> None:
    text = " ".join(["turn off the kitchen lights and lock the door"] * 25)
    assert split_request(_ckpt(tokenizer), text, Thresholds(0.5, 0.5)) == [text]


def test_split_request_returns_texts(tokenizer: BPETokenizer) -> None:
    out = split_request(_ckpt(tokenizer), TEXT, Thresholds(0.5, 0.5))
    assert out and all(isinstance(t, str) and t for t in out)


def test_thresholds_round_trip() -> None:
    assert Thresholds.from_json(TH.to_json()) == TH


def _case(gold: list[list[int]], exists: list[float], members: list[list[float]]) -> SplitCase:
    return SplitCase(exists, members, WORDS, TEXT, gold)


def test_pick_thresholds_respects_budget() -> None:
    whole = [list(range(len(WORDS)))]
    cases = [_case([FIRST, SECOND], _exists(2), _members(FIRST, SECOND)) for _ in range(50)]
    # 100 single-action requests; 5 look like two actions with P(exists) 0.7: split at a loose
    # threshold (5% wrong splits), not at a strict one.
    loose = [0.99, 0.7, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    cases += [_case(whole, loose, _members(FIRST, SECOND)) for _ in range(5)]
    cases += [_case(whole, _exists(1), _members(whole[0])) for _ in range(95)]
    th = pick_thresholds(cases, max_wrong_split=0.02)
    assert th.exists > 0.7
    wrong = sum(decide(c.exists_p, c.member_p, WORDS, TEXT, th) is not None for c in cases[50:])
    assert wrong / 100 <= 0.02
    right = sum(decide(c.exists_p, c.member_p, WORDS, TEXT, th) == c.gold for c in cases[:50])
    assert right == 50


@pytest.mark.parametrize("max_wrong", [0.0, 0.02])
def test_pick_thresholds_always_returns_a_pair(max_wrong: float) -> None:
    whole = [list(range(len(WORDS)))]
    cases = [_case(whole, [0.99, 0.999] + [0.01] * 6, _members(FIRST, SECOND))]
    assert isinstance(pick_thresholds(cases, max_wrong), Thresholds)
