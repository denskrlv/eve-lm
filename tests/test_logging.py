import logging

import pytest

from eve.cli import setup_logging


def test_status_lines_go_to_stdout_like_print(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging()
    logging.getLogger("eve.train.loop").info("resumed run at step 3")
    logging.getLogger("eve.data.corpus").warning("the corpus ran out")
    assert capsys.readouterr().out == "resumed run at step 3\nwarning: the corpus ran out\n"


def test_setting_up_twice_does_not_repeat_lines(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging()
    setup_logging()
    logging.getLogger("eve.train.loop").info("once")
    assert capsys.readouterr().out == "once\n"


def test_debug_lines_stay_hidden(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging()
    logging.getLogger("eve.train.loop").debug("noise")
    assert capsys.readouterr().out == ""
