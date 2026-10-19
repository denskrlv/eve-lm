import logging

from colibri.cli import setup_logging


def test_status_lines_go_to_stdout_like_print(capsys):
    setup_logging()
    logging.getLogger("colibri.train.loop").info("resumed run at step 3")
    logging.getLogger("colibri.data.corpus").warning("the corpus ran out")
    assert capsys.readouterr().out == "resumed run at step 3\nwarning: the corpus ran out\n"


def test_setting_up_twice_does_not_repeat_lines(capsys):
    setup_logging()
    setup_logging()
    logging.getLogger("colibri.train.loop").info("once")
    assert capsys.readouterr().out == "once\n"


def test_debug_lines_stay_hidden(capsys):
    setup_logging()
    logging.getLogger("colibri.train.loop").debug("noise")
    assert capsys.readouterr().out == ""
