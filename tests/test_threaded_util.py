from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from spcal.gui.util import Worker


def test_worker_runnable(qtbot: QtBot):
    def func(a: float, b: float, c: float = 10.0):
        if c < 0.0:
            raise ValueError(c)
        return (a + b) * c

    threadpool = QtCore.QThreadPool()

    worker = Worker(func, 1, 2, c=3)
    with qtbot.waitSignals(
        [worker.signals.finished, worker.signals.result],
        check_params_cbs=[None, lambda x: x == 9.0],
        timeout=100,
    ):
        threadpool.start(worker)

    worker = Worker(func, 1, 2, c=-3)
    with qtbot.waitSignal(worker.signals.exception, timeout=100):
        threadpool.start(worker)
