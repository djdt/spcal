from typing import Callable

from PySide6 import QtCore, QtGui


def create_action(
    icon: str, label: str, status: str, func: Callable, checkable: bool = False
) -> QtGui.QAction:
    action = QtGui.QAction(QtGui.QIcon.fromTheme(icon), label)
    action.setStatusTip(status)
    action.setToolTip(status)
    action.triggered.connect(func)
    action.setCheckable(checkable)
    return action


class WorkerSignals(QtCore.QObject):
    finished = QtCore.Signal()
    exception = QtCore.Signal(type, BaseException, object)
    result = QtCore.Signal(object)


class Worker(QtCore.QRunnable):
    def __init__(
        self,
        func: Callable,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            result = self.func(*self.args, **self.kwargs)
        except Exception as e:
            self.signals.exception.emit(type(e), e, e.__traceback__)
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()
