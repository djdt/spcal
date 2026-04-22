from typing import Callable

from PySide6 import QtCore, QtGui


def create_action(
    icon: str,
    label: str,
    status: str,
    func: Callable | QtCore.SignalInstance | None = None,
    checkable: bool = False,
) -> QtGui.QAction:
    action = QtGui.QAction(QtGui.QIcon.fromTheme(icon), label)
    action.setStatusTip(status)
    action.setToolTip(status)
    if func is not None:
        action.triggered.connect(func)
    action.setCheckable(checkable)
    return action
