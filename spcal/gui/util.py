from typing import Callable

from PySide6 import QtGui


def create_action(
    icon: str, label: str, status: str, func: Callable, checkable: bool = False
) -> QtGui.QAction:
    action = QtGui.QAction(QtGui.QIcon.fromTheme(icon), label)
    action.setStatusTip(status)
    action.setToolTip(status)
    action.triggered.connect(func)
    action.setCheckable(checkable)
    return action
