from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from spcal.gui.objects import (
    ContextMenuRedirectFilter,
    DragDropRedirectFilter,
    KeepMenuOpenFilter,
    DoubleOrEmptyValidator,
    DoubleOrPercentValidator,
)


class RedirectTestWidget(QtWidgets.QWidget):
    contextMenuSignal = QtCore.Signal()
    dragEnterSignal = QtCore.Signal()
    dragLeaveSignal = QtCore.Signal()
    dragMoveSignal = QtCore.Signal()
    dropSignal = QtCore.Signal()

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        self.contextMenuSignal.emit()
        event.accept()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        self.dragEnterSignal.emit()
        event.accept()

    def dragLeaveEvent(self, event: QtGui.QDragLeaveEvent):
        self.dragLeaveSignal.emit()
        event.accept()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        self.dragMoveSignal.emit()
        event.accept()

    def dropEvent(self, event: QtGui.QDropEvent):
        self.dropSignal.emit()
        event.accept()


def test_context_menu_redirect_filter(qtbot: QtBot):
    to = RedirectTestWidget()
    qtbot.addWidget(to)

    frm = QtWidgets.QWidget()
    qtbot.addWidget(frm)

    filter = ContextMenuRedirectFilter(to)
    frm.installEventFilter(filter)

    with qtbot.waitSignal(to.contextMenuSignal, timeout=100):
        event = QtGui.QContextMenuEvent(
            QtGui.QContextMenuEvent.Reason.Mouse,
            QtCore.QPoint(0, 0),
            QtCore.QPoint(0, 0),
        )
        QtWidgets.QApplication.sendEvent(frm, event)


def test_drag_drop_redirect_filter(qtbot: QtBot):
    to = RedirectTestWidget()
    to.setAcceptDrops(True)
    qtbot.addWidget(to)

    frm = QtWidgets.QWidget()
    frm.setAcceptDrops(True)
    qtbot.addWidget(frm)

    filter = DragDropRedirectFilter(to)
    frm.installEventFilter(filter)

    mime = QtCore.QMimeData()
    mime.setText("test")

    with qtbot.waitSignal(to.dragEnterSignal, timeout=100):
        event = QtGui.QDragEnterEvent(
            QtCore.QPoint(0, 0),
            QtCore.Qt.DropAction.CopyAction,
            mime,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        QtWidgets.QApplication.sendEvent(frm, event)

    with qtbot.waitSignal(to.dragMoveSignal, timeout=100):
        event = QtGui.QDragMoveEvent(
            QtCore.QPoint(1, 1),
            QtCore.Qt.DropAction.CopyAction,
            mime,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        QtWidgets.QApplication.sendEvent(frm, event)

    with qtbot.waitSignal(to.dropSignal, timeout=100):
        event = QtGui.QDropEvent(
            QtCore.QPoint(0, 0),
            QtCore.Qt.DropAction.CopyAction,
            mime,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
            QtCore.QEvent.Type.Drop,
        )
        QtWidgets.QApplication.sendEvent(frm, event)

    with qtbot.waitSignal(to.dragLeaveSignal, timeout=100):
        event = QtGui.QDragLeaveEvent()
        QtWidgets.QApplication.sendEvent(frm, event)


# Tested manually, for some reason menu doesn't like to show
# def test_keep_menu_open_filter(qtbot: QtBot):
#     button = QtWidgets.QPushButton()
#     button.setText("Test")
#
#     menu = QtWidgets.QMenu("Test Menu", parent=button)
#     actions = [QtGui.QAction(x, parent=button) for x in "ABC"]
#     for action in actions:
#         action.setCheckable(True)
#     menu.addActions(actions)
#     button.setMenu(menu)
#
#     qtbot.addWidget(button)
#
#     menu.installEventFilter(KeepMenuOpenFilter(menu))
#
#     with qtbot.waitExposed(button):
#         button.show()
#


def test_double_or_empty_validator(qtbot: QtBot):
    le = QtWidgets.QLineEdit("")
    le.setValidator(DoubleOrEmptyValidator(-1.0, 1.0, 4))
    qtbot.addWidget(le)

    # empty
    assert le.hasAcceptableInput()
    le.setText("2.0")
    assert not le.hasAcceptableInput()
    le.setText("0.00001")
    assert not le.hasAcceptableInput()


def test_double_or_percent_validator(qtbot: QtBot):
    le = QtWidgets.QLineEdit("")
    le.setValidator(
        DoubleOrPercentValidator(
            -1.0, 1.0, decimals=2, percent_bottom=0.0, percent_top=10.0
        )
    )
    qtbot.addWidget(le)

    # empty
    assert not le.hasAcceptableInput()
    le.setText("1.0")
    assert le.hasAcceptableInput()
    le.setText("2.0")
    assert not le.hasAcceptableInput()
    le.setText("0.5000")
    assert not le.hasAcceptableInput()
    le.setText("2%")
    assert le.hasAcceptableInput()
    le.setText("2.0%")
    assert le.hasAcceptableInput()
    le.setText("2.0 %")
    assert not le.hasAcceptableInput()
    le.setText("2.0%%")
    assert not le.hasAcceptableInput()
    le.setText("20%")
    assert not le.hasAcceptableInput()
