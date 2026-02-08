from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from spcal.gui.objects import (
    ContextMenuRedirectFilter,
    DoubleOrEmptyValidator,
    DoubleOrPercentValidator,
)


class TestWidget(QtWidgets.QWidget):
    testSignal = QtCore.Signal()

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        self.testSignal.emit()


def test_context_menu_redirect_filter(qtbot: QtBot):
    w = TestWidget()
    qtbot.addWidget(w)

    w2 = QtWidgets.QWidget()
    qtbot.addWidget(w2)

    filter = ContextMenuRedirectFilter(w)
    w2.installEventFilter(filter)

    with qtbot.waitSignal(w.testSignal, timeout=100):
        event = QtGui.QContextMenuEvent(
            QtGui.QContextMenuEvent.Reason.Mouse, QtCore.QPoint(0, 0)
        )
        QtWidgets.QApplication.sendEvent(w2, event)


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
