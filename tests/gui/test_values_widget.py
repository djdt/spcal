from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from spcal.gui.widgets import ValueWidget


def test_value_widget(qtbot: QtBot):
    w = ValueWidget()
    qtbot.addWidget(w)
    with qtbot.wait_exposed(w):
        w.show()

    assert w.value() is None
    assert w.error() is None
    assert w.text() == ""
    assert not w.hasAcceptableInput()

    w.setValue(0.123456789)
    assert w.value() == 0.123456789

    w.clearFocus()
    assert w.text() == "0.123457"
    w.setFocus()
    qtbot.wait(1)  # test sometimes fails without wait
    assert w.text() == "0.123456789"
    w.clearFocus()

    # Value should not change
    assert w.value() == 0.123456789

    w.setFocus()
    qtbot.keyClick(w, QtCore.Qt.Key.Key_1)
    assert w.text() == "0.1234567891"
    assert w.value() == 0.1234567891
    w.clearFocus()
    assert w.text() == "0.123457"

    w.setViewFormat(8)
    assert w.text() == "0.12345679"

    w.setError(0.123456789)
    # Don't know how to test the paint event
    assert w.error() == 0.123456789
    w.repaint()

    w.setViewFormat(".0f")
    assert w.text() == "0"

    w.setEditFormat(1)
    w.setFocus()
    assert w.text() == "0.1"
    w.clearFocus()
    w.setEditFormat(".0f")
    w.setFocus()
    assert w.text() == "0"
