import numpy as np
import pytest
from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from spcal.gui.widgets.units import UnitsWidget


def test_units_widget(qtbot: QtBot):
    w = UnitsWidget({"a": 1.0, "b": 0.1, "c": 0.01}, default_unit="c")
    qtbot.addWidget(w)

    assert w.baseValue() is None
    assert w.value() is None
    assert w.baseError() is None
    assert w.error() is None
    assert w.unit() == "c"
    assert not w.hasAcceptableInput()

    w.setValue(1.0)
    assert w.baseValue() == 0.01

    w.setUnit("b")
    assert w.baseValue() == 0.01
    assert w.value() == pytest.approx(0.1)

    w.setBaseValue(10.0)
    assert w.value() == 100.0

    w.setBestUnit()
    assert w.unit() == "c"

    w.setReadOnly(True)
    assert w.lineedit.isReadOnly()

    assert w.isEnabled()
    w.setEnabled(False)
    assert not w.lineedit.isEnabled()
    assert not w.combo.isEnabled()
    assert not w.isEnabled()

    w.setToolTip("tip")
    assert w.lineedit.toolTip() == "tip"
    assert w.combo.toolTip() == "tip"

    # Make sure this does not cause a RecursionError
    w.setValue(np.nan)


def test_units_widget_error(qtbot: QtBot):
    w = UnitsWidget({"a": 1.0, "b": 0.1, "c": 0.01}, base_value=50.0)
    qtbot.addWidget(w)

    assert w.baseError() is None
    w.setBaseError(10.0)
    assert w.error() == 10.0

    w.setUnit("b")
    assert w.error() == 100.0
    w.setBaseError(1.0)
    assert w.error() == 10.0

    w.setError(1.0)
    assert w.baseError() == 0.1

    # Make sure this does not cause a RecursionError
    w.setError(np.nan)


def test_units_signals(qtbot: QtBot):
    w = UnitsWidget({"a": 1.0, "b": 0.1})
    qtbot.addWidget(w)

    # Enter 1.0
    qtbot.keyClick(w.lineedit, QtCore.Qt.Key_1)
    qtbot.keyClick(w.lineedit, QtCore.Qt.Key_Period)
    qtbot.keyClick(w.lineedit, QtCore.Qt.Key_0)
    qtbot.keyClick(w.lineedit, QtCore.Qt.Key_Return)

    assert w.baseValue() == 1.0

    qtbot.keyClick(w.lineedit, QtCore.Qt.Key_2)
    w.clearFocus()
    assert w.baseValue() == 1.02


def test_units_sync(qtbot: QtBot):
    w = UnitsWidget({"a": 1.0, "b": 0.1}, base_value=1.0)
    x = UnitsWidget({"a": 1.0, "b": 0.1}, base_value=1.0)
    qtbot.addWidget(w)
    qtbot.addWidget(x)

    w.sync(x)

    x.setBaseValue(100.0)
    assert w.baseValue() == 100.0

    w.setUnit("b")
    assert x.unit() == "b"

    x.setError(2.0)
    assert w.error() == 2.0


def test_units_view_format(qtbot: QtBot):
    w = UnitsWidget({"a": 1.0, "b": 1.0})
    w.setValue(1.01)
    w.setViewFormat(1, "f")
    assert w.lineedit.text() == "1.0"
    w.setViewFormat(2, "f")
    assert w.lineedit.text() == "1.01"
    w.setEditFormat(1, "f")  # not tested
