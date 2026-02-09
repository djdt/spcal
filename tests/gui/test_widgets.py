import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from spcal.gui.widgets.collapsablewidget import CollapsableWidget
from spcal.gui.widgets.elidedlabel import ElidedLabel
from spcal.gui.widgets.periodictable import PeriodicTableSelector
from spcal.gui.widgets.units import UnitsWidget
from spcal.gui.widgets.values import ValueWidget
from spcal.isotope import ISOTOPE_TABLE


def test_collapsable_widget(qtbot: QtBot):
    widget = CollapsableWidget("Test")
    label = QtWidgets.QLabel("Label")
    widget.setWidget(label)

    qtbot.addWidget(widget)

    with qtbot.waitExposed(widget):
        widget.show()

    assert not label.isVisible()
    qtbot.mouseClick(widget.button, QtCore.Qt.MouseButton.LeftButton)
    assert label.isVisible()


def test_edlided_label(qtbot: QtBot):
    label = ElidedLabel("a very long line of text that should be elided")
    label.setFixedWidth(50)

    qtbot.addWidget(label)

    with qtbot.waitExposed(label):
        label.show()

    label.setText("short text")


def test_periodic_table_selector(qtbot: QtBot):
    pt = PeriodicTableSelector()
    qtbot.addWidget(pt)

    # length of all isotopes in database
    assert len(pt.enabledIsotopes()) == 344
    assert pt.selectedIsotopes() == []

    pt.buttons["C"].isotope_actions[13].setChecked(True)
    pt.buttons["Au"].isotope_actions[197].setChecked(True)

    selected = pt.selectedIsotopes()
    assert selected is not None
    assert len(selected) == 2

    # unselect C
    qtbot.mouseClick(pt.buttons["C"], QtCore.Qt.MouseButton.LeftButton)
    selected = pt.selectedIsotopes()
    assert selected is not None
    assert len(selected) == 1

    # reselect C, default isotope
    qtbot.mouseClick(pt.buttons["C"], QtCore.Qt.MouseButton.LeftButton)
    selected = pt.selectedIsotopes()
    assert selected is not None
    assert len(selected) == 2
    assert selected[0].isotope == 12

    # Select 3 isotopes, remove other selected
    enabled = pt.enabledIsotopes()
    pt.setSelectedIsotopes([enabled[x] for x in [50, 60, 70]])
    selected = pt.selectedIsotopes()
    assert selected is not None
    assert len(selected) == 3
    assert all(s.symbol == x for s, x in zip(selected, ["Ti", "Cr", "Ni"]))

    pt.setIsotopeColors(
        [enabled[50], enabled[60]],
        [QtGui.QColor.fromRgb(255, 0, 0), QtGui.QColor.fromRgb(0, 255, 0)],
    )
    assert pt.buttons["Ti"].indicator is not None
    assert pt.buttons["Ti"].indicator.red() == 255
    assert pt.buttons["Cr"].indicator is not None
    assert pt.buttons["Cr"].indicator.green() == 255

    with qtbot.waitExposed(pt):
        pt.show()

    # Remove color
    pt.setIsotopeColors([enabled[50]], [QtGui.QColor.fromRgb(255, 0, 0)])
    assert pt.buttons["Cr"].indicator is None

    # Select all Tin
    pt.buttons["Sn"].selectAllIsotopes(0.1)
    assert len(pt.buttons["Sn"].selectedIsotopes()) == 3

    # Only enable 3, 2 selected
    pt.setEnabledIsotopes(
        [
            ISOTOPE_TABLE[("Sn", 116)],
            ISOTOPE_TABLE[("Sn", 117)],
            ISOTOPE_TABLE[("Sn", 118)],
        ]
    )
    assert len(pt.enabledIsotopes()) == 3
    assert len(pt.selectedIsotopes()) == 2


def test_units_widget(qtbot: QtBot):
    w = UnitsWidget({"a": 1.0, "b": 0.1, "c": 0.01}, default_unit="c")
    qtbot.addWidget(w)

    assert w.baseValue() is None
    assert w.value() is None
    assert w.baseError() is None
    assert w.error() is None
    assert w.unit() == "c"
    assert w.hasAcceptableInput()

    with qtbot.waitSignal(w.baseValueChanged, timeout=100):
        w.setValue(1.0)
    assert w.baseValue() == 0.01

    with qtbot.assertNotEmitted(w.baseValueChanged):
        w.setBaseValue(0.01)

    w.setUnit("b")
    assert w.baseValue() == 0.01
    assert np.isclose(w.value(), 0.1)  # type: ignore

    w.setBaseValue(10.0)
    assert w.value() == 100.0

    w.setBestUnit()
    assert w.unit() == "c"

    w.setReadOnly(True)
    assert w._value.isReadOnly()

    assert w.isEnabled()
    w.setEnabled(False)
    assert not w._value.isEnabled()
    assert not w.combo.isEnabled()
    assert not w.isEnabled()

    w.setToolTip("tip")
    assert w._value.toolTip() == "tip"
    assert w.combo.toolTip() == "tip"

    # Make sure this does not cause a RecursionError
    with qtbot.waitSignal(w.baseValueChanged, timeout=100):
        w.setValue(np.nan)
    with qtbot.assertNotEmitted(w.baseValueChanged):
        w.setValue(None)


def test_units_widget_error(qtbot: QtBot):
    w = UnitsWidget({"a": 1.0, "b": 0.1, "c": 0.01}, base_value=50.0)
    qtbot.addWidget(w)

    assert w.baseError() is None
    with qtbot.waitSignal(w.baseErrorChanged, timeout=100):
        w.setBaseError(10.0)
    assert w.error() == 10.0

    with qtbot.assertNotEmitted(w.baseErrorChanged):
        w.setError(10.0)

    w.setUnit("b")
    assert w.error() == 100.0

    w.setBaseError(1.0)
    assert w.error() == 10.0

    w.setError(1.0)
    assert w.baseError() == 0.1

    # Make sure this does not cause a RecursionError
    with qtbot.waitSignal(w.baseErrorChanged, timeout=100):
        w.setError(np.nan)
    with qtbot.assertNotEmitted(w.baseErrorChanged):
        w.setError(None)


def test_value_widget(qtbot: QtBot):
    w = ValueWidget()
    qtbot.addWidget(w)
    with qtbot.wait_exposed(w):
        w.show()

    assert w.value() is None
    assert w.error() is None
    assert w.text() == ""
    assert w.hasAcceptableInput()

    with qtbot.waitSignal(w.valueChanged, timeout=100):
        w.setValue(0.123456789)
    assert w.value() == 0.123456789

    with qtbot.assertNotEmitted(w.valueChanged):
        w.setValue(0.123456789)

    w.clearFocus()
    assert w.text() == "0.123457"
    w.setFocus()
    qtbot.wait(100)  # test sometimes fails without wait
    assert w.text() == "0.123456789"
    w.clearFocus()

    # Value should not change
    assert w.value() == 0.123456789

    w.setFocus()
    qtbot.keyClicks(w, "1.23212321")
    assert w.text() == "1.23212321"
    assert w.value() == 1.23212321
    w.clearFocus()
    assert w.text() == "1.23212"

    w.setSigFigs(8)
    assert w.text() == "1.2321232"

    with qtbot.waitSignal(w.errorChanged, timeout=100):
        w.setError(0.123456789)

    with qtbot.assertNotEmitted(w.errorChanged):
        w.setError(0.123456789)

    # Don't know how to test the paint event
    assert w.error() == 0.123456789
    w.repaint()

    # Range cuts value
    w.setValue(10.1)
    w.setRange(4.0, 6.0)
    assert w.value() == 6.0
    w.stepDown()
    assert w.value() == 5.0
    w.stepDown()
    assert w.value() == 4.0
    w.stepDown()  # reached min
    assert w.value() == 4.0

    with qtbot.waitSignal(w.valueChanged):
        w.setValue(None)
    with qtbot.waitSignal(w.errorChanged):
        w.setError(None)

    with qtbot.assertNotEmitted(w.valueChanged):
        w.setValue(None)
    with qtbot.assertNotEmitted(w.errorChanged):
        w.setError(None)


def test_value_widget_step_function(qtbot: QtBot):
    w = ValueWidget(value=100.0, step=lambda v, i: v + (v * i * 0.1))
    qtbot.addWidget(w)
    with qtbot.wait_exposed(w):
        w.show()

    assert w.value() == 100.0
    w.stepDown()
    assert w.value() == 90.0
    w.stepDown()
    assert w.value() == 81.0
    w.stepBy(2)
    assert w.value() == 97.2
