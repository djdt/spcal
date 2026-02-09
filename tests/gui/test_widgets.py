from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from spcal.gui.widgets.collapsablewidget import CollapsableWidget
from spcal.gui.widgets.periodictable import PeriodicTableSelector
from spcal.gui.widgets.units import UnitsWidget
from spcal.gui.widgets.values import ValueWidget


def test_collapsable_widget(qtbot: QtBot):
    widget = CollapsableWidget("Test")

    qtbot.addWidget(widget)

    with qtbot.waitExposed(widget):
        widget.show()


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


def test_value_widget(qtbot: QtBot):
    w = ValueWidget()
    qtbot.addWidget(w)
    with qtbot.wait_exposed(w):
        w.show()

    assert w.value() is None
    assert w.error() is None
    assert w.text() == ""
    assert w.hasAcceptableInput()

    w.setValue(0.123456789)
    assert w.value() == 0.123456789

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
