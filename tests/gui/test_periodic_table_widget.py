import numpy as np
from PySide6 import QtCore, QtGui
from pytestqt.qtbot import QtBot

from spcal.gui.widgets import PeriodicTableSelector


def test_periodic_table_selector(qtbot: QtBot):
    pt = PeriodicTableSelector()
    qtbot.addWidget(pt)

    # length of all isotopes in database
    assert len(pt.enabledIsotopes()) == 344
    assert pt.selectedIsotopes() is None

    pt.buttons["C"].actions[13].setChecked(True)
    pt.buttons["Au"].actions[197].setChecked(True)

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
    assert selected[0]["Isotope"] == 12

    # Select 3 isotopes, remove other selected
    pt.setSelectedIsotopes(pt.enabledIsotopes()[[50, 60, 70]])
    selected = pt.selectedIsotopes()
    assert selected is not None
    assert len(selected) == 3
    assert np.all(selected["Symbol"] == ["Ti", "Cr", "Ni"])

    pt.setIsotopeColors(
        pt.enabledIsotopes()[[50, 60]],
        [QtGui.QColor.fromRgb(255, 0, 0), QtGui.QColor.fromRgb(0, 255, 0)],
    )
    assert pt.buttons["Ti"].palette().button().color().red() == 255
    assert pt.buttons["Cr"].palette().button().color().green() == 255

    # Remove color
    pt.setIsotopeColors(
        pt.enabledIsotopes()[[50]],
        [QtGui.QColor.fromRgb(255, 0, 0)],
    )
    assert pt.buttons["Cr"].palette().button().color().green() != 255
