from pathlib import Path

import numpy as np
from pytestqt.qtbot import QtBot

from spcal.gui.dialogs._import import NuImportDialog
from spcal.gui.main import SPCalWindow


def test_sample_reload_options(qtbot: QtBot):
    window = SPCalWindow()
    qtbot.add_widget(window)
    with qtbot.wait_exposed(window):
        window.show()

    dlg = window.sample.dialogLoadFile(Path(__file__).parent.parent.joinpath("data/nu"))
    assert isinstance(dlg, NuImportDialog)
    dlg.table.buttons["Fe"].actions[54].setChecked(True)
    dlg.table.buttons["Fe"].actions[56].setChecked(True)
    dlg.table.buttons["Co"].actions[59].setChecked(True)
    dlg.table.buttons["Zn"].actions[64].setChecked(True)
    dlg.dwelltime.setBaseValue(1e-3)

    with qtbot.wait_signal(window.sample.dataLoaded):
        dlg.accept()

    # Set values
    window.sample.io["Fe54"].density.setBaseValue(1.0)
    window.sample.io["Fe56"].response.setBaseValue(2.0)
    window.sample.io["Co59"].density.setBaseValue(3.0)
    window.sample.io["Zn64"].response.setBaseValue(4.0)

    # Check import options are loaded correctly
    dlg = window.sample.dialogLoadFile(Path(__file__).parent.parent.joinpath("data/nu"))
    isotopes = dlg.table.selectedIsotopes()
    assert np.all(isotopes["Symbol"] == ["Fe", "Fe", "Co", "Zn"])
    assert np.all(isotopes["Isotope"] == [54, 56, 59, 64])
    # The dwelltime is not set by default
    assert dlg.dwelltime.baseValue() != 1e-3

    with qtbot.wait_signal(window.sample.dataLoaded):
        dlg.accept()

    # Check if overwriting on re-load
    assert window.sample.io["Fe54"].density.baseValue() == 1.0
    assert window.sample.io["Fe56"].response.baseValue() == 2.0
    assert window.sample.io["Co59"].density.baseValue() == 3.0
    assert window.sample.io["Zn64"].response.baseValue() == 4.0
