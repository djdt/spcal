from pathlib import Path

import numpy as np
from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from spcal.gui.main import SPCalWindow
from spcal.result import Filter

# Determined experimentally
uptake = 1.567e-6
dwelltime = 1e-4
response = 16.08e9
efficiency = 0.062
density = 19.32e3


def test_edit_and_enable_names(qtbot: QtBot):
    window = SPCalWindow()
    qtbot.add_widget(window)
    with qtbot.wait_exposed(window):
        window.show()

    npz = np.load(Path(__file__).parent.parent.joinpath("data/agilent_au_data.npz"))
    data = np.array(npz["au50nm"], dtype=[("Au", np.float32), ("Ag", np.float32)])
    ref = np.array(npz["au15nm"], dtype=[("Au", np.float32)])

    window.options.efficiency_method.setCurrentText("Reference Particle")
    window.options.uptake.setBaseValue(uptake)

    window.sample.loadData(data, {"path": "test/data.csv", "dwelltime": dwelltime})

    window.sample.io["Au"].density.setBaseValue(density)
    window.sample.io["Au"].response.setBaseValue(response)
    window.sample.io["Ag"].density.setBaseValue(density)
    window.sample.io["Ag"].response.setBaseValue(response)

    window.reference.loadData(ref, {"path": "test/ref.csv", "dwelltime": dwelltime})

    window.reference.io["Au"].density.setBaseValue(density)
    window.reference.io["Au"].diameter.setBaseValue(15e-9)

    window.tabs.setCurrentWidget(window.results)

    assert window.sample.enabled_names == ["Au", "Ag"]
    assert window.reference.enabled_names == ["Au"]
    assert list(window.results.results.keys()) == ["Au", "Ag"]

    window.results.filters = [[Filter("Au", "signal", ">", 0.1)]]

    window.sample.io.combo_name.setItemText(0, "Au2")
    window.sample.io.combo_name.lineEdit().editingFinished.emit()

    assert window.sample.enabled_names == ["Au2", "Ag"]
    assert window.reference.enabled_names == ["Au2"]
    assert list(window.results.results.keys()) == ["Ag", "Au2"]
    assert window.results.filters[0][0].name == "Au2"

    dlg = window.sample.io.combo_name.openEnableDialog()
    dlg.texts.item(0).setCheckState(QtCore.Qt.CheckState.Unchecked)
    dlg.accept()

    assert window.sample.enabled_names == ["Ag"]
    assert window.reference.enabled_names == ["Au2"]
    window.results.updateResults()
    assert list(window.results.results.keys()) == ["Ag"]
