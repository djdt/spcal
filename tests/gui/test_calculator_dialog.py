from pathlib import Path

import numpy as np
import numpy.lib.recfunctions as rfn
from pytestqt.qtbot import QtBot

from spcal.gui.dialogs.calculator import CalculatorDialog
from spcal.gui.main import SPCalWindow


def test_calculator_dialog(qtbot: QtBot):
    window = SPCalWindow()
    qtbot.add_widget(window)

    npz = np.load(Path(__file__).parent.parent.joinpath("data/tofwerk_auag.npz"))
    data = rfn.unstructured_to_structured(
        np.stack((npz["107Ag"], npz["197Au"]), axis=1),
        dtype=[("Au", np.float32), ("Ag", np.float32)],
    )

    window.options.efficiency_method.setCurrentText("Reference Particle")

    window.sample.loadData(data, {"path": "test/data.csv", "dwelltime": 0.1})
    window.sample.io["Ag"].response.setBaseValue(100.0)
    window.sample.io["Au"].response.setBaseValue(200.0)

    window.reference.loadData(data.copy(), {"path": "test/ref.csv", "dwelltime": 0.1})
    window.reference.io["Ag"].response.setBaseValue(100.0)
    window.reference.io["Au"].response.setBaseValue(200.0)

    dlg = CalculatorDialog(window.sample, window.reference, parent=window)

    with qtbot.wait_exposed(dlg):
        dlg.show()

    dlg.formula.setPlainText("Ag + Au")
    dlg.accept()

    assert "{Ag+Au}" in window.sample.names
    assert "{Ag+Au}" in window.reference.names

    assert window.sample.io["{Ag+Au}"].response.baseValue() == 300.0
    assert window.reference.io["{Ag+Au}"].response.baseValue() == 300.0

    # Prevent altering further tests
    CalculatorDialog.current_expressions.clear()
