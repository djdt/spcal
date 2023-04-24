from pathlib import Path

import numpy as np
import numpy.lib.recfunctions as rfn
import pytest
from pytestqt.qtbot import QtBot

from spcal.gui.dialogs.calculator import CalculatorDialog
from spcal.gui.main import SPCalWindow
from spcal.io.session import restoreSession, saveSession

npz = np.load(Path(__file__).parent.parent.joinpath("data/tofwerk_auag.npz"))
data = rfn.unstructured_to_structured(
    np.stack((npz["107Ag"], npz["197Au"]), axis=1),
    dtype=[("Au", np.float32), ("Ag", np.float32)],
)


@pytest.fixture(scope="session")
def tmp_session_path(tmp_path_factory):
    return tmp_path_factory.mktemp("session").joinpath("tmp.spcal")


def test_save_session(tmp_session_path: Path, qtbot: QtBot):
    window = SPCalWindow()
    qtbot.addWidget(window)

    window.options.efficiency_method.setCurrentText("Reference Particle")
    window.options.uptake.setBaseValue(0.2)
    window.options.error_rate_gaussian.setValue(0.003)
    window.options.error_rate_poisson.setValue(0.004)

    window.sample.loadData(data, {"path": "test/data.csv", "dwelltime": 0.1})
    window.sample.io["Ag"].response.setBaseValue(1.0)
    window.sample.io["Au"].response.setBaseValue(2.0)
    window.sample.io["Au"].density.setBaseValue(3.0)

    window.reference.loadData(data.copy(), {"path": "test/ref.csv", "dwelltime": 0.1})
    window.reference.io["Au"].density.setBaseValue(4.0)

    dlg = CalculatorDialog(window.sample, window.reference, parent=window)
    qtbot.add_widget(dlg)
    dlg.formula.setPlainText("Ag + Au")
    dlg.accept()

    saveSession(tmp_session_path, window.options, window.sample, window.reference)

    CalculatorDialog.current_expressions.clear()


def test_restore_session(tmp_session_path: Path, qtbot: QtBot):
    window = SPCalWindow()
    qtbot.addWidget(window)

    restoreSession(tmp_session_path, window.options, window.sample, window.reference)

    assert window.options.efficiency_method.currentText() == "Reference Particle"
    assert window.options.dwelltime.baseValue() == 0.1
    assert window.options.uptake.baseValue() == 0.2
    assert window.options.error_rate_gaussian.value() == 0.003
    assert window.options.error_rate_poisson.value() == 0.004

    assert window.sample.names == ("Au", "Ag", "{Ag+Au}")
    assert str(window.sample.import_options["path"]) == "test/data.csv"
    assert window.sample.io["Ag"].response.baseValue() == 1.0
    assert window.sample.io["Au"].response.baseValue() == 2.0
    assert window.sample.io["Au"].density.baseValue() == 3.0

    assert window.reference.names == ("Au", "Ag", "{Ag+Au}")
    assert str(window.reference.import_options["path"]) == "test/ref.csv"
    assert window.reference.io["Au"].density.baseValue() == 4.0

    assert CalculatorDialog.current_expressions["{Ag+Au}"] == "+ Ag Au"

    CalculatorDialog.current_expressions.clear()
