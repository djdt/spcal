from pathlib import Path

import numpy as np
from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from spcal.gui.dialogs.response import ResponseDialog


def generate_data(mean: float) -> np.ndarray:
    data = np.empty(
        1000,
        dtype=[
            ("A", np.float32),
            ("B", np.float32),
            ("C", np.float32),
            ("D", np.float32),
        ],
    )
    assert data.dtype.names is not None
    for name in data.dtype.names:
        data[name] = np.random.normal(size=1000, loc=mean, scale=0.1)
    return data


def test_response_dialog(qtbot: QtBot):
    dlg = ResponseDialog()
    qtbot.add_widget(dlg)
    with qtbot.wait_exposed(dlg):
        dlg.show()

    dlg.loadData(generate_data(10.0), {"path": Path("0.csv")})
    assert dlg.model.columnCount() == 1
    assert dlg.model.rowCount() == 4
    dlg.model.setData(dlg.model.index(0, 0), 0.0, QtCore.Qt.ItemDataRole.EditRole)
    dlg.model.setData(dlg.model.index(1, 0), 0.0, QtCore.Qt.ItemDataRole.EditRole)
    dlg.model.setData(dlg.model.index(2, 0), 10.0, QtCore.Qt.ItemDataRole.EditRole)

    dlg.loadData(generate_data(20.0), {"path": Path("1.csv")})
    dlg.model.setData(dlg.model.index(0, 1), 10.0, QtCore.Qt.ItemDataRole.EditRole)
    dlg.model.setData(dlg.model.index(1, 1), 20.0, QtCore.Qt.ItemDataRole.EditRole)
    assert dlg.model.columnCount() == 2
    assert dlg.model.rowCount() == 4

    dlg.combo_unit.setCurrentText("mg/L")

    def check_response(responses: dict):
        if not np.isclose(responses["A"], 1e6, rtol=0.01):
            return False
        if not np.isclose(responses["B"], 0.5e6, rtol=0.01):
            return False
        if not np.isclose(responses["C"], 1e6, rtol=0.01):  # single point
            return False
        return True

    with qtbot.wait_signal(
        dlg.responsesSelected, timeout=100, check_params_cb=check_response
    ):
        dlg.accept()

    # Todo, test for export
