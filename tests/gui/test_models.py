import numpy as np
from PySide6 import QtCore
from pytestqt.modeltest import ModelTester

from spcal.gui.models import NumpyRecArrayTableModel


def test_numpy_recarray_table_model(qtmodeltester: ModelTester):
    array = np.empty(10, dtype=[("str", "U16"), ("int", int), ("float", float)])
    array["str"] = "A"
    array["int"] = np.arange(10)
    array["float"] = np.random.random(10)

    model = NumpyRecArrayTableModel(
        array,
        fill_values={"U": "0", "i": 0},
        column_formats={"int": "{:.1f}"},
        column_flags={"str": ~QtCore.Qt.ItemFlag.ItemIsEditable},
    )

    assert model.columnCount() == 3
    assert model.rowCount() == 10
    # Header
    assert model.headerData(0, QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole) == "str"
    assert model.headerData(1, QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole) == "int"
    assert model.headerData(2, QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole) == "float"

    for i in range(10):
        assert model.headerData(i, QtCore.Qt.Vertical, QtCore.Qt.DisplayRole) == str(i)

    # Col flags
    assert model.flags(model.index(1, 1)) & QtCore.Qt.ItemFlag.ItemIsEditable
    assert not model.flags(model.index(1, 0)) & QtCore.Qt.ItemFlag.ItemIsEditable

    # Insert
    model.insertRows(1, 1)
    assert model.array.shape == (11,)
    assert model.array[1]["str"] == "0"
    assert model.array[1]["int"] == 0
    assert np.isnan(model.array[1]["float"])

    # Remove
    # model.removeRow(7)
    # assert model.array.shape == (10,)

    # Data
    assert model.data(model.index(1, 1)) == "0.0"
    assert model.data(model.index(1, 2)) == ""  # nan

    model.setData(model.index(1, 1), 10, QtCore.Qt.EditRole)
    assert model.array["int"][1] == 10

    qtmodeltester.check(model, force_py=True)
