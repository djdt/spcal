from typing import Callable
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot
from pytestqt.modeltest import ModelTester

from spcal.gui.modelviews import IsotopeRole
from spcal.gui.modelviews.basic import BasicTableView
from spcal.gui.modelviews.datafile import DataFileDelegate, DataFileModel
from spcal.gui.modelviews.headers import CheckableHeaderView, ComboHeaderView
from spcal.gui.modelviews.isotope import (
    IsotopeComboBox,
    IsotopeNameDelegate,
    IsotopeNameValidator,
)
from spcal.gui.modelviews.models import NumpyRecArrayTableModel, SearchColumnsProxyModel
from spcal.gui.modelviews.response import ConcentrationModel, IntensityModel
from spcal.gui.modelviews.units import UnitsHeaderView, UnitsModel
from spcal.gui.modelviews.values import ValueWidgetDelegate


from spcal.gui.widgets.values import ValueWidget
from spcal.isotope import ISOTOPE_TABLE


def test_basic_table(qtbot: QtBot):
    table = BasicTableView()
    qtbot.addWidget(table)

    model = QtGui.QStandardItemModel()
    table.setModel(model)

    qtbot.addWidget(table)
    with qtbot.waitExposed(table):
        table.show()

    model.setRowCount(3)
    model.setColumnCount(2)

    model.setItem(0, 0, QtGui.QStandardItem("a"))
    model.setItem(0, 1, QtGui.QStandardItem("b"))
    model.setItem(1, 0, QtGui.QStandardItem("c"))
    model.setItem(1, 1, QtGui.QStandardItem("d"))
    model.setItem(2, 0, QtGui.QStandardItem("e"))
    model.setItem(2, 1, QtGui.QStandardItem("f"))

    table.setCurrentIndex(model.index(0, 0))
    assert table.currentIndex().row() == 0
    table._advance()
    assert table.currentIndex().row() == 1

    table.clearSelection()
    table.selectRow(0)
    table._copy()
    mime_data = QtWidgets.QApplication.clipboard().mimeData()
    assert mime_data.text() == "a\tb"

    # Paste is repeated
    table.selectAll()
    table._paste()
    assert model.item(0, 0).text() == "a"
    assert model.item(0, 1).text() == "b"
    assert model.item(1, 0).text() == "a"
    assert model.item(1, 1).text() == "b"
    assert model.item(2, 0).text() == "a"
    assert model.item(2, 1).text() == "b"

    table._copy()
    mime_data = QtWidgets.QApplication.clipboard().mimeData()
    assert mime_data.text() == "a\tb\na\tb\na\tb"

    table._delete()
    assert model.item(0, 0).text() == ""
    assert model.item(0, 1).text() == ""
    assert model.item(1, 0).text() == ""
    assert model.item(1, 1).text() == ""

    model.item(0, 0).setText("a")
    table._filldown()
    assert model.item(1, 0).text() == "a"
    table.clearSelection()
    table.selectRow(0)
    model.item(1, 0).setText("b")
    table._filldown()
    assert model.item(1, 0).text() == "a"
    assert model.item(2, 0).text() == "a"

    # table._cut()  # Same as _copy, _delete

    QtWidgets.QApplication.clipboard().setText("1\t2\n3\t4")
    table._paste()
    assert model.item(0, 0).text() == "1"
    assert model.item(0, 1).text() == "2"
    assert model.item(1, 0).text() == "3"
    assert model.item(1, 1).text() == "4"

    table.contextMenuEvent(
        QtGui.QContextMenuEvent(
            QtGui.QContextMenuEvent.Reason.Mouse,
            QtCore.QPoint(0, 0),
            QtCore.QPoint(0, 0),
        )
    )


def test_checkable_header_view(qtbot: QtBot):
    header = CheckableHeaderView(QtCore.Qt.Orientation.Horizontal)

    table = QtWidgets.QTableView()
    table.setHorizontalHeader(header)

    model = QtGui.QStandardItemModel()
    table.setModel(model)

    model.setRowCount(1)
    model.setColumnCount(3)
    model.setHorizontalHeaderLabels(["A", "B", "C"])

    qtbot.addWidget(table)
    with qtbot.waitExposed(table):
        table.show()

    assert header.checkState(0) == QtCore.Qt.CheckState.Unchecked
    assert header.checkState(1) == QtCore.Qt.CheckState.Unchecked
    assert header.checkState(2) == QtCore.Qt.CheckState.Unchecked

    with qtbot.waitSignal(header.checkStateChanged, timeout=100):
        header.setCheckState(0, QtCore.Qt.CheckState.Checked)
    assert header.checkState(0) == QtCore.Qt.CheckState.Checked

    with qtbot.waitSignal(header.checkStateChanged, timeout=100):
        qtbot.mouseClick(
            header.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            pos=QtCore.QPoint(header.sectionViewportPosition(1), header.height() // 2),
        )
    assert header.checkState(1) == QtCore.Qt.CheckState.Checked


def test_combo_header_view(qtbot: QtBot):
    header = ComboHeaderView(
        {1: ["B1", "B2", "B3"], 2: ["C1", "C2"]}, QtCore.Qt.Orientation.Horizontal
    )

    table = QtWidgets.QTableView()
    table.setHorizontalHeader(header)

    model = QtGui.QStandardItemModel()
    table.setModel(model)

    model.setRowCount(1)
    model.setColumnCount(3)
    model.setHorizontalHeaderLabels(["A", "B1", "C2"])

    qtbot.addWidget(table)
    with qtbot.waitExposed(table):
        table.show()

    with qtbot.waitSignal(header.sectionChanged, timeout=100):
        qtbot.mouseClick(
            header.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            pos=QtCore.QPoint(header.sectionViewportPosition(1), header.height() // 2),
            delay=10,
        )
        combo = header.findChild(QtWidgets.QComboBox)
        assert isinstance(combo, QtWidgets.QComboBox)
        assert combo.itemText(0) == "B1"
        assert combo.itemText(1) == "B2"
        assert combo.itemText(2) == "B3"
        combo.setCurrentIndex(1)

    assert model.headerData(1, QtCore.Qt.Orientation.Horizontal) == "B2"


def test_datafile_model(qtmodeltester: ModelTester, random_datafile_gen: Callable):
    model = DataFileModel([random_datafile_gen() for _ in range(5)])
    qtmodeltester.check(model, force_py=True)


def test_datafile_delegate(qtbot: QtBot, random_datafile_gen):
    view = QtWidgets.QListView()
    model = DataFileModel([random_datafile_gen() for _ in range(5)])
    qtbot.addWidget(view)

    view.setModel(model)
    view.setItemDelegate(DataFileDelegate())

    with qtbot.waitExposed(view):
        view.show()

    delegate = view.itemDelegateForIndex(model.index(0, 0))
    assert isinstance(delegate, DataFileDelegate)
    # Click close button
    with qtbot.waitSignal(model.rowsRemoved, timeout=100):
        pos = view.rectForIndex(model.index(0, 0)).topRight()
        pos -= QtCore.QPoint(
            DataFileDelegate.margin + 1, -(DataFileDelegate.margin + 1)
        )
        qtbot.mouseClick(view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=pos)
    # Click edit button
    with qtbot.waitSignal(model.editIsotopesRequested, timeout=100):
        pos = view.rectForIndex(model.index(0, 0)).bottomRight()
        pos -= QtCore.QPoint(DataFileDelegate.margin + 1, DataFileDelegate.margin + 1)
        qtbot.mouseClick(view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=pos)


def test_isotope_combo_box(qtbot: QtBot):
    combo = IsotopeComboBox()
    qtbot.addWidget(combo)

    with qtbot.waitExposed(combo):
        combo.show()

    combo.addIsotopes([ISOTOPE_TABLE[("Ag", 107)], ISOTOPE_TABLE[("Ag", 109)]])

    assert str(combo.currentIsotope()) == "107Ag"
    with qtbot.waitSignal(combo.isotopeChanged, timeout=100):
        combo.setCurrentIndex(1)
    assert str(combo.currentIsotope()) == "109Ag"
    with qtbot.waitSignal(combo.isotopeChanged, timeout=100):
        combo.setCurrentIsotope(ISOTOPE_TABLE[("Ag", 107)])
    assert combo.currentIndex() == 0


def test_isotope_name_validator():
    val = IsotopeNameValidator()

    assert val.validate("", 0)[0] == QtGui.QValidator.State.Intermediate
    assert val.validate("197Au", 0)[0] == QtGui.QValidator.State.Acceptable
    assert val.validate("197 Au", 0)[0] == QtGui.QValidator.State.Intermediate
    assert val.validate("Au", 0)[0] == QtGui.QValidator.State.Intermediate
    assert val.validate("197", 0)[0] == QtGui.QValidator.State.Intermediate
    assert val.validate("Au197", 0)[0] == QtGui.QValidator.State.Acceptable

    assert val.fixup("Au197") == "197Au"
    assert val.fixup("Au197->197") == "197Au"
    assert val.fixup("Au") == "197Au"


def test_isotope_name_delegate(qtbot: QtBot):
    table = QtWidgets.QTableWidget()
    qtbot.addWidget(table)

    table.setRowCount(1)
    table.setColumnCount(2)
    table.setItem(0, 0, QtWidgets.QTableWidgetItem("107Ag"))
    table.setItem(0, 1, QtWidgets.QTableWidgetItem("109Ag"))
    table.setItemDelegate(IsotopeNameDelegate())

    with qtbot.waitExposed(table):
        table.show()

    delegate = table.itemDelegateForIndex(table.model().index(0, 0))
    assert isinstance(delegate, IsotopeNameDelegate)
    editor = delegate.createEditor(
        table, QtWidgets.QStyleOptionViewItem(), table.model().index(0, 0)
    )
    assert isinstance(editor, QtWidgets.QLineEdit)
    assert editor.text() == "107Ag"
    editor.setText("Au")
    assert not editor.hasAcceptableInput()
    editor.setText("197Au")
    assert editor.hasAcceptableInput()

    with qtbot.waitSignal(table.model().dataChanged, timeout=100):
        delegate.setModelData(editor, table.model(), table.model().index(0, 0))

    assert table.model().index(0, 0).data() == "197Au"
    assert table.model().index(0, 0).data(IsotopeRole) == ISOTOPE_TABLE[("Au", 197)]


def test_numpy_recarray_table_model(qtmodeltester: ModelTester):
    array = np.empty(10, dtype=[("str", "U16"), ("int", int), ("float", float)])
    array["str"] = "A"
    array["int"] = np.arange(10)
    array["float"] = np.random.random(10)

    model = NumpyRecArrayTableModel(
        array,
        fill_values={"U": "0", "i": 0},
        name_formats={"int": "{:.1f}"},
        name_flags={"str": ~QtCore.Qt.ItemFlag.ItemIsEditable},
    )

    assert model.columnCount() == 3
    assert model.rowCount() == 10
    # Header
    assert (
        model.headerData(
            0, QtCore.Qt.Orientation.Horizontal, QtCore.Qt.ItemDataRole.DisplayRole
        )
        == "str"
    )
    assert (
        model.headerData(
            1, QtCore.Qt.Orientation.Horizontal, QtCore.Qt.ItemDataRole.DisplayRole
        )
        == "int"
    )
    assert (
        model.headerData(
            2, QtCore.Qt.Orientation.Horizontal, QtCore.Qt.ItemDataRole.DisplayRole
        )
        == "float"
    )

    for i in range(10):
        assert model.headerData(
            i, QtCore.Qt.Orientation.Vertical, QtCore.Qt.ItemDataRole.DisplayRole
        ) == str(i)

    # Col flags
    assert model.flags(model.index(1, 1)) & QtCore.Qt.ItemFlag.ItemIsEditable
    assert not model.flags(model.index(1, 0)) & QtCore.Qt.ItemFlag.ItemIsEditable

    # Insert
    model.insertRows(1, 1)
    assert model.array.shape == (11,)
    assert model.array[1]["str"] == "0"
    assert model.array[1]["int"] == 0
    assert np.isnan(model.array[1]["float"])

    # Data
    assert model.data(model.index(1, 1)) == "0.0"
    assert model.data(model.index(1, 2)) == ""  # nan

    model.setData(model.index(1, 1), 10, QtCore.Qt.ItemDataRole.EditRole)
    assert model.array["int"][1] == 10

    qtmodeltester.check(model, force_py=True)


def test_numpy_recarray_table_model_horizontal(qtmodeltester: ModelTester):
    array = np.empty(10, dtype=[("str", "U16"), ("int", int), ("float", float)])
    array["str"] = "A"
    array["int"] = np.arange(10)
    array["float"] = np.random.random(10)

    model = NumpyRecArrayTableModel(
        array,
        orientation=QtCore.Qt.Orientation.Horizontal,
        fill_values={"U": "0", "i": 0},
        name_formats={"int": "{:.1f}"},
        name_flags={"str": ~QtCore.Qt.ItemFlag.ItemIsEditable},
    )

    assert model.columnCount() == 10
    assert model.rowCount() == 3
    # Header
    assert (
        model.headerData(
            0, QtCore.Qt.Orientation.Vertical, QtCore.Qt.ItemDataRole.DisplayRole
        )
        == "str"
    )
    assert (
        model.headerData(
            1, QtCore.Qt.Orientation.Vertical, QtCore.Qt.ItemDataRole.DisplayRole
        )
        == "int"
    )
    assert (
        model.headerData(
            2, QtCore.Qt.Orientation.Vertical, QtCore.Qt.ItemDataRole.DisplayRole
        )
        == "float"
    )

    for i in range(10):
        assert model.headerData(
            i, QtCore.Qt.Orientation.Horizontal, QtCore.Qt.ItemDataRole.DisplayRole
        ) == str(i)

    # Col flags
    assert model.flags(model.index(1, 1)) & QtCore.Qt.ItemFlag.ItemIsEditable
    assert not model.flags(model.index(0, 1)) & QtCore.Qt.ItemFlag.ItemIsEditable

    # Insert
    model.insertColumns(1, 1)
    assert model.array.shape == (11,)
    assert model.array[1]["str"] == "0"
    assert model.array[1]["int"] == 0
    assert np.isnan(model.array[1]["float"])

    # Data
    assert model.data(model.index(1, 1)) == "0.0"
    assert model.data(model.index(2, 1)) == ""  # nan

    model.setData(model.index(1, 1), 10, QtCore.Qt.ItemDataRole.EditRole)
    assert model.array["int"][1] == 10

    qtmodeltester.check(model, force_py=True)


def test_search_columns_proxy_model(qtbot: QtBot):
    model = QtGui.QStandardItemModel()
    model.setColumnCount(3)
    model.setRowCount(5)

    proxy = SearchColumnsProxyModel([0, 1])
    proxy.setSourceModel(model)

    view = QtWidgets.QTableView()
    qtbot.addWidget(view)
    view.setModel(proxy)

    for i in range(model.rowCount()):
        for j in range(model.columnCount()):
            item = QtGui.QStandardItem(f"{i},{j}")
            model.setItem(i, j, item)

    assert proxy.rowCount() == 5
    assert proxy.columnCount() == 3

    proxy.setSearchString("0,0")
    assert proxy.rowCount() == 1
    assert proxy.columnCount() == 3

    proxy.setSearchString("0,2")
    assert proxy.rowCount() == 0
    assert proxy.columnCount() == 3

    proxy.setSearchString("1")
    assert proxy.rowCount() == 5
    assert proxy.columnCount() == 3

    proxy.setSearchString(",1")
    assert proxy.rowCount() == 5
    assert proxy.columnCount() == 3

    proxy.setSearchString("1,1")
    assert proxy.rowCount() == 1
    assert proxy.columnCount() == 3

    proxy.setSearchString("2")
    assert proxy.rowCount() == 1
    assert proxy.columnCount() == 3


def test_response_models(qtmodeltester: ModelTester, random_datafile_gen: Callable):
    isotopes = [
        ISOTOPE_TABLE[("Fe", 56)],
        ISOTOPE_TABLE[("Cu", 63)],
        ISOTOPE_TABLE[("Zn", 66)],
    ]

    concs = {
        random_datafile_gen(isotopes=isotopes, size=10): {iso: 1.0 for iso in isotopes}
    }

    conc_model = ConcentrationModel()
    conc_model.beginResetModel()
    conc_model.isotopes = isotopes  # type: ignore
    conc_model.concentrations = concs  # type: ignore
    conc_model.endResetModel()

    assert (
        conc_model.data(conc_model.index(0, 0), QtCore.Qt.ItemDataRole.EditRole) == 1.0
    )
    assert conc_model.data(conc_model.index(0, 0), IsotopeRole) == isotopes[0]

    qtmodeltester.check(conc_model)

    intensities = {random_datafile_gen(isotopes=isotopes, size=10): {}}

    intensity_model = IntensityModel()
    intensity_model.beginResetModel()
    intensity_model.isotopes = isotopes  # type: ignore
    intensity_model.intensities = intensities
    intensity_model.endResetModel()

    assert np.isclose(
        intensity_model.data(
            intensity_model.index(0, 0), QtCore.Qt.ItemDataRole.EditRole
        ),
        np.mean(next(iter(intensities.keys())).signals[str(isotopes[0])]),
    )
    assert intensity_model.data(intensity_model.index(0, 0), IsotopeRole) == isotopes[0]

    qtmodeltester.check(intensity_model)


# def test_units_model(qtbot: QtBot):

# def test_units_header_view(qtbot: QtBot):


def test_value_widget_delegate(qtbot: QtBot):
    table = QtWidgets.QTableWidget()
    qtbot.addWidget(table)

    table.setRowCount(1)
    table.setColumnCount(2)
    table.setItem(0, 0, QtWidgets.QTableWidgetItem(""))
    table.setItem(0, 1, QtWidgets.QTableWidgetItem(""))
    table.setItemDelegate(ValueWidgetDelegate())

    with qtbot.waitExposed(table):
        table.show()

    delegate = table.itemDelegateForIndex(table.model().index(0, 0))
    assert isinstance(delegate, ValueWidgetDelegate)
    editor = delegate.createEditor(
        table, QtWidgets.QStyleOptionViewItem(), table.model().index(0, 0)
    )
    assert isinstance(editor, ValueWidget)
    assert editor.value() is None
    editor.setValue(10.0)
    with qtbot.waitSignal(table.model().dataChanged, timeout=100):
        delegate.setModelData(editor, table.model(), table.model().index(0, 0))

    assert table.item(0, 0).text() == "10"  # type: ignore
