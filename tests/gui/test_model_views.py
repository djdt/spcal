import numpy as np
from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot
from pytestqt.modeltest import ModelTester

from spcal.datafile import SPCalTextDataFile
from spcal.gui.modelviews import IsotopeRole
from spcal.gui.modelviews.basic import BasicTableView
from spcal.gui.modelviews.datafile import DataFileDelegate, DataFileModel
from spcal.gui.modelviews.headers import CheckableHeaderView, ComboHeaderView
from spcal.gui.modelviews.isotope import (
    IsotopeComboBox,
    IsotopeNameDelegate,
    IsotopeNameValidator,
)
from spcal.isotope import ISOTOPE_TABLE


def random_datafile() -> SPCalTextDataFile:
    data = np.random.random(100).astype([("Au", np.float32)])
    return SPCalTextDataFile(
        Path(),
        data,
        np.linspace(0.0, 1.0, 100),
        isotope_table={ISOTOPE_TABLE[("Au", 197)]: "Au"},
        instrument_type="quadrupole",
    )


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


def test_datafile_model(qtmodeltester: ModelTester):
    model = DataFileModel([random_datafile() for _ in range(5)])
    qtmodeltester.check(model, force_py=True)


def test_datafile_delegate(qtbot: QtBot):
    view = QtWidgets.QListView()
    model = DataFileModel([random_datafile() for _ in range(5)])
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
