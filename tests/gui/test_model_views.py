from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from spcal.gui.modelviews.basic import BasicTableView
from spcal.gui.modelviews.datafile import DataFileDelegate, DataFileModel
from spcal.gui.modelviews.headers import CheckableHeaderView, ComboHeaderView
from spcal.gui.modelviews.isotope import (
    IsotopeComboBox,
    IsotopeModel,
    IsotopeComboDelegate,
    IsotopeNameDelegate,
    IsotopeNameValidator,
)
# from spcal.gui.modelviews.models import


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
