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
