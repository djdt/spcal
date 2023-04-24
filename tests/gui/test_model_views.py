from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from spcal.gui.modelviews import BasicTable


def test_basic_table(qtbot: QtBot):
    table = BasicTable()
    qtbot.addWidget(table)

    table.setRowCount(2)
    table.setColumnCount(2)

    table.setItem(0, 0, QtWidgets.QTableWidgetItem("a"))
    table.setItem(0, 1, QtWidgets.QTableWidgetItem("b"))
    table.setItem(1, 0, QtWidgets.QTableWidgetItem("c"))
    table.setItem(1, 1, QtWidgets.QTableWidgetItem("d"))

    table.setCurrentCell(0, 0)
    assert table.currentRow() == 0
    table._advance()
    assert table.currentRow() == 1

    table.clearSelection()
    table.setRangeSelected(QtWidgets.QTableWidgetSelectionRange(0, 0, 0, 1), True)
    table._copy()
    mime_data = QtWidgets.QApplication.clipboard().mimeData()
    assert mime_data.text() == "a\tb"

    table.clearSelection()
    table.setRangeSelected(QtWidgets.QTableWidgetSelectionRange(1, 0, 1, 1), True)
    table._cut()  # Same as _copy, _delete
    mime_data = QtWidgets.QApplication.clipboard().mimeData()
    assert mime_data.text() == "c\td"
    assert table.item(1, 0).text() == ""
    assert table.item(1, 1).text() == ""

    table.setRangeSelected(QtWidgets.QTableWidgetSelectionRange(0, 0, 1, 1), True)
    table._delete()
    assert table.item(0, 0).text() == ""
    assert table.item(0, 1).text() == ""

    QtWidgets.QApplication.clipboard().setText("1\t2\n3\t4")
    table._paste()
    assert table.item(0, 0).text() == "1"
    assert table.item(0, 1).text() == "2"
    assert table.item(1, 0).text() == "3"
    assert table.item(1, 1).text() == "4"

    table.contextMenuEvent(
        QtGui.QContextMenuEvent(
            QtGui.QContextMenuEvent.Mouse, QtCore.QPoint(0, 0), QtCore.QPoint(0, 0)
        )
    )
