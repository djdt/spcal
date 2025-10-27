import numpy as np
from PySide6 import QtCore, QtWidgets

from spcal.gui.widgets.unitstable import UnitsTable
from spcal.processing import SPCalProcessingResult
from spcal.siunits import signal_units


class SPCalOutputsDock(QtWidgets.QDockWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Outputs")

        self.combo_key = QtWidgets.QComboBox()

        self.table = UnitsTable(
            [
                ("Count", None, None, None),
                ("PNC", {"#/L": 1.0, "#/ml": 1000.0}, "#/L", None),
                ("Background", signal_units, "counts", None),
                ("LOD", signal_units, "counts", None),
                ("Mean", signal_units, "counts", None),
                ("Median", signal_units, "counts", None),
                ("Mode", signal_units, "counts", None),
            ]
        )
        self.table.setEditTriggers(QtWidgets.QTableView.EditTrigger.NoEditTriggers)

        self.row_indicies = {}
        self.setWidget(self.table)

    def setIsotopes(self, isotopes: list[str]):
        self.row_indicies = {isotope: i for i, isotope in enumerate(isotopes)}
        self.table.setRowCount(len(isotopes))
        self.table.setVerticalHeaderLabels(isotopes)

    def setResults(self, results: dict[str, SPCalProcessingResult]):
        for isotope, result in results.items():
            row = self.row_indicies[isotope]
            self.table.setBaseValueForItem(row, 0, result.number, result.number_error)
            print(result.number)


if __name__ == "__main__":
    app = QtWidgets.QApplication()
    dock = SPCalOutputsDock()
    dock.table.setRowCount(5)

    for i in range(dock.table.rowCount()):
        for j in range(dock.table.columnCount()):
            item = QtWidgets.QTableWidgetItem()
            item.setData(QtCore.Qt.ItemDataRole.EditRole, np.random.random())
            dock.table.setItem(i, j, item)

    dock.show()
    # dock.table.itemDelegate(dock.table.indexFromItem(dock.table.item(0, 0))).commitData
    app.exec()
