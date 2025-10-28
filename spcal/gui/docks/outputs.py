import numpy as np
from PySide6 import QtCore, QtWidgets

from spcal.calc import mode as modefn
from spcal.gui.widgets.unitstable import UnitsTable
from spcal.processing import SPCalProcessingMethod, SPCalProcessingResult
from spcal.siunits import (
    mass_concentration_units,
    mass_units,
    number_concentration_units,
    signal_units,
    size_units,
    volume_units,
)


class SPCalOutputsDock(QtWidgets.QDockWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Outputs")

        self.results = {}

        self.combo_key = QtWidgets.QComboBox()
        self.combo_key.addItems(SPCalProcessingMethod.CALIBRATION_KEYS)
        self.combo_key.currentTextChanged.connect(self.updateOutputsForKey)

        self.table = UnitsTable(
            [
                ("Number", None, None, None),
                ("Concentration", number_concentration_units, "#/L", None),
                ("Background", signal_units, "cts", None),
                ("LOD", signal_units, "cts", None),
                ("Mean", signal_units, "cts", None),
                ("Median", signal_units, "cts", None),
                ("Mode", signal_units, "cts", None),
            ]
        )
        self.table.setEditTriggers(QtWidgets.QTableView.EditTrigger.NoEditTriggers)
        # self.proto = QtWidgets.QTableWidgetItem()
        # self.proto.setFlags(self.proto.flags() & ~ QtCore.Qt.ItemFlag.ItemIsEditable)
        # self.table.setItemPrototype(self.proto)

        self.row_indicies = {}

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.combo_key, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.table, 1)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setWidget(widget)

    def setIsotopes(self, isotopes: list[str]):
        self.row_indicies = {isotope: i for i, isotope in enumerate(isotopes)}
        self.table.setRowCount(len(isotopes))
        self.table.setVerticalHeaderLabels(isotopes)

    def setResults(self, results: dict[str, SPCalProcessingResult]):
        self.results = results
        for isotope, result in results.items():
            row = self.row_indicies[isotope]
            self.table.setBaseValueForItem(row, 0, result.number, result.number_error)

        self.updateOutputsForKey(self.combo_key.currentText())

    def updateOutputsForKey(self, key: str):
        units = signal_units
        default_unit = "cts"
        conc_units = number_concentration_units
        default_conc_unit = "#/L"

        if key == "mass":
            units = mass_units
            default_unit = "fg"
            default_conc_unit = "µg/L"
            conc_units = mass_concentration_units
        elif key == "size":
            units = size_units
            default_unit = "nm"
        elif key == "volume":
            units = volume_units
            default_unit = "µm³"
        elif key != "signal":
            raise ValueError(f"unknown key '{key}'")

        # calc best unit from mean?

        headers = [
            ("Number", None, None, None),
            ("Concentration", conc_units, default_conc_unit, None),
            ("Background", units, default_unit, None),
            ("LOD", units, default_unit, None),
            ("Mean", units, default_unit, None),
            ("Median", units, default_unit, None),
            ("Mode", units, default_unit, None),
        ]
        self.table.setHeaders(headers)

        for isotope, result in self.results.items():
            row = self.row_indicies[isotope]

            if result.method.canCalibrate(key, result.isotope):
                bg = float(
                    result.method.calibrateTo(result.background, key, result.isotope)
                )
                lod = float(
                    result.method.calibrateTo(
                        float(np.nanmean(result.limit.detection_threshold)),
                        key,
                        result.isotope,
                    )
                )
                bg_error = float(
                    result.method.calibrateTo(
                        result.background_error, key, result.isotope
                    )
                )
            else:
                bg, lod, bg_error = None, None, None

            mean, median, mode, std = None, None, None, None
            detections = result.calibrated(key)

            if detections is not None:
                mean = float(np.mean(detections))
                median = float(np.median(detections))
                mode = float(modefn(detections))
                std = float(np.std(detections))

            self.table.setBaseValueForItem(
                row,
                1,
                result.number_concentration
                if key != "mass"
                else result.mass_concentration,
            )
            self.table.setBaseValueForItem(row, 2, bg, bg_error)
            self.table.setBaseValueForItem(row, 3, lod)
            self.table.setBaseValueForItem(row, 4, mean, std)
            self.table.setBaseValueForItem(row, 5, median)
            self.table.setBaseValueForItem(row, 6, mode)


if __name__ == "__main__":
    app = QtWidgets.QApplication()
    dock = SPCalOutputsDock()
    dock.table.setRowCount(5)

    for i in range(dock.table.rowCount()):
        for j in range(dock.table.columnCount()):
            item = QtWidgets.QTableWidgetItem()
            item.setData(QtCore.Qt.ItemDataRole.EditRole, np.random.random())
            item.setData(QtCore.Qt.ItemDataRole.UserRole, np.random.random())
            dock.table.setItem(i, j, item)

    dock.show()
    # dock.table.itemDelegate(dock.table.indexFromItem(dock.table.item(0, 0))).commitData
    app.exec()
