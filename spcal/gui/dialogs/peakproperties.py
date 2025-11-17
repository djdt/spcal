import numpy as np
from PySide6 import QtCore, QtWidgets

from spcal.detection import detection_maxima
from spcal.gui.modelviews.basic import BasicTable
from spcal.gui.modelviews.isotope import IsotopeComboBox
from spcal.isotope import SPCalIsotope
from spcal.processing import SPCalProcessingResult
from spcal.siunits import time_units


class PeakPropertiesDialog(QtWidgets.QDialog):
    def __init__(
        self,
        results: dict[SPCalIsotope, SPCalProcessingResult],
        current: SPCalIsotope,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("SPCal Peak Properties")
        self.resize(600, 400)

        self.results = results

        self.combo_isotope = IsotopeComboBox()
        self.combo_isotope.addIsotopes(list(results.keys()))
        self.combo_isotope.isotopeChanged.connect(self.updateValues)

        self.table = BasicTable()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnCount(5)
        self.table.setRowCount(3)

        for i in range(3):
            for j in range(5):
                item = QtWidgets.QTableWidgetItem()
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(i, j, item)

        self.table.setHorizontalHeaderLabels(["min", "max", "mean", "std", "median"])
        self.table.setVerticalHeaderLabels(["width", "height", "skew"])

        self.width_units = QtWidgets.QComboBox()
        self.width_units.addItems(list(time_units.keys()))
        self.width_units.setCurrentText("μs")
        self.width_units.currentTextChanged.connect(self.updateValues)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.combo_isotope, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.table, 1)

        width_layout = QtWidgets.QHBoxLayout()
        width_layout.addStretch(1)
        width_layout.addWidget(
            QtWidgets.QLabel("Width units:"), 0, QtCore.Qt.AlignmentFlag.AlignRight
        )
        width_layout.addWidget(self.width_units, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addLayout(width_layout)

        self.updateValues()

        self.setLayout(layout)

    def clear(self):
        for i in range(3):
            for j in range(5):
                item = self.table.item(i, j)
                if item is None:
                    raise ValueError("item is None")
                item.setText("")

    def updateValues(self):
        result = self.results[self.combo_isotope.currentIsotope()]

        if result.detections.size == 0:
            self.clear()
            return

        maxima = detection_maxima(result.signals, result.regions)

        # heights from peak maxima to baseline
        heights = result.signals[maxima] - result.limit.mean_signal

        widths = result.times[result.regions[:, 1]] - result.times[result.regions[:, 0]]
        # symmetry as how offcenter peak maxima is
        skews = (
            (result.times[maxima] - result.times[result.regions[:, 0]]) / widths - 0.5
        ) * 2.0
        # widths converted to seconds
        widths /= time_units[self.width_units.currentText()]

        sf = int(QtCore.QSettings().value("SigFigs", 4))  # type: ignore

        for i, x in enumerate([widths, heights, skews]):
            self.table.item(i, 0).setText(f"{np.min(x):.{sf}g}")
            self.table.item(i, 1).setText(f"{np.max(x):.{sf}g}")
            self.table.item(i, 2).setText(f"{np.mean(x):.{sf}g}")
            self.table.item(i, 3).setText(f"{np.std(x):.{sf}g}")
            self.table.item(i, 4).setText(f"{np.median(x):.{sf}g}")
