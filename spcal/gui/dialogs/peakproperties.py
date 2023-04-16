import numpy as np
from PySide6 import QtCore, QtWidgets

from spcal.detection import detection_maxima
from spcal.gui.inputs import InputWidget
from spcal.siunits import time_units


class PeakPropertiesDialog(QtWidgets.QDialog):
    def __init__(
        self,
        input: InputWidget,
        current_name: str | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("SPCal Peak Properties")
        self.resize(600, 400)

        self.input = input

        self.combo_names = QtWidgets.QComboBox()
        self.combo_names.addItems(self.input.detection_names)
        if current_name is not None:
            self.combo_names.setCurrentText(current_name)
        self.combo_names.currentTextChanged.connect(self.updateValues)

        self.table = QtWidgets.QTableWidget()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnCount(5)
        self.table.setRowCount(3)

        for i in range(3):
            for j in range(5):
                item = QtWidgets.QTableWidgetItem()
                self.table.setItem(i, j, item)

        self.table.setHorizontalHeaderLabels(["min", "max", "mean", "std", "median"])
        self.table.setVerticalHeaderLabels(["width", "height", "skew"])

        self.width_units = QtWidgets.QComboBox()
        self.width_units.addItems(list(time_units.keys()))
        self.width_units.setCurrentText("μs")
        self.width_units.currentTextChanged.connect(self.updateValues)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.combo_names, 0, QtCore.Qt.AlignmentFlag.AlignRight)
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

    def updateValues(self) -> None:
        name = self.combo_names.currentText()

        detected = np.flatnonzero(self.input.detections[name])
        regions = self.input.regions[detected]
        trim = self.input.trimRegion(name)
        response = self.input.trimmedResponse(name)
        maxima = detection_maxima(response, regions) + trim[0]

        # heights from peak maxima to baseline
        heights = response[maxima] - self.input.limits[name].mean_background

        widths = regions[:, 1] - regions[:, 0]
        # symmetry as how offcenter peak maxima is
        skews = ((maxima - regions[:, 0]) / widths - 0.5) * 2.0
        # widths converted to seconds
        widths = widths.astype(np.float64) * self.input.options.dwelltime.baseValue()
        widths /= time_units[self.width_units.currentText()]

        sf = QtCore.QSettings().value("sigfigs")

        for i, x in enumerate([widths, heights, skews]):
            self.table.item(i, 0).setText(f"{np.min(x):.{sf}g}")
            self.table.item(i, 1).setText(f"{np.max(x):.{sf}g}")
            self.table.item(i, 2).setText(f"{np.mean(x):.{sf}g}")
            self.table.item(i, 3).setText(f"{np.std(x):.{sf}g}")
            self.table.item(i, 4).setText(f"{np.median(x):.{sf}g}")
