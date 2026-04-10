import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.modelviews.basic import BasicTableView
from spcal.gui.modelviews.isotope import IsotopeComboBox

from spcal.isotope import SPCalIsotopeBase
from spcal.processing.result import SPCalProcessingResult
from spcal.siunits import time_units

from spcal.gui.io import get_save_spcal_path


class PeakPropertiesDialog(QtWidgets.QDialog):
    def __init__(
        self,
        results: dict[SPCalIsotopeBase, SPCalProcessingResult],
        current: SPCalIsotopeBase,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("SPCal Peak Properties")
        self.resize(600, 600)

        self.view = SinglePlotGraphicsView("Peak Properties")

        self.results = results
        self.widths: np.ndarray | None = None
        self.heights: np.ndarray | None = None
        self.skews: np.ndarray | None = None

        self.combo_isotope = IsotopeComboBox()
        self.combo_isotope.addIsotopes(list(results.keys()))
        self.combo_isotope.setCurrentIsotope(current)
        self.combo_isotope.isotopeChanged.connect(self.updateValues)

        self.table = BasicTableView()
        self.model = QtGui.QStandardItemModel()
        self.table.setModel(self.model)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.model.setColumnCount(5)
        self.model.setRowCount(3)
        self.table.setCurrentIndex(self.model.index(0, 0))
        self.table.selectionModel().currentChanged.connect(self.updateGraph)

        for i in range(3):
            for j in range(5):
                item = QtGui.QStandardItem()
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.model.setItem(i, j, item)

        self.model.setHorizontalHeaderLabels(["min", "max", "mean", "std", "median"])
        self.model.setVerticalHeaderLabels(["width", "height", "skew"])

        self.width_units = QtWidgets.QComboBox()
        self.width_units.addItems(list(time_units.keys()))
        self.width_units.setCurrentText("µs")
        self.width_units.currentTextChanged.connect(self.updateValues)

        self.button_export = QtWidgets.QPushButton("Export")
        self.button_export.setIcon(QtGui.QIcon.fromTheme("document-save"))
        self.button_export.pressed.connect(self.dialogSave)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.view, 2)
        layout.addWidget(self.combo_isotope, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.table, 1)

        width_layout = QtWidgets.QHBoxLayout()
        width_layout.addWidget(self.button_export, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
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
                item = self.model.item(i, j)
                if item is None:
                    raise ValueError("item is None")
                item.setText("")

    def dialogSave(self):
        if self.widths is None or self.heights is None or self.skews is None:
            return

        path = get_save_spcal_path(self, [("CSV Documents", ".csv")])
        if path is None:
            return

        data = np.stack((self.widths, self.heights, self.skews), axis=1)
        with path.open("w") as fp:
            fp.write("width,height,skew\n")
            np.savetxt(fp, data, delimiter=",")

    def updateGraph(self):
        self.view.clear()
        if self.widths is None or self.heights is None or self.skews is None:
            return
        row = self.table.currentIndex().row()
        if row == 0:
            data, label = (
                self.widths / time_units[self.width_units.currentText()],
                "Width",
            )
            label += f" ({self.width_units.currentText()})"
        elif row == 1:
            data, label = self.heights, "Height"
        elif row == 2:
            data, label = self.skews, "Skew"
        else:
            return

        counts, edges = np.histogram(data)
        self.view.data_for_export[f"{label.lower()}_counts"] = counts
        self.view.data_for_export[f"{label.lower()}_edges"] = edges

        self.view.plot.xaxis.setLabel(label)
        self.view.plot.drawHistogram(counts, edges, width=1.0)

    def updateValues(self):
        result = self.results[self.combo_isotope.currentIsotope()]

        if result.detections.size == 0:
            self.clear()
            return

        # heights from peak maxima to baseline
        self.heights = result.signals[result.maxima] - result.limit.mean_signal

        self.widths = (
            result.times[result.regions[:, 1]] - result.times[result.regions[:, 0]]
        )
        # symmetry as how offcenter peak maxima is
        self.skews = (
            (result.times[result.maxima] - result.times[result.regions[:, 0]])
            / self.widths
            - 0.5
        ) * 2.0

        sf = int(QtCore.QSettings().value("SigFigs", 4))  # type: ignore

        for i, x in enumerate(
            [
                self.widths / time_units[self.width_units.currentText()],
                self.heights,
                self.skews,
            ]
        ):
            self.model.item(i, 0).setText(f"{np.min(x):.{sf}g}")
            self.model.item(i, 1).setText(f"{np.max(x):.{sf}g}")
            self.model.item(i, 2).setText(f"{np.mean(x):.{sf}g}")
            self.model.item(i, 3).setText(f"{np.std(x):.{sf}g}")
            self.model.item(i, 4).setText(f"{np.median(x):.{sf}g}")

        self.updateGraph()
