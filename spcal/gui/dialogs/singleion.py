from pathlib import Path

import h5py
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.dists.util import extract_compound_poisson_lognormal_parameters
from spcal.gui.graphs.scatter import ScatterView
from spcal.gui.graphs.singleion import SingleIonHistogramView
from spcal.gui.io import get_open_spcal_path
from spcal.gui.modelviews import BasicTable
from spcal.io import nu, tofwerk


class SingleIonDialog(QtWidgets.QDialog):
    distributionSelected = QtCore.Signal(np.ndarray)
    extractedParameters = QtCore.Signal(np.ndarray)

    def __init__(
        self, dist: np.ndarray | None = None, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent)
        self.setWindowTitle("Single Ion Distribution")

        self.hist = SingleIonHistogramView()
        self.scatter = ScatterView()
        self.table = BasicTable()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["m/z", "λ", "µ", "σ"])

        self.masses = np.array([])
        self.counts = np.array([])

        controls_box = QtWidgets.QGroupBox()

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Reset
            | QtWidgets.QDialogButtonBox.StandardButton.Open
            | QtWidgets.QDialogButtonBox.StandardButton.Apply
            | QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        self.button_box.clicked.connect(self.buttonPressed)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(controls_box, 0, 0)
        layout.addWidget(self.scatter, 0, 1)
        layout.addWidget(self.hist, 1, 0)
        layout.addWidget(self.table, 1, 1)
        layout.addWidget(self.button_box, 2, 0, 1, 2)

        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 2)
        self.setLayout(layout)

    def buttonPressed(self, button: QtWidgets.QAbstractButton) -> None:
        sb = self.button_box.standardButton(button)

        if sb == QtWidgets.QDialogButtonBox.StandardButton.Reset:
            pass
        elif sb == QtWidgets.QDialogButtonBox.StandardButton.Open:
            self.loadSingleIonData()
        elif sb == QtWidgets.QDialogButtonBox.StandardButton.Close:
            self.reject()

    def loadSingleIonData(self, path: str | Path | None = None) -> None:
        if path is None:
            path = get_open_spcal_path(self, "Single Ion Data")
            if path is None:
                return
        else:
            path = Path(path)

        # todo: add a progress bar

        if nu.is_nu_directory(path):
            self.masses, self.counts, _ = nu.read_nu_directory(path, raw=True)
        elif tofwerk.is_tofwerk_file(path):
            with h5py.File(path, "r") as h5:
                if "PeakData" in h5["PeakData"]:
                    data = h5["PeakData"]["PeakData"]
                else:  # pragma: no cover, covered above
                    data = tofwerk.integrate_tof_data(h5)
                self.masses = np.asarray(h5["PeakData"]["PeakTable"]["mass"])
                self.counts = (
                    data
                    * h5["FullSpectra"].attrs["Single Ion Signal"][0]
                    * tofwerk.factor_extraction_to_acquisition(h5)
                ).reshape(-1, self.masses.size)
        else:
            raise ValueError(f"{path.stem} is neither a Nu or TOFWERK file")

        # Remove clearly gaussian signals
        zeros = np.count_nonzero(self.counts == 0, axis=0)
        self.masses, self.counts = self.masses[zeros > 0], self.counts[:, zeros > 0]
        self.updateHistogram()
        self.updateExtractedParameters()

    def updateExtractedParameters(self) -> None:
        self.lams, self.mus, self.sigmas = (
            extract_compound_poisson_lognormal_parameters(self.counts)
        )
        self.valid = np.logical_and(
            np.logical_and(self.sigmas > 0.2, self.sigmas < 0.95),
            np.logical_and(self.lams > 0.005, self.lams < 10.0),
        )

        self.hist.drawLognormalFit(np.mean(self.mus[self.valid]), np.mean(self.sigmas[self.valid]))
        self.updateScatter()
        self.updateTable()

    def updateHistogram(self) -> None:
        self.hist.clear()
        if self.counts.size > 0:
            self.hist.draw(self.counts[self.counts > 0])

    def updateScatter(self) -> None:
        self.scatter.clear()
        self.scatter.drawData(self.masses[self.valid], self.sigmas[self.valid])
        self.scatter.drawData(
            self.masses[~self.valid],
            self.sigmas[~self.valid],
            brush=QtGui.QBrush(QtCore.Qt.GlobalColor.red),
        )
        self.scatter.drawFit(self.masses[self.valid], self.sigmas[self.valid])

    def updateTable(self) -> None:
        if self.counts.size == 0 or self.lams.size == 0:
            return
        self.table.setRowCount(self.lams.size)

        for i in range(self.table.rowCount()):
            for j, v in enumerate(
                [self.masses[i], self.lams[i], self.mus[i], self.sigmas[i]]
            ):
                item = QtWidgets.QTableWidgetItem(f"{v:.4f}")
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(i, j, item)

    def accept(self) -> None:
        self.distributionSelected.emit(self.counts)
        self.extractedParameters.emit(
            np.stack(
                (
                    self.masses[self.valid],
                    self.mus[self.valid],
                    self.sigmas[self.valid],
                ),
                axis=1,
            )
        )
        super().accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication()

    win = SingleIonDialog()
    win.loadSingleIonData("/home/tom/Downloads/11-43-47 1ppb att")
    win.show()

    app.exec()
