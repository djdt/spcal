from pathlib import Path
from typing import ClassVar

import h5py
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtGui import QValidator

from spcal.dists.util import extract_compound_poisson_lognormal_parameters
from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.singleion import (
    SingleIonAreaScatterView,
)
from spcal.gui.io import get_open_spcal_path
from spcal.io import nu, tofwerk
from spcal.processing.method import SPCalProcessingMethod


class OddValueSpinBox(QtWidgets.QSpinBox):
    def stepBy(self, steps: int):
        steps = steps * self.singleStep() * 2
        self.setValue(self.value() + steps)

    def validate(self, input: str, pos: int) -> QValidator.State:
        try:
            value = int(input)
        except ValueError:
            return QValidator.State.Invalid
        if value % 2 != 1:
            return QValidator.State.Intermediate
        return QValidator.State.Acceptable


class SingleIonAreaSignalsPopup(QtWidgets.QDialog):
    def __init__(
        self, mz: float, y: np.ndarray, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent)
        self.view = SinglePlotGraphicsView(f"{mz:2f} m/z", ylabel="Counts")
        self.view.plot.xaxis.setVisible(False)
        self.view.setInteractive(False)
        self.setWindowTitle("Single Ion Inspection")
        self.setWindowFlags(QtCore.Qt.WindowType.Popup)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.view.plot.drawCurve(np.arange(y.size), y)

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(QtCore.QMargins(1, 1, 1, 1))
        layout.addWidget(self.view)
        self.setLayout(layout)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(
            int(400 * self.devicePixelRatio()), int(200 * self.devicePixelRatio())
        )


class SingleIonAreaDialog(QtWidgets.QDialog):
    resetRequested = QtCore.Signal()
    parametersExtracted = QtCore.Signal(np.ndarray)

    NUM_ZEROS_FOR_ERROR: ClassVar = {1: 350, 2: 2100, 5: 8900}

    def __init__(
        self, params: np.ndarray | None = None, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent)
        self.setWindowTitle("Single Ion Distribution")

        self.scatter = SingleIonAreaScatterView()
        self.scatter.pointClicked.connect(self.onPointClicked)

        self.masses = np.array([])
        self.counts = np.array([])

        self.lams = np.array([])
        self.mus = np.array([])
        self.sigmas = np.array([])
        self.valid = np.array([])

        self.screening_method = SPCalProcessingMethod()
        self.screening_method.limit_options.limit_method = "poisson"
        self.screening_method.limit_options.poisson_kws["alpha"] = 1e-7

        self.required_nonzero = QtWidgets.QSpinBox()
        self.required_nonzero.setRange(0, 10000)
        self.required_nonzero.setValue(350)  # approx 5 % error
        self.required_nonzero.setSingleStep(1000)

        self.max_sigma_difference = QtWidgets.QDoubleSpinBox()
        self.max_sigma_difference.setRange(0.01, 1.0)
        self.max_sigma_difference.setValue(0.1)
        self.max_sigma_difference.setSingleStep(0.01)
        self.max_sigma_difference.valueChanged.connect(self.updateValidParameters)

        self.smoothing = OddValueSpinBox()
        self.smoothing.setSpecialValueText("None")
        self.smoothing.setRange(1, 9)
        self.smoothing.setValue(-1)
        self.smoothing.setSingleStep(1)
        self.smoothing.valueChanged.connect(self.updateScatterInterp)

        self.controls_box = QtWidgets.QGroupBox()
        controls_layout = QtWidgets.QFormLayout()
        controls_layout.addRow("Dist. from mean:", self.max_sigma_difference)
        controls_layout.addRow("Smoothing:", self.smoothing)
        self.controls_box.setLayout(controls_layout)
        self.enableControls(False)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Reset
            | QtWidgets.QDialogButtonBox.StandardButton.Open
            | QtWidgets.QDialogButtonBox.StandardButton.Apply
            | QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        self.button_box.clicked.connect(self.buttonPressed)
        self.button_box.button(
            QtWidgets.QDialogButtonBox.StandardButton.Apply
        ).setEnabled(False)

        layout = QtWidgets.QVBoxLayout()
        layout_horz = QtWidgets.QHBoxLayout()
        layout_horz.addWidget(self.controls_box, 0)
        layout_horz.addWidget(self.scatter, 1)
        layout.addLayout(layout_horz, 1)
        layout.addWidget(self.button_box, 0)

        self.setLayout(layout)

        # A 'read-only' mode for existing parameters
        if params is not None and params.size > 0:
            self.scatter.drawData(params["mass"], params["sigma"])

    @QtCore.Slot()
    def onPointClicked(self, pos: QtCore.QPointF, index: int):
        sia = np.exp(self.mus[index] + 0.5 * self.sigmas[index] ** 2)
        popup = SingleIonAreaSignalsPopup(
            pos.x(), self.counts[:, index] / sia, parent=self
        )
        popup.show()

    def buttonPressed(self, button: QtWidgets.QAbstractButton):
        sb = self.button_box.standardButton(button)
        if sb == QtWidgets.QDialogButtonBox.StandardButton.Reset:
            self.clear()
            self.resetRequested.emit()
        elif sb == QtWidgets.QDialogButtonBox.StandardButton.Apply:
            self.accept()
        elif sb == QtWidgets.QDialogButtonBox.StandardButton.Open:
            self.loadSingleIonData()
        elif sb == QtWidgets.QDialogButtonBox.StandardButton.Close:
            self.reject()

    def completeChanged(self):
        button = self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Apply)
        button.setEnabled(self.isComplete())

    def isComplete(self) -> bool:
        return bool(self.valid.size > 0 and np.any(self.valid))

    def enableControls(self, enabled: bool):
        self.controls_box.setEnabled(enabled)

    def clear(self):
        self.masses = np.array([])
        self.counts = np.array([])
        self.lams = np.array([])
        self.mus = np.array([])
        self.sigmas = np.array([])
        self.valid = np.array([])

        # self.hist.clear()
        self.scatter.clear()

        self.enableControls(False)

    def loadSingleIonData(self, path: str | Path | None = None):
        if path is None:
            path = get_open_spcal_path(self, "Single Ion Data")
            if path is None:
                return
        else:
            path = Path(path)
        if nu.is_nu_directory(path) or nu.is_nu_run_info_file(path):
            self.masses, self.counts, _, info = nu.read_directory(
                path, autoblank="all", raw=True
            )
            self.reported_mu = info["AverageSingleIonArea"]
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
                self.reported_mu = np.log(
                    h5["FullSpectra"].attrs["Single Ion Signal"][0]
                )
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid File",
                f"'{path.stem}' is not a valid TOF data file.\nOnly Nu Instruments and TOFWERK data is supported.",
            )
            raise ValueError(f"{path.stem} is neither a Nu or TOFWERK file")

        self.updateExtractedParameters()
        self.enableControls(True)

    def updateExtractedParameters(self):
        self.scatter.clear()
        if not self.max_sigma_difference.hasAcceptableInput():
            return

        self.lams, self.mus, self.sigmas = (
            extract_compound_poisson_lognormal_parameters(self.counts).T
        )

        self.scatter.drawData(self.masses, self.sigmas)

        self.updateValidParameters()

    def updateValidParameters(self):
        # most likely invalid
        outside_sigma_range = np.logical_or(self.sigmas < 0.2, self.sigmas > 0.95)
        # outside_lambda_range = np.logical_or(self.lams < 0.005, self.lams > 10.0)

        nonzeros = np.count_nonzero(self.counts, axis=0)
        zeros = self.counts.shape[0] - nonzeros

        insufficient_zeros = zeros < 150  # approx 5 % error
        insufficient_nonzeros = nonzeros < self.required_nonzero.value()

        idx_error = np.zeros(self.counts.shape[1], int)
        idx_error[outside_sigma_range] = 1
        idx_error[insufficient_zeros] = 2
        idx_error[insufficient_nonzeros] = 3

        valid = idx_error == 0

        poly = np.polynomial.Polynomial.fit(self.masses[valid], self.sigmas[valid], 1)

        self.valid = (
            np.abs(self.sigmas - poly(self.masses)) < self.max_sigma_difference.value()
        )
        self.valid = np.logical_and(self.valid, valid)

        if self.scatter.points is not None:
            brushes = np.array(
                [
                    QtGui.QBrush(QtCore.Qt.GlobalColor.black),
                    QtGui.QBrush(QtCore.Qt.GlobalColor.red),
                    QtGui.QBrush(QtCore.Qt.GlobalColor.yellow),
                    QtGui.QBrush(QtCore.Qt.GlobalColor.yellow),
                ]
            )
            symbols = np.array(["o", "x", "t", "t1"])
            self.scatter.points.setBrush(brushes[idx_error])
            self.scatter.points.setSymbol(symbols[idx_error])

        self.scatter.drawMaxDifference(poly, self.max_sigma_difference.value())

        mean_mu = np.mean(self.mus[self.valid])
        mean_sigma = np.mean(self.sigmas[self.valid])

        self.scatter.plot.setTitle(f"Average: µ={mean_mu:.2f}, σ={mean_sigma:.2f}")

        self.completeChanged()

        self.updateScatterInterp()

    def updateScatterInterp(self):
        xs, ys = self.smoothedParameters(
            self.masses[self.valid], self.sigmas[self.valid]
        )
        self.scatter.drawInterpolationLine(xs, ys)

    def smoothedParameters(
        self, xs: np.ndarray, ys: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        smoothing = self.smoothing.value()
        if smoothing < 3:
            return xs, ys
        elif smoothing % 2 == 1:
            _xs = np.arange(xs[0], xs[-1] + 1.0, 1.0)
            _ys = np.interp(_xs, xs, ys)
            _ys[smoothing // 2 - 1 : -(smoothing // 2 + 1)] = np.convolve(
                _ys, np.ones(smoothing) / smoothing, mode="valid"
            )
            return xs, np.interp(xs, _xs, _ys)
        else:
            raise ValueError(f"invalid smoothing window {smoothing}")

    def accept(self):
        if self.masses.size > 0:
            mz, mu = self.smoothedParameters(
                self.masses[self.valid], self.mus[self.valid]
            )
            _, sigma = self.smoothedParameters(
                self.masses[self.valid], self.sigmas[self.valid]
            )
            params = np.empty(
                mz.size, dtype=[("mass", float), ("mu", float), ("sigma", float)]
            )
            params["mass"] = mz
            params["mu"] = mu
            params["sigma"] = sigma
            self.parametersExtracted.emit(params)
        super().accept()


if __name__ == "__main__":
    # options
    # 1. manual input of single SIA shape (like old)
    # 2.
    app = QtWidgets.QApplication()

    win = SingleIonAreaDialog()
    # win.loadSingleIonData("/home/tom/Downloads/NT032/14-37-30 1 ppb att")
    # win.loadSingleIonData("/home/tom/Downloads/NT032/14-36-31 10 ppb att/")
    win.loadSingleIonData("/home/tom/Downloads/NT032/14-35-55 10 ppb unatt/")
    # win.loadSingleIonData("/mnt/storage/TOF/2026 Greenland Ice/13-02-23 mix10ppb/")
    win.show()

    app.exec()
