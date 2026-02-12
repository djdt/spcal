from pathlib import Path

import h5py
import numpy as np
from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QValidator

from spcal.dists.util import (
    extract_compound_poisson_lognormal_parameters,
)
from spcal.gui.graphs.singleion import SingleIonHistogramView, SingleIonScatterView
from spcal.gui.io import get_open_spcal_path
from spcal.io import nu, tofwerk


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


class SingleIonDialog(QtWidgets.QDialog):
    resetRequested = QtCore.Signal()
    parametersExtracted = QtCore.Signal(np.ndarray)

    def __init__(
        self,
        params: np.ndarray | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Single Ion Distribution")

        self.hist = SingleIonHistogramView()
        self.hist.plot.yaxis.setLabel("Event Density")
        assert self.hist.plot.vb is not None
        self.hist.plot.vb.setMouseEnabled(x=False, y=False)
        self.scatter = SingleIonScatterView()

        self.masses = np.array([])
        self.counts = np.array([])

        self.lams = np.array([])
        self.mus = np.array([])
        self.sigmas = np.array([])
        self.valid = np.array([])

        self.minimum_value = QtWidgets.QDoubleSpinBox()
        self.minimum_value.setRange(0.0, 1e9)
        self.minimum_value.setValue(0.0)
        self.minimum_value.setStepType(
            QtWidgets.QDoubleSpinBox.StepType.AdaptiveDecimalStepType
        )
        self.minimum_value.valueChanged.connect(self.updateExtractedParameters)
        self.minimum_value.valueChanged.connect(self.updateMinMaxValueRanges)

        self.maximum_value = QtWidgets.QDoubleSpinBox()
        self.maximum_value.setRange(0.0, 1e9)
        self.maximum_value.setValue(0.0)
        self.maximum_value.setStepType(
            QtWidgets.QDoubleSpinBox.StepType.AdaptiveDecimalStepType
        )
        self.maximum_value.valueChanged.connect(self.updateExtractedParameters)
        self.maximum_value.valueChanged.connect(self.updateMinMaxValueRanges)

        layout_range = QtWidgets.QHBoxLayout()
        layout_range.addWidget(self.minimum_value)
        layout_range.addWidget(QtWidgets.QLabel("-"))
        layout_range.addWidget(self.maximum_value)

        self.check_restrict = QtWidgets.QCheckBox("Restrict to single ion signals.")
        self.check_restrict.checkStateChanged.connect(self.updateValidParameters)

        self.hist_controls_box = QtWidgets.QGroupBox()
        hist_controls_box_layout = QtWidgets.QFormLayout()
        hist_controls_box_layout.addRow("Range:", layout_range)
        hist_controls_box_layout.addRow(self.check_restrict)
        self.hist_controls_box.setLayout(hist_controls_box_layout)

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

        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.hist, 0, 0)
        layout.addWidget(self.hist_controls_box, 1, 0)
        layout.addWidget(self.scatter, 0, 1)
        layout.addWidget(self.controls_box, 1, 1)
        layout.addWidget(self.button_box, 2, 0, 1, 2)

        layout.setColumnStretch(0, 2)
        layout.setColumnStretch(1, 3)
        self.setLayout(layout)

        # A 'read-only' mode for existing parameters
        if params is not None and params.size > 0:
            self.scatter.drawData(params["mass"], params["sigma"])

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
        self.hist_controls_box.setEnabled(enabled)

    def clear(self):
        self.masses = np.array([])
        self.counts = np.array([])
        self.lams = np.array([])
        self.mus = np.array([])
        self.sigmas = np.array([])
        self.valid = np.array([])

        self.hist.clear()
        self.scatter.clear()

        self.enableControls(False)

    def updateMinMaxValueRanges(self):
        if self.minimum_value.hasAcceptableInput():
            min = self.minimum_value.value()
            self.maximum_value.setRange(min, 1e9)
        if self.maximum_value.hasAcceptableInput():
            max = self.maximum_value.value()
            self.minimum_value.setRange(0.0, max)

    def loadSingleIonData(self, path: str | Path | None = None):
        if path is None:
            path = get_open_spcal_path(self, "Single Ion Data")
            if path is None:
                return
        else:
            path = Path(path)
        if nu.is_nu_directory(path) or nu.is_nu_run_info_file(path):
            self.masses, self.counts, _, info = nu.read_directory(path, raw=True)
            self.reported_mu = info["AverageSingleIonArea"]
        elif tofwerk.is_tofwerk_file(path):
            with h5py.File(path, "r") as h5:
                if "PeakData" in h5["PeakData"]:  # type: ignore
                    data = h5["PeakData"]["PeakData"]  # type: ignore
                else:  # pragma: no cover, covered above
                    data = tofwerk.integrate_tof_data(h5)
                self.masses = np.asarray(h5["PeakData"]["PeakTable"]["mass"])  # type: ignore
                self.counts = (
                    data
                    * h5["FullSpectra"].attrs["Single Ion Signal"][0]  # type: ignore
                    * tofwerk.factor_extraction_to_acquisition(h5)
                ).reshape(-1, self.masses.size)
                self.reported_mu = np.log(
                    h5["FullSpectra"].attrs["Single Ion Signal"][0]  # type: ignore
                )
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid File",
                f"'{path.stem}' is not a valid TOF data file.\nOnly Nu Instruments and TOFWERK data is supported.",
            )
            raise ValueError(f"{path.stem} is neither a Nu or TOFWERK file")

            self.masses, self.counts, info = nu.read_directory(path, raw=True)
            self.reported_mu = np.log(info["AverageSingleIonArea"])

        # Remove clearly gaussian signals
        zeros = np.count_nonzero(self.counts == 0, axis=0)
        self.masses, self.counts = self.masses[zeros > 0], self.counts[:, zeros > 0]

        max = np.nanmax(self.counts)
        self.minimum_value.setRange(0.0, max)
        self.maximum_value.setRange(0.0, max)
        self.minimum_value.setValue(0.0)
        self.maximum_value.setValue(max)

        self.updateExtractedParameters()
        self.enableControls(True)

    def updateExtractedParameters(self):
        self.scatter.clear()
        if not self.max_sigma_difference.hasAcceptableInput():
            return
        mask = np.logical_and(
            self.counts >= self.minimum_value.value(),
            self.counts <= self.maximum_value.value(),
        )
        mask = np.logical_or(self.counts == 0, mask)
        self.lams, self.mus, self.sigmas = (
            extract_compound_poisson_lognormal_parameters(self.counts, mask).T
        )

        self.scatter.drawData(self.masses, self.sigmas)

        self.updateValidParameters()

    def updateValidParameters(self):
        # most likely invalid
        valid = np.logical_and(
            np.logical_and(self.sigmas > 0.2, self.sigmas < 0.95),
            np.logical_and(self.lams > 0.005, self.lams < 10.0),
        )
        if not np.any(valid):
            return

        poly = np.polynomial.Polynomial.fit(self.masses[valid], self.sigmas[valid], 1)

        self.valid = (
            np.abs(self.sigmas - poly(self.masses)) < self.max_sigma_difference.value()
        )

        if self.check_restrict.isChecked():
            p0 = np.exp(-self.lams)
            p2 = (self.lams**2 * p0) / 2.0
            single_ions = np.logical_and(p0 > 1e-2, p2 < 1e-3)
            self.valid = np.logical_and(self.valid, single_ions)

        self.scatter.setValid(self.valid)
        self.scatter.drawMaxDifference(poly, self.max_sigma_difference.value())

        mean_mu, mean_sigma = (
            np.mean(self.mus[self.valid]),
            np.mean(self.sigmas[self.valid]),
        )

        self.scatter.plot.setTitle(f"Average: µ={mean_mu:.2f}, σ={mean_sigma:.2f}")

        self.completeChanged()

        self.updateScatterInterp()
        self.updateHistogram()

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
        return xs, ys

    def updateHistogram(self):
        if np.count_nonzero(self.valid) > 0:
            counts = self.counts[:, self.valid]

            mask = np.logical_and(
                counts > self.minimum_value.value(),
                counts <= self.maximum_value.value(),
            )

            hist, edges = np.histogram(
                counts[mask],
                bins=200,
                density=True,
            )
            self.hist.drawHist(hist, edges)
        elif self.hist.hist_curve is not None:
            self.hist.hist_curve.clear()

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

    win = SingleIonDialog()
    win.loadSingleIonData("/home/tom/Downloads/NT032/14-37-30 1 ppb att")
    # win.loadSingleIonData("/home/tom/Downloads/NT032/14-36-31 10 ppb att/")
    win.show()

    app.exec()
