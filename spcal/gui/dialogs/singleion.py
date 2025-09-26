from pathlib import Path

import h5py
import numpy as np
from PySide6 import QtCore, QtWidgets

from spcal.dists.util import (
    extract_compound_poisson_lognormal_parameters,
)
from spcal.gui.graphs.singleion import SingleIonHistogramView, SingleIonScatterView
from spcal.gui.io import get_open_spcal_path
from spcal.io import nu, tofwerk


class SingleIonDialog(QtWidgets.QDialog):
    # distributionSelected = QtCore.Signal(np.ndarray)
    resetRequested = QtCore.Signal()
    parametersExtracted = QtCore.Signal(np.ndarray)

    def __init__(
        self, dist: np.ndarray | None = None, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent)
        self.setWindowTitle("Single Ion Distribution")

        self.hist = SingleIonHistogramView()
        self.hist.plot.setMouseEnabled(x=False, y=False)
        self.scatter = SingleIonScatterView()

        self.masses = np.array([])
        self.counts = np.array([])

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

        hist_controls_box = QtWidgets.QGroupBox()
        hist_controls_box_layout = QtWidgets.QFormLayout()
        hist_controls_box_layout.addRow("Range:", layout_range)
        hist_controls_box_layout.addRow(self.check_restrict)
        hist_controls_box.setLayout(hist_controls_box_layout)

        self.max_sigma_difference = QtWidgets.QDoubleSpinBox()
        self.max_sigma_difference.setRange(0.01, 1.0)
        self.max_sigma_difference.setValue(0.1)
        self.max_sigma_difference.setSingleStep(0.01)
        self.max_sigma_difference.valueChanged.connect(self.updateValidParameters)

        self.combo_interp = QtWidgets.QComboBox()
        self.combo_interp.addItems(["Linear", "Moving Average", "Savitzky-Golay"])
        self.combo_interp.currentTextChanged.connect(self.updateScatterInterp)

        controls_box = QtWidgets.QGroupBox()
        controls_layout = QtWidgets.QFormLayout()
        controls_layout.addRow("Dist. from mean:", self.max_sigma_difference)
        controls_layout.addRow("Interpolation:", self.combo_interp)
        controls_box.setLayout(controls_layout)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Reset
            | QtWidgets.QDialogButtonBox.StandardButton.Open
            | QtWidgets.QDialogButtonBox.StandardButton.Apply
            | QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        self.button_box.clicked.connect(self.buttonPressed)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.hist, 0, 0)
        layout.addWidget(hist_controls_box, 1, 0)
        layout.addWidget(self.scatter, 0, 1)
        layout.addWidget(controls_box, 1, 1)
        layout.addWidget(self.button_box, 2, 0, 1, 2)

        layout.setColumnStretch(0, 2)
        layout.setColumnStretch(1, 3)
        self.setLayout(layout)

    def buttonPressed(self, button: QtWidgets.QAbstractButton) -> None:
        sb = self.button_box.standardButton(button)

        if sb == QtWidgets.QDialogButtonBox.StandardButton.Reset:
            self.resetRequested.emit()
        elif sb == QtWidgets.QDialogButtonBox.StandardButton.Apply:
            self.accept()
        elif sb == QtWidgets.QDialogButtonBox.StandardButton.Open:
            self.loadSingleIonData()
        elif sb == QtWidgets.QDialogButtonBox.StandardButton.Close:
            self.reject()

    def updateMinMaxValueRanges(self) -> None:
        if self.minimum_value.hasAcceptableInput():
            min = self.minimum_value.value()
            self.maximum_value.setRange(min, 1e9)
        if self.maximum_value.hasAcceptableInput():
            max = self.maximum_value.value()
            self.minimum_value.setRange(0.0, max)

    def loadSingleIonData(self, path: str | Path | None = None) -> None:
        if path is None:
            path = get_open_spcal_path(self, "Single Ion Data")
            if path is None:
                return
        else:
            path = Path(path)

        if nu.is_nu_directory(path):
            self.masses, self.counts, info = nu.read_nu_directory(path, raw=True)
            self.reported_mu = np.log(info["AverageSingleIonArea"])
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

        # Remove clearly gaussian signals
        zeros = np.count_nonzero(self.counts == 0, axis=0)
        self.masses, self.counts = self.masses[zeros > 0], self.counts[:, zeros > 0]

        max = np.amax(self.counts)
        self.minimum_value.setRange(0.0, max)
        self.maximum_value.setRange(0.0, max)
        self.minimum_value.setValue(0.0)
        self.maximum_value.setValue(max)

        self.updateExtractedParameters()

    def updateExtractedParameters(self) -> None:
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

    def updateValidParameters(self) -> None:
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

        self.updateScatterInterp()
        self.updateHistogram()

    def updateScatterInterp(self) -> None:
        xs, ys = self.interpolatedParameters(
            self.masses[self.valid], self.sigmas[self.valid]
        )
        self.scatter.drawInterpolationLine(xs, ys)

    def interpolatedParameters(
        self, _xs: np.ndarray, _ys: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        interp = self.combo_interp.currentText()
        if interp == "Linear":
            xs, ys = _xs, _ys
        elif interp == "Moving Average":
            xs = np.arange(_xs.min(), _xs.max() + 1, 1.0)
            ys = np.interp(xs, _xs, _ys)

            ma = np.convolve(ys, np.ones(5) / 5.0, mode="valid")
            ys[2:-2] = ma
        elif interp == "Savitzky-Golay":
            t = np.arange(7)
            poly = np.polynomial.Polynomial.fit(t, [0, 0, 0, 1, 0, 0, 0], 2)
            psf = poly(t)

            xs = np.arange(_xs.min(), _xs.max() + 1, 1)
            ys = np.interp(xs, _xs, _ys)

            sg = np.convolve(ys, psf, mode="valid")
            ys[3:-3] = sg
        else:
            raise ValueError(f"unknown interpolation '{interp}'")
        return xs, ys

    def updateHistogram(self) -> None:
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

    def accept(self) -> None:
        # self.distributionSelected.emit(self.counts)
        _, mu = self.interpolatedParameters(
            self.masses[self.valid], self.mus[self.valid]
        )
        mz, sigma = self.interpolatedParameters(
            self.masses[self.valid], self.sigmas[self.valid]
        )
        self.parametersExtracted.emit(np.stack((mz, mu, sigma), axis=1))
        super().accept()


if __name__ == "__main__":
    # options
    # 1. manual input of single SIA shape (like old)
    # 2.
    app = QtWidgets.QApplication()

    win = SingleIonDialog()
    # win.loadSingleIonData("/home/tom/Downloads/Raw data/NT012A/11-43-47 1ppb att")
    win.loadSingleIonData("/home/tom/Downloads/NT032/14-36-31 10 ppb att/")
    win.show()

    app.exec()
