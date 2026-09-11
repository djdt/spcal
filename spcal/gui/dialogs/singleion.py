from importlib.resources import files
from pathlib import Path

import h5py
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtGui import QValidator

from spcal.calc import sorted_any_close, sparse_gaussian
from spcal.dists.util import extract_compound_poisson_lognormal_parameters
from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.singleion import (
    SingleIonAreaScatterView,
)
from spcal.gui.io import get_open_spcal_path
from spcal.gui.widgets.periodictable import PeriodicTableSelector
from spcal.io import nu, tofwerk
from spcal.isotope import ISOTOPE_TABLE, SPCalIsotope
from spcal.processing.method import SPCalProcessingMethod


def isotopesForMasses(valid_isotopes: list[SPCalIsotope], masses: np.ndarray):
    isotopes = []
    for iso in valid_isotopes:
        if np.any(np.abs(masses - iso.mass) < 0.1):
            isotopes.append(iso)
    return isotopes


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


class SingleIonIsotopesDialog(QtWidgets.QDialog):
    isotopesSelected = QtCore.Signal(list)

    def __init__(
        self,
        enabled: list[SPCalIsotope],
        selected: list[SPCalIsotope],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        # [
        #     iso
        #     for iso in ISOTOPE_TABLE.values()
        #     if iso.composition is not None and iso.composition > min_composition
        # ]

        self.table = PeriodicTableSelector(enabled, selected)

        self.buttons = QtWidgets.QDialogButtonBox

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.Reset
        )
        button_screen = QtWidgets.QPushButton("Screen")
        button_screen.setIcon(QtGui.QIcon.fromTheme("edit-find"))
        self.button_box.addButton(
            button_screen, QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
        )
        self.button_box.clicked.connect(self.onButtonClicked)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.table, 1)
        layout.addWidget(self.button_box, 0)
        self.setLayout(layout)

    def onButtonClicked(self, button: QtWidgets.QAbstractButton):
        sb = self.button_box.standardButton(button)
        if sb == QtWidgets.QDialogButtonBox.StandardButton.Reset:
            self.table.setSelectedIsotopes([])
        elif sb == QtWidgets.QDialogButtonBox.StandardButton.Ok:
            self.accept()
        else:  # Close
            self.reject()

    def completeChanged(self):
        complete = self.isComplete()
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            complete
        )

    def isComplete(self) -> bool:
        return len(self.table.selectedIsotopes()) > 0

    def accept(self):
        self.isotopesSelected.emit(self.table.selectedIsotopes())
        super().accept()


class SingleIonAreaDialog(QtWidgets.QDialog):
    resetRequested = QtCore.Signal()
    parametersExtracted = QtCore.Signal(np.ndarray)

    def __init__(
        self, params: np.ndarray | None = None, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent)
        self.setWindowTitle("Single Ion Distribution")

        """ The SIA guide is calculated from data in https://doi.org/10.1039/d5ja00230c.
            This is from several Nu Vitesse instruments
        """
        self.guide_data = np.load(
            files("spcal.resources").joinpath("sia_shape_guide.npz").open("rb"),
            allow_pickle=False,
        )

        self.scatter = SingleIonAreaScatterView()
        self.scatter.pointClicked.connect(self.onPointClicked)

        self.masses = np.array([])
        self.selected_masses = np.array([])
        self.counts = np.array([])

        self.lams = np.array([])
        self.mus = np.array([])
        self.sigmas = np.array([])
        self.valid = np.array([])

        self.screening_method = SPCalProcessingMethod()
        self.screening_method.limit_options.limit_method = "poisson"
        self.screening_method.limit_options.poisson_kws["alpha"] = 1e-7

        self.required_nonzero_error = QtWidgets.QComboBox()
        self.required_nonzero_error.addItems(["1 %", "2 %", "5 %"])
        self.required_nonzero_error.setItemData(
            0, 8900, QtCore.Qt.ItemDataRole.UserRole
        )
        self.required_nonzero_error.setItemData(
            1, 2100, QtCore.Qt.ItemDataRole.UserRole
        )
        self.required_nonzero_error.setItemData(2, 350, QtCore.Qt.ItemDataRole.UserRole)
        self.required_nonzero_error.setCurrentIndex(0)
        self.required_nonzero_error.currentIndexChanged.connect(
            self.updateValidParameters
        )

        self.check_peaks = QtWidgets.QCheckBox("Remove signals with particles")
        self.check_peaks.setToolTip(
            "Remove signals with values greater than 10 times the non-zero mean."
        )
        self.check_peaks.setChecked(True)
        self.check_peaks.checkStateChanged.connect(self.updateValidParameters)

        self.selected_isotopes = []
        self.button_select_isotopes = QtWidgets.QPushButton("Set isotopes...")
        self.button_select_isotopes.pressed.connect(self.dialogSelectIsotopes)

        self.controls_box = QtWidgets.QGroupBox()
        controls_layout = QtWidgets.QFormLayout()
        # controls_layout.addRow("Dist. from mean:", self.max_sigma_difference)
        controls_layout.addRow("Max σ error:", self.required_nonzero_error)
        controls_layout.addWidget(self.button_select_isotopes)
        controls_layout.addRow(self.check_peaks)
        # controls_layout.addRow("Smoothing:", self.smoothing)
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

    def dialogSelectIsotopes(self) -> QtWidgets.QDialog:
        dlg = SingleIonIsotopesDialog(
            self.enabled_isotopes, self.selected_isotopes, parent=self
        )
        dlg.isotopesSelected.connect(self.setSelectedIsotopes)
        dlg.open()
        return dlg

    def setSelectedIsotopes(self, isotopes: list[SPCalIsotope]):
        self.selected_isotopes = sorted(isotopes, key=lambda iso: iso.mass)
        self.updateValidParameters()

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

        natural_isotopes = [
            iso for iso in ISOTOPE_TABLE.values() if iso.composition is not None
        ]
        natural_isotopes = sorted(natural_isotopes, key=lambda iso: iso.mass)
        natural_masses = np.fromiter(
            (iso.mass for iso in natural_isotopes), dtype=float
        )
        valid_natural = sorted_any_close(natural_masses, self.masses, atol=0.1)

        enabled_isotopes = [
            iso
            for iso in natural_isotopes
            if iso.composition is not None and iso.composition > 0.1
        ]
        enabled_masses = np.fromiter(
            (iso.mass for iso in enabled_isotopes), dtype=float
        )
        valid_enabled = sorted_any_close(enabled_masses, self.masses, atol=0.1)

        self.enabled_isotopes = [
            iso for iso, v in zip(natural_isotopes, valid_natural) if v
        ]
        self.selected_isotopes = [
            iso for iso, v in zip(enabled_isotopes, valid_enabled) if v
        ]

        # trim to valid masses
        enabled_masses = np.fromiter(
            (iso.mass for iso in self.enabled_isotopes), dtype=float
        )
        valid = sorted_any_close(self.masses, enabled_masses, atol=0.1)

        self.masses = self.masses[valid]
        self.counts = self.counts[:, valid]

        self.updateExtractedParameters()
        self.enableControls(True)

    def updateExtractedParameters(self):
        self.scatter.clear()
        # if not self.max_sigma_difference.hasAcceptableInput():
        #     return

        self.lams, self.mus, self.sigmas = (
            extract_compound_poisson_lognormal_parameters(self.counts).T
        )

        self.scatter.drawData(self.masses, self.sigmas)

        self.updateValidParameters()

    def updateGuide(self):
        idx = np.searchsorted(self.masses[self.valid] + 0.5, self.guide_data["mass"])
        valid = np.abs(self.masses[self.valid][idx] - self.guide_data["mass"]) < 0.1
        offset = np.nanmedian(
            self.guide_data["median"][valid] - self.sigmas[self.valid][idx][valid]
        )

        xs = self.guide_data["mass"]
        min = sparse_gaussian(
            xs, self.guide_data["median"] - 1.5 * self.guide_data["iqr"], 3.0
        )
        max = sparse_gaussian(
            xs, self.guide_data["median"] + 1.5 * self.guide_data["iqr"], 3.0
        )

        self.scatter.drawGuide(self.guide_data["mass"], min - offset, max - offset)

    def updateValidParameters(self):
        # most likely invalid
        # outside_sigma_range = np.logical_or(self.sigmas < 0.3, self.sigmas > 0.9)
        # outside_lambda_range = np.logical_or(self.lams < 0.005, self.lams > 10.0)
        #
        idx_error = np.zeros(self.counts.shape[1], int)

        selected_isotope_masses = np.fromiter(
            (iso.mass for iso in self.selected_isotopes), dtype=float
        )
        not_selected = ~sorted_any_close(self.masses, selected_isotope_masses, atol=0.1)
        idx_error[not_selected] = 1

        nonzeros = np.count_nonzero(self.counts, axis=0)
        zeros = self.counts.shape[0] - nonzeros

        required_zeros = self.required_nonzero_error.currentData(
            QtCore.Qt.ItemDataRole.UserRole
        )

        insufficient_zeros = zeros < 150  # approx 5 % error in lambda
        insufficient_nonzeros = nonzeros < required_zeros

        idx_error[insufficient_zeros] = 3
        idx_error[insufficient_nonzeros] = 4

        if self.check_peaks.isChecked():
            nonzero_mean = np.sum(self.counts, axis=0) / nonzeros
            has_peaks = np.count_nonzero(self.counts > nonzero_mean * 10.0, axis=0) > 1
            idx_error[has_peaks] = 2

        self.valid = idx_error == 0

        if self.scatter.points is not None:
            brushes = np.array(
                [
                    QtGui.QBrush(QtCore.Qt.GlobalColor.black),
                    QtGui.QBrush(QtCore.Qt.GlobalColor.white),
                    QtGui.QBrush(QtCore.Qt.GlobalColor.red),
                    QtGui.QBrush(QtCore.Qt.GlobalColor.yellow),
                    QtGui.QBrush(QtCore.Qt.GlobalColor.yellow),
                ]
            )
            symbols = np.array(["o", "o", "x", "t1", "t"])
            self.scatter.points.setBrush(brushes[idx_error])
            self.scatter.points.setSymbol(symbols[idx_error])

        self.updateGuide()
        mean_mu = np.mean(self.mus[self.valid])
        mean_sigma = np.mean(self.sigmas[self.valid])

        self.scatter.plot.setTitle(f"Average: µ={mean_mu:.2f}, σ={mean_sigma:.2f}")

        self.completeChanged()

    #     self.updateScatterInterp()
    #
    # def updateScatterInterp(self):
    #     xs, ys = self.smoothedParameters(
    #         self.masses[self.valid], self.sigmas[self.valid]
    #     )
    #     self.scatter.drawInterpolationLine(xs, ys)

    # def smoothedParameters(
    #     self, xs: np.ndarray, ys: np.ndarray

    #     smoothing = self.smoothing.value()
    #     if smoothing < 3:
    #         return xs, ys
    #     elif smoothing % 2 == 1:
    #         _xs = np.arange(xs[0], xs[-1] + 1.0, 1.0)
    #         _ys = np.interp(_xs, xs, ys)
    #         _ys[smoothing // 2 - 1 : -(smoothing // 2 + 1)] = np.convolve(
    #             _ys, np.ones(smoothing) / smoothing, mode="valid"
    #         )
    #         return xs, np.interp(xs, _xs, _ys)
    #     else:
    #         raise ValueError(f"invalid smoothing window {smoothing}")

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
    # win.loadSingleIonData("/home/tom/Downloads/SIAs/NT032/14-35-55 10 ppb unatt/")
    win.loadSingleIonData("/mnt/storage/TOF/2026 Greenland Ice/13-02-23 mix10ppb/")
    win.show()

    app.exec()
