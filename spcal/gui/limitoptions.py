from statistics import NormalDist

import h5py
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.dialogs.advancedoptions import AdvancedPoissonDialog
from spcal.gui.graphs.singleion import SingleIonView
from spcal.gui.io import get_open_spcal_path
from spcal.gui.util import create_action
from spcal.gui.widgets import ValueWidget
from spcal.io import nu, tofwerk


class LimitOptions(QtWidgets.QGroupBox):
    limitOptionsChanged = QtCore.Signal()

    def __init__(
        self, name: str, alpha: float = 0.001, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(name, parent=parent)
        sf = int(QtCore.QSettings().value("SigFigs", 4))
        self.alpha = ValueWidget(
            alpha, validator=QtGui.QDoubleValidator(1e-16, 0.5, 9), format=sf
        )
        self.alpha.setToolTip("False positive error rate.")
        self.alpha.valueChanged.connect(self.limitOptionsChanged)

        layout = QtWidgets.QFormLayout()
        layout.addRow("α (Type I):", self.alpha)
        self.setLayout(layout)

    def state(self) -> dict:
        return {"alpha": self.alpha.value()}

    def setState(self, state: dict) -> None:
        self.blockSignals(True)
        if "alpha" in state:
            self.alpha.setValue(state["alpha"])
        self.blockSignals(False)
        self.limitOptionsChanged.emit()


class CompoundPoissonOptions(LimitOptions):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__("Compound Poisson", 1e-6, parent=parent)
        sf = int(QtCore.QSettings().value("SigFigs", 4))

        self.single_ion_dist: np.ndarray | None = None

        self.single_ion_average = ValueWidget(
            1.0, validator=QtGui.QDoubleValidator(1e-6, 1e99, 9), format=sf
        )
        self.single_ion_average.setPlaceholderText("1.0")
        self.single_ion_average.setToolTip(
            "The average single ion area, this is used "
            "to simulate a SIA distribution if none is provided."
        )
        self.single_ion_average.valueChanged.connect(self.limitOptionsChanged)

        self.accumulations = ValueWidget(
            1, validator=QtGui.QIntValidator(1, 100), format=".0f"
        )
        self.accumulations.setToolTip(
            "The number of ion extraction events per acquisition."
        )
        self.accumulations.valueChanged.connect(self.limitOptionsChanged)

        self.action_open_si = create_action(
            "document-open",
            "Open SIA",
            "Load a single-ion distribution from a file.",
            self.loadSingleIonData,
        )
        self.action_show_si = create_action(
            "view-object-histogram-linear",
            "Show SIA",
            "Show the single-ion distribution as a histogram.",
            self.showSingleIonData,
        )
        self.action_clear_si = create_action(
            "edit-clear-history",
            "Clear SIA",
            "Clear single ion distribution.",
            self.clearSingleIon,
        )
        self.toolbar = QtWidgets.QToolBar()
        self.toolbar.addActions(
            [self.action_open_si, self.action_show_si, self.action_clear_si]
        )

        layout_sia = QtWidgets.QHBoxLayout()
        layout_sia.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout_sia.addWidget(self.single_ion_average)
        layout_sia.addWidget(self.toolbar)

        self.layout().addRow("Single Ion Area:", layout_sia)
        self.layout().addRow("Accumulations:", self.accumulations)

    def loadSingleIonData(self) -> None:
        path = get_open_spcal_path(self, "Single Ion Data")
        if path is None:
            return

        if tofwerk.is_tofwerk_file(path):
            data = h5py.File(path)["SingleIon"]["Data"]
        elif nu.is_nu_directory(path):
            _, counts, _ = nu.read_nu_directory(path, cycle=1, raw=True)
            data = nu.single_ion_distribution(counts)
        else:
            with path.open("r") as fp:
                delimiter = "\t"
                skip_rows = 0
                for line in fp.readlines(1024):
                    try:
                        delimiter = next(d for d in ["\t", ";", ",", " "] if d in line)
                        float(line.split(delimiter)[-1])
                        break
                    except (ValueError, StopIteration):
                        pass
                    skip_rows += 1
                count = line.count(delimiter) + 1

            if count == 1:  # raw points from dist
                data = np.genfromtxt(  # type: ignore
                    path,
                    delimiter=delimiter,
                    skip_header=skip_rows,
                    dtype=np.float64,
                    usecols=(0),
                    invalid_raise=False,
                    loose=True,
                )
            elif count == 2:  # hist as bin, count
                bins, counts = np.genfromtxt(  # type: ignore
                    path,
                    delimiter=delimiter,
                    skip_header=skip_rows,
                    dtype=np.float64,
                    usecolss=(0, 1),
                    unpack=True,
                    invalid_raise=False,
                    loose=True,
                )
                data = np.stack((bins, counts), axis=1)
            else:
                raise ValueError(
                    "Text data must consist of either a column of values "
                    "or 2 columns of bins and counts."
                )
        self.setSingleIon(data)

    def showSingleIonData(self) -> QtWidgets.QDialog:
        sia = self.getSingleIon()
        if sia is None:
            return

        view = SingleIonView()
        view.draw(sia)

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Single Ion Distribution")
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(view)
        dlg.setLayout(layout)
        dlg.open()
        return dlg

    def clearSingleIon(self) -> None:
        self.setSingleIon(self.single_ion_average.value())

    def getSingleIon(self) -> float | None | np.ndarray:
        return (
            self.single_ion_dist
            if self.single_ion_dist is not None
            else self.single_ion_average.value() or 1.0
        )

    def setSingleIon(self, sia: float | None | np.ndarray) -> None:
        if sia is None or isinstance(sia, float):
            self.single_ion_average.setValue(sia)
            self.single_ion_dist = None
            self.single_ion_average.setEnabled(True)
        else:
            self.single_ion_dist = sia
            self.single_ion_average.setEnabled(False)
            if sia.ndim == 2:  # hist of bins, counts
                mean = np.average(sia[:, 0], weights=sia[:, 1])
            else:
                mean = np.mean(sia)
            self.single_ion_average.setValue(mean)

    def state(self) -> dict:
        return {
            "alpha": self.alpha.value(),
            "single ion": self.getSingleIon(),
            "accumulations": int(self.accumulations.value() or 1),
        }

    def setState(self, state: dict) -> None:
        self.blockSignals(True)
        if "alpha" in state:
            self.alpha.setValue(state["alpha"])
        if "single ion" in state:
            self.setSingleIon(state["single ion"])
        if "accumulations" in state:
            self.accumulations.setValue(state["accumulations"])
        self.blockSignals(False)
        self.limitOptionsChanged.emit()


class GaussianOptions(LimitOptions):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__("Gaussian", 2.867e-7, parent=parent)  # 5.0 sigma
        self.alpha.valueChanged.connect(self.updateSigma)

        sf = int(QtCore.QSettings().value("SigFigs", 4))
        self.sigma = ValueWidget(
            5.0, validator=QtGui.QDoubleValidator(0.0, 8.0, 4), format=sf
        )
        self.sigma.setToolTip(
            "Type I error rate as number of standard deviations from mean."
        )
        self.sigma.valueChanged.connect(self.updateAlpha)

        self.layout().addRow("σ:", self.sigma)

    def updateAlpha(self) -> None:
        sigma = self.sigma.value()
        if sigma is None:
            return
        alpha = 1.0 - NormalDist().cdf(sigma)
        self.alpha.valueChanged.disconnect(self.updateSigma)
        self.alpha.setValue(float(f"{alpha:.4g}"))
        self.alpha.valueChanged.connect(self.updateSigma)

    def updateSigma(self) -> None:
        alpha = self.alpha.value()
        if alpha is None:
            alpha = 1e-6
        sigma = NormalDist().inv_cdf(1.0 - alpha)
        self.sigma.valueChanged.disconnect(self.updateAlpha)
        self.sigma.setValue(round(sigma, 4))
        self.sigma.valueChanged.connect(self.updateAlpha)


class PoissonOptions(LimitOptions):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__("Poisson", 0.001, parent=parent)
        self.button_advanced = QtWidgets.QPushButton("Advanced Options...")
        self.button_advanced.pressed.connect(self.dialogAdvancedOptions)
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(
            self.button_advanced, 1, QtCore.Qt.AlignmentFlag.AlignRight
        )
        self.layout().addRow(button_layout)

        settings = QtCore.QSettings()
        self.formula = settings.value("Poisson/Formula", "Formula C")
        self.eta = float(settings.value("Poisson/Eta", 1.0))
        self.epsilon = float(settings.value("Poisson/Epsilon", 0.5))
        self.t_sample = float(settings.value("Poisson/Tsample", 1.0))
        self.t_blank = float(settings.value("Poisson/Tblank", 1.0))

        # self.button_advanced.setEnabled(False)

    def dialogAdvancedOptions(self) -> AdvancedPoissonDialog:
        dlg = AdvancedPoissonDialog(
            self.formula,
            self.eta,
            self.epsilon,
            self.t_sample,
            self.t_blank,
            parent=self,
        )
        dlg.optionsSelected.connect(self.setOptions)
        dlg.open()
        return dlg

    def setOptions(self, formula: str, opt1: float, opt2: float) -> None:
        settings = QtCore.QSettings()
        if formula == "Currie":
            self.eta = opt1 or 1.0
            self.epsilon = opt2 or 0.5
            settings.setValue("Poisson/Eta", self.eta)
            settings.setValue("Poisson/Epsilon", self.epsilon)
        else:
            self.t_sample = opt1 or 1.0
            self.t_blank = opt2 or 1.0
            settings.setValue("Poisson/Tsample", self.t_sample)
            settings.setValue("Poisson/Tblank", self.t_blank)

        self.limitOptionsChanged.emit()
