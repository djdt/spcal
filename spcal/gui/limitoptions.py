import logging
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

logger = logging.getLogger(__name__)


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

    def isComplete(self) -> bool:
        return False


class CompoundPoissonOptions(LimitOptions):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__("Compound Poisson", 1e-6, parent=parent)
        sf = int(QtCore.QSettings().value("SigFigs", 4))

        self.single_ion_dist: np.ndarray | None = None

        self.lognormal_sigma = ValueWidget(
            0.47, validator=QtGui.QDoubleValidator(1e-9, 10.0, 9), format=sf
        )
        self.lognormal_sigma.setPlaceholderText("0.45")
        self.lognormal_sigma.setToolTip(
            "Shape parameter for the log-normal approximation of the SIA. "
            "Used if a SIA distribution is not passed."
        )
        self.lognormal_sigma.valueChanged.connect(self.limitOptionsChanged)

        self.extract_sigma = QtWidgets.QCheckBox("Automatic")
        self.extract_sigma.stateChanged.connect(self.limitOptionsChanged)

        self.method = QtWidgets.QComboBox()
        self.method.addItems(["Approximation", "Lookup Table", "Simulation"])
        self.method.setCurrentText("Lookup Table")
        self.method.currentIndexChanged.connect(self.limitOptionsChanged)
        # diabale simulation
        item = self.method.model().item(2)
        item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEnabled)

        self.label_sis = QtWidgets.QLabel("Not loaded.")

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

        layout_sigma = QtWidgets.QHBoxLayout()
        layout_sigma.addWidget(self.lognormal_sigma)
        layout_sigma.addWidget(self.extract_sigma)

        layout_sia = QtWidgets.QHBoxLayout()
        layout_sia.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout_sia.addWidget(self.label_sis, 1)
        layout_sia.addWidget(self.toolbar, 0)

        self.layout().addRow("Method:", self.method)
        self.layout().addRow("SIA σ:", layout_sigma)
        self.layout().addRow("SIA Dist:", layout_sia)

    def loadSingleIonData(self) -> None:
        path = get_open_spcal_path(self, "Single Ion Data")
        if path is None:
            return

        try:
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
                            delimiter = next(
                                d for d in ["\t", ";", ",", " "] if d in line
                            )
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
                        "Text data must consist of either one column of distribution "
                        "values or 2 columns of bins and counts."
                    )
            self.setSingleIon(data)
        except ValueError as e:
            logger.exception(e)
            QtWidgets.QMessageBox.warning(self, "Unable to load SIA data", str(e))

    def showSingleIonData(self) -> QtWidgets.QDialog:
        if self.single_ion_dist is None:
            return

        view = SingleIonView()
        view.draw(self.single_ion_dist)

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Single Ion Distribution")
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(view)
        dlg.setLayout(layout)
        dlg.open()
        return dlg

    def clearSingleIon(self) -> None:
        self.setSingleIon(None)

    def setSingleIon(self, sia: None | np.ndarray) -> None:
        if np.any(self.single_ion_dist != sia):
            self.single_ion_dist = sia
            self.limitOptionsChanged.emit()
            item = self.method.model().item(2)
            if self.single_ion_dist is None:
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEnabled)
                self.label_sis.setText("Not loaded.")
            else:
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEnabled)
                self.label_sis.setText(f"{self.single_ion_dist.size} points")

    def state(self) -> dict:
        return {
            "alpha": self.alpha.value(),
            "method": self.method.currentText().lower(),
            "sigma": self.lognormal_sigma.value(),
            "extract sigma": self.extract_sigma.isChecked(),
            "single ion": (
                self.single_ion_dist
                if self.single_ion_dist is not None
                else np.array([])
            ),  # No None
            "simulate": self.method.currentIndex() == 1,
        }

    def setState(self, state: dict) -> None:
        self.blockSignals(True)
        if "alpha" in state:
            self.alpha.setValue(state["alpha"])
        if "method" in state:
            text = " ".join(x.capitalize() for x in state["method"].split(" "))
            self.method.setCurrentText(text)
        if "sigma" in state:
            self.lognormal_sigma.setValue(state["sigma"])
        if "single ion" in state:
            self.setSingleIon(
                state["single ion"] if state["single ion"].size > 0 else None
            )
        if "simulate" in state:
            if state["simulate"]:
                self.method.setCurrentIndex(1)
            else:
                self.method.setCurrentIndex(0)
        self.blockSignals(False)
        self.limitOptionsChanged.emit()

    def isComplete(self) -> bool:
        if self.method.currentText() == "Simulation":
            if self.single_ion_dist is None:
                return False
        else:
            if not self.lognormal_sigma.hasAcceptableInput():
                return False

        return self.alpha.hasAcceptableInput()


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

    def isComplete(self) -> bool:
        return self.alpha.hasAcceptableInput()


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

        self.formula = "Formula C"
        self.eta = 2.0
        self.epsilon = 0.5
        self.t_sample = 1.0
        self.t_blank = 1.0
        # settings = QtCore.QSettings()
        # self.formula = settings.value("Poisson/Formula", "Formula C")
        # self.eta = float(settings.value("Poisson/Eta", 2.0))
        # self.epsilon = float(settings.value("Poisson/Epsilon", 0.5))
        # self.t_sample = float(settings.value("Poisson/Tsample", 1.0))
        # self.t_blank = float(settings.value("Poisson/Tblank", 1.0))

    def state(self) -> dict:
        if self.formula == "Currie":
            params = {"eta": self.eta, "epsilon": self.epsilon}
        else:
            params = {"t_sample": self.t_sample, "t_blank": self.t_blank}
        return {"alpha": self.alpha.value(), "formula": self.formula, "params": params}

    def setState(self, state: dict) -> None:
        self.blockSignals(True)
        if "alpha" in state:
            self.alpha.setValue(state["alpha"])
        if "formula" in state:
            self.formula = state["formula"]
        if "params" in state:
            if "eta" in state["params"]:
                self.eta = state["params"]["eta"]
            if "epsilon" in state["params"]:
                self.epsilon = state["params"]["epsilon"]
            if "t_sample" in state["params"]:
                self.t_sample = state["params"]["t_sample"]
            if "t_blank" in state["params"]:
                self.t_blank = state["params"]["t_blank"]
        self.blockSignals(False)
        self.limitOptionsChanged.emit()

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
        self.formula = formula
        # settings = QtCore.QSettings()
        # settings.value("Poisson/Formula", formula)
        if formula == "Currie":
            self.eta = opt1 or 2.0
            self.epsilon = opt2 or 0.5
            # settings.setValue("Poisson/Eta", self.eta)
            # settings.setValue("Poisson/Epsilon", self.epsilon)
        else:
            self.t_sample = opt1 or 1.0
            self.t_blank = opt2 or 1.0
            # settings.setValue("Poisson/Tsample", self.t_sample)
            # settings.setValue("Poisson/Tblank", self.t_blank)

        self.limitOptionsChanged.emit()

    def isComplete(self) -> bool:
        return self.alpha.hasAcceptableInput()
