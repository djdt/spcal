import logging
from statistics import NormalDist

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.dialogs.advancedoptions import AdvancedPoissonDialog
from spcal.gui.dialogs.singleion import SingleIonDialog
from spcal.gui.widgets import ValueWidget

logger = logging.getLogger(__name__)


class LimitOptions(QtWidgets.QGroupBox):
    limitOptionsChanged = QtCore.Signal()

    def __init__(
        self, name: str, alpha: float = 0.001, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(name, parent=parent)
        sf = int(QtCore.QSettings().value("SigFigs", 4))  # type: ignore
        self.alpha = ValueWidget(
            alpha, validator=QtGui.QDoubleValidator(1e-16, 0.5, 9), format=sf
        )
        self.alpha.setToolTip("False positive error rate.")
        self.alpha.valueChanged.connect(self.limitOptionsChanged)

        layout = QtWidgets.QFormLayout()
        layout.addRow("α (Type I):", self.alpha)
        self.setLayout(layout)

    def layout(self) -> QtWidgets.QFormLayout:
        return super().layout()

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
        sf = int(QtCore.QSettings().value("SigFigs", 4))  # type: ignore

        self.single_ion_parameters = np.array([])  # array of (..., [mz, mu,sigma )

        self.lognormal_sigma = ValueWidget(
            0.47, validator=QtGui.QDoubleValidator(1e-9, 10.0, 9), format=sf
        )
        self.lognormal_sigma.setPlaceholderText("0.45")
        self.lognormal_sigma.setToolTip(
            "Shape parameter for the log-normal approximation of the SIA. "
        )
        self.lognormal_sigma.valueChanged.connect(self.limitOptionsChanged)

        self.method = QtWidgets.QComboBox()
        self.method.addItems(["Approximation", "Lookup Table"])
        self.method.setCurrentText("Lookup Table")
        self.method.currentIndexChanged.connect(self.limitOptionsChanged)

        self.button_sia = QtWidgets.QPushButton("Single Ion Options...")
        self.button_sia.pressed.connect(self.dialogSingleIon)
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.button_sia, 1, QtCore.Qt.AlignmentFlag.AlignRight)

        self.layout().addRow("Method:", self.method)
        self.layout().addRow("SIA σ:", self.lognormal_sigma)
        self.layout().addRow(button_layout)

    def dialogSingleIon(self) -> SingleIonDialog:
        dlg = SingleIonDialog(
            self.single_ion_parameters,
            parent=self,
        )
        dlg.parametersExtracted.connect(self.setSingleIonParameters)
        dlg.resetRequested.connect(self.clearSingleIon)

        dlg.open()
        if self.single_ion_parameters.size == 0:
            dlg.loadSingleIonData()
        return dlg

    def clearSingleIon(self) -> None:
        self.setSingleIonParameters(np.array([]))

    def setSingleIonParameters(self, params: np.ndarray) -> None:
        self.single_ion_parameters = params
        self.lognormal_sigma.setEnabled(self.single_ion_parameters.size == 0)
        self.limitOptionsChanged.emit()

    def state(self) -> dict:
        return {
            "alpha": self.alpha.value(),
            "method": self.method.currentText().lower(),
            "sigma": self.lognormal_sigma.value(),
            "single ion parameters": self.single_ion_parameters,
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
        if "single ion parameters" in state:
            self.single_ion_parameters = state["single ion parameters"]
            self.lognormal_sigma.setEnabled(self.single_ion_parameters.size == 0)

        self.blockSignals(False)
        self.limitOptionsChanged.emit()

    def isComplete(self) -> bool:
        if (
            self.lognormal_sigma.isEnabled()
            and not self.lognormal_sigma.hasAcceptableInput()
        ):
            return False

        return self.alpha.hasAcceptableInput()


class GaussianOptions(LimitOptions):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__("Gaussian", 2.867e-7, parent=parent)  # 5.0 sigma
        self.alpha.valueChanged.connect(self.updateSigma)

        sf = int(QtCore.QSettings().value("SigFigs", 4))  # type: ignore
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
