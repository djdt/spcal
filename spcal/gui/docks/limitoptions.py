import logging
from statistics import NormalDist

import numpy as np
from PySide6 import QtCore, QtWidgets

from spcal.gui.dialogs.advancedoptions import (
    AdvancedPoissonDialog,
    AdvancedThresholdOptions,
)
from spcal.gui.dialogs.singleion import SingleIonDialog
from spcal.gui.widgets import CollapsableWidget, ValueWidget
from spcal.processing import SPCalLimitOptions

logger = logging.getLogger(__name__)


class LimitOptionsBaseWidget(QtWidgets.QGroupBox):
    optionsChanged = QtCore.Signal()

    def __init__(
        self, name: str, alpha: float = 0.001, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent=parent)
        sf = int(QtCore.QSettings().value("SigFigs", 4))  # type: ignore
        self.alpha = ValueWidget(
            alpha, min=1e-16, max=0.5, step=lambda x, s: x * 10**s, sigfigs=sf
        )
        self.alpha.setToolTip("False positive error rate.")
        self.alpha.valueChanged.connect(self.optionsChanged)

        layout = QtWidgets.QFormLayout()
        layout.addRow("α (Type I):", self.alpha)
        self.setLayout(layout)

    def parameters(self) -> dict:
        return {"alpha": self.alpha.value()}

    def setParameters(self, state: dict):
        self.blockSignals(True)
        if "alpha" in state:
            self.alpha.setValue(state["alpha"])
        self.blockSignals(False)
        self.optionsChanged.emit()

    def isComplete(self) -> bool:
        return False

    def setSignificantFigures(self, sf: int):
        self.alpha.setSigFigs(sf)


class CompoundPoissonOptionsWidget(LimitOptionsBaseWidget):
    def __init__(
        self,
        alpha: float = 1e-6,
        sigma: float = 0.6,
        single_ion_parameters: np.ndarray | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__("Compound Poisson", alpha, parent=parent)
        sf = int(QtCore.QSettings().value("SigFigs", 4))  # type: ignore

        self.single_ion_parameters = (
            single_ion_parameters  # array of (..., [mz, mu,sigma] )
        )

        self.lognormal_sigma = ValueWidget(
            sigma, min=1e-9, max=10.0, step=0.05, sigfigs=sf
        )
        self.lognormal_sigma.setToolTip(
            "Shape parameter for the log-normal approximation of the SIA. "
        )
        self.lognormal_sigma.valueChanged.connect(self.optionsChanged)

        self.button_sia = QtWidgets.QPushButton("Single Ion Options...")
        self.button_sia.pressed.connect(self.dialogSingleIon)
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.button_sia, 1, QtCore.Qt.AlignmentFlag.AlignRight)

        layout = self.layout()
        assert isinstance(layout, QtWidgets.QFormLayout)
        layout.addRow("SIA σ:", self.lognormal_sigma)
        layout.addRow(button_layout)

    def dialogSingleIon(self) -> SingleIonDialog:
        dlg = SingleIonDialog(
            self.single_ion_parameters,
            parent=self,
        )
        dlg.parametersExtracted.connect(self.setSingleIonParameters)
        dlg.resetRequested.connect(self.clearSingleIon)

        dlg.open()
        if self.single_ion_parameters is None:
            dlg.loadSingleIonData()
        return dlg

    def clearSingleIon(self):
        self.setSingleIonParameters(None)

    def setSingleIonParameters(self, params: np.ndarray | None):
        self.single_ion_parameters = params
        self.lognormal_sigma.setEnabled(self.single_ion_parameters is None)
        self.optionsChanged.emit()

    def parameters(self) -> dict:
        return {
            "alpha": self.alpha.value(),
            "sigma": self.lognormal_sigma.value(),
            "single ion parameters": self.single_ion_parameters,
        }

    def setParameters(self, state: dict):
        self.blockSignals(True)
        if "alpha" in state:
            self.alpha.setValue(state["alpha"])
        if "sigma" in state:
            self.lognormal_sigma.setValue(state["sigma"])
        if "single ion parameters" in state:
            self.single_ion_parameters = state["single ion parameters"]
            self.lognormal_sigma.setEnabled(self.single_ion_parameters is None)

        self.blockSignals(False)
        self.optionsChanged.emit()

    def isComplete(self) -> bool:
        if (
            self.lognormal_sigma.isEnabled()
            and not self.lognormal_sigma.hasAcceptableInput()
        ):
            return False

        return self.alpha.hasAcceptableInput()

    def setSignificantFigures(self, sf: int):
        super().setSignificantFigures(sf)
        self.lognormal_sigma.setSigFigs(sf)


class GaussianOptionsWidget(LimitOptionsBaseWidget):
    def __init__(
        self,
        alpha: float = 2.867e-7,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__("Gaussian", alpha, parent=parent)  # 5.0 sigma
        self.alpha.valueChanged.connect(self.updateSigma)

        sf = int(QtCore.QSettings().value("SigFigs", 4))  # type: ignore
        self.sigma = ValueWidget(0.0, min=0.0, max=8.0, step=1.0, sigfigs=sf)
        self.sigma.setToolTip(
            "Type I error rate as number of standard deviations from mean."
        )
        self.sigma.valueChanged.connect(self.updateAlpha)
        self.updateSigma()

        layout = self.layout()
        assert isinstance(layout, QtWidgets.QFormLayout)
        layout.addRow("σ:", self.sigma)

    def updateAlpha(self):
        sigma = self.sigma.value()
        if sigma is None:
            return
        alpha = 1.0 - NormalDist().cdf(sigma)
        self.alpha.valueChanged.disconnect(self.updateSigma)
        self.alpha.setValue(float(f"{alpha:.4g}"))
        self.alpha.valueChanged.connect(self.updateSigma)

    def updateSigma(self):
        alpha = self.alpha.value()
        if alpha is None:
            alpha = 1e-6
        sigma = NormalDist().inv_cdf(1.0 - alpha)
        self.sigma.valueChanged.disconnect(self.updateAlpha)
        self.sigma.setValue(round(sigma, 4))
        self.sigma.valueChanged.connect(self.updateAlpha)

    def isComplete(self) -> bool:
        return self.alpha.hasAcceptableInput()

    def setSignificantFigures(self, sf: int):
        super().setSignificantFigures(sf)
        self.sigma.setSigFigs(sf)


class PoissonOptionsWidget(LimitOptionsBaseWidget):
    def __init__(
        self,
        alpha: float = 0.001,
        function: str = "formula c",
        eta: float = 2.0,
        epsilon: float = 0.5,
        t_sample: float = 1.0,
        t_blank: float = 1.0,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__("Poisson", alpha, parent=parent)
        self.button_advanced = QtWidgets.QPushButton("Advanced Options...")
        self.button_advanced.pressed.connect(self.dialogAdvancedOptions)
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(
            self.button_advanced, 1, QtCore.Qt.AlignmentFlag.AlignRight
        )
        layout = self.layout()
        assert isinstance(layout, QtWidgets.QFormLayout)
        layout.addRow(button_layout)

        self.function = function
        self.eta = eta
        self.epsilon = epsilon
        self.t_sample = t_sample
        self.t_blank = t_blank

    def parameters(self) -> dict:
        return {
            "alpha": self.alpha.value(),
            "beta": 0.05,
            "t_sample": self.t_sample,
            "t_blank": self.t_blank,
            "eta": self.eta,
            "epsilon": self.epsilon,
            "function": self.function,
        }

    def setParameters(self, state: dict):
        self.blockSignals(True)
        if "alpha" in state:
            self.alpha.setValue(state["alpha"])
        if "formula" in state:
            self.function = state["formula"].title()
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
        self.optionsChanged.emit()

    def dialogAdvancedOptions(self) -> AdvancedPoissonDialog:
        dlg = AdvancedPoissonDialog(
            self.function.capitalize(),
            self.eta,
            self.epsilon,
            self.t_sample,
            self.t_blank,
            parent=self,
        )
        dlg.optionsSelected.connect(self.setOptions)
        dlg.open()
        return dlg

    def setOptions(self, formula: str, opt1: float, opt2: float):
        self.function = formula.lower()
        if formula == "currie":
            self.eta = opt1 or 2.0
            self.epsilon = opt2 or 0.5
        else:
            self.t_sample = opt1 or 1.0
            self.t_blank = opt2 or 1.0

        self.optionsChanged.emit()

    def isComplete(self) -> bool:
        return self.alpha.hasAcceptableInput()


class SPCalLimitOptionsWidget(QtWidgets.QWidget):
    optionsChanged = QtCore.Signal()

    def __init__(
        self,
        limit_options: SPCalLimitOptions,
        accumulation_method: str,
        points_required: int,
        prominence_required: float,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        self.limit_accumulation = accumulation_method
        self.points_required = points_required
        self.prominence_required = prominence_required

        self.window_size = QtWidgets.QSpinBox()
        self.window_size.setSingleStep(100)
        self.window_size.setRange(3, 9999999)
        self.window_size.setValue(1000)
        self.window_size.setToolTip("Size of window for moving thresholds.")
        self.window_size.setEnabled(False)
        self.check_window = QtWidgets.QCheckBox("Use window")
        self.check_window.setToolTip(
            "Calculate threhold for each point using data from surrounding points."
        )
        self.check_window.toggled.connect(self.window_size.setEnabled)
        self.check_window.toggled.connect(self.optionsChanged)

        self.check_iterative = QtWidgets.QCheckBox("Iterative")
        self.check_iterative.setToolTip("Iteratively filter on non detections.")

        self.limit_method = QtWidgets.QComboBox()
        self.limit_method.addItems(
            [
                "Automatic",
                "Highest",
                "Compound Poisson",
                "Gaussian",
                "Poisson",
                "Manual Input",
            ]
        )
        self.limit_method.setCurrentText(limit_options.limit_method.capitalize())
        for i, tooltip in enumerate(
            [
                "Automatically determine the best method.",
                "Use the highest of Gaussian and Poisson.",
                "Estimate ToF limits using a compound distribution based on the "
                "number of accumulations and the single ion distribution.",
                "Threshold using the mean and standard deviation.",
                "Threshold using poisson statistics, see the MARLAP manual.",
                "Manually define limits in the sample and reference tabs.",
            ]
        ):
            self.limit_method.setItemData(
                i, tooltip, QtCore.Qt.ItemDataRole.ToolTipRole
            )

        self.limit_method.setToolTip(
            self.limit_method.currentData(QtCore.Qt.ItemDataRole.ToolTipRole)
        )
        self.limit_method.currentIndexChanged.connect(
            lambda i: self.limit_method.setToolTip(
                self.limit_method.itemData(i, QtCore.Qt.ItemDataRole.ToolTipRole)
            )
        )

        self.button_advanced_options = QtWidgets.QPushButton("Advanced Options...")
        self.button_advanced_options.pressed.connect(self.dialogAdvancedOptions)

        self.gaussian = GaussianOptionsWidget(
            limit_options.gaussian_kws["alpha"]
        )
        self.poisson = PoissonOptionsWidget(
            limit_options.poisson_kws["alpha"],
            limit_options.poisson_kws["function"],
            limit_options.poisson_kws.get("eta",2.0),
            limit_options.poisson_kws.get("epsilon", 0.5),
            limit_options.poisson_kws.get("t_sample",1.0),
            limit_options.poisson_kws.get("t_blank",1.0),
        )
        self.compound = CompoundPoissonOptionsWidget(
            limit_options.compound_poisson_kws["alpha"],
            limit_options.compound_poisson_kws["sigma"],
            limit_options.single_ion_parameters,
        )

        self.window_size.valueChanged.connect(self.optionsChanged)
        self.limit_method.currentIndexChanged.connect(self.optionsChanged)
        self.check_iterative.checkStateChanged.connect(self.optionsChanged)
        self.gaussian.optionsChanged.connect(self.optionsChanged)
        self.poisson.optionsChanged.connect(self.optionsChanged)
        self.compound.optionsChanged.connect(self.optionsChanged)

        layout_window_size = QtWidgets.QHBoxLayout()
        layout_window_size.addWidget(self.window_size, 1)
        layout_window_size.addWidget(self.check_window, 1)

        layout_method = QtWidgets.QHBoxLayout()
        layout_method.addWidget(self.limit_method, 1)
        layout_method.addWidget(
            self.check_iterative, 0, QtCore.Qt.AlignmentFlag.AlignRight
        )

        gaussian_collapse = CollapsableWidget("Gaussian Limit Options")
        gaussian_collapse.setWidget(self.gaussian)
        poisson_collapse = CollapsableWidget("Poisson Limit Options")
        poisson_collapse.setWidget(self.poisson)
        compound_collapse = CollapsableWidget("Compound Poisson Limit Options")
        compound_collapse.setWidget(self.compound)

        layout = QtWidgets.QVBoxLayout()

        form_layout = QtWidgets.QFormLayout()

        form_layout.addRow("Window size:", layout_window_size)
        form_layout.addRow("Method:", layout_method)

        layout.addLayout(form_layout, 0)

        layout.addWidget(gaussian_collapse, 0)
        layout.addWidget(poisson_collapse, 0)
        layout.addWidget(compound_collapse, 0)

        layout.addWidget(
            self.button_advanced_options, 0, QtCore.Qt.AlignmentFlag.AlignRight
        )
        self.setLayout(layout)

    def setLimitOptions(self, options: SPCalLimitOptions):
        self.blockSignals(True)
        self.limit_method.setCurrentText(options.limit_method.capitalize())
        self.gaussian.setParameters(options.gaussian_kws)
        self.poisson.setParameters(options.poisson_kws)
        self.compound.setParameters(options.compound_poisson_kws)
        self.compound.single_ion_parameters = options.single_ion_parameters
        self.check_window.setChecked(options.window_size > 0)
        if options.window_size > 0:
            self.window_size.setValue(options.window_size)
        self.check_iterative.setChecked(options.max_iterations > 1)
        self.blockSignals(False)
        self.optionsChanged.emit()

    def limitOptions(self) -> SPCalLimitOptions:
        return SPCalLimitOptions(
            limit_method=self.limit_method.currentText().lower(),
            gaussian_kws=self.gaussian.parameters(),
            poisson_kws=self.poisson.parameters(),
            compound_poisson_kws=self.compound.parameters(),
            window_size=int(
                self.window_size.value() or 0 if self.check_window.isChecked() else 0
            ),
            max_iterations=100 if self.check_iterative.isChecked() else 1,
            single_ion_parameters=self.compound.single_ion_parameters,
        )

    def dialogAdvancedOptions(self):
        dlg = AdvancedThresholdOptions(
            self.limit_accumulation.title(),
            self.points_required,
            self.prominence_required,
            parent=self,
        )
        dlg.optionsSelected.connect(self.setAdvancedOptions)
        dlg.open()

    def advancedOptions(self) -> tuple[str, int, float]:
        return (self.limit_accumulation, self.points_required, self.prominence_required)

    def setAdvancedOptions(
        self, accumlation_method: str, points_required: int, prominence_required: float
    ):
        self.limit_accumulation = accumlation_method
        self.points_required = points_required
        self.prominence_required = prominence_required

        settings = QtCore.QSettings()
        settings.setValue("Threshold/AccumulationMethod", accumlation_method)
        settings.setValue("Threshold/PointsRequired", points_required)
        settings.setValue("Threshold/ProminenceRequired", prominence_required)

        self.optionsChanged.emit()


class SPCalLimitOptionsDock(QtWidgets.QDockWidget):
    optionsChanged = QtCore.Signal()

    def __init__(
        self,
        options: SPCalLimitOptions,
        accumulation_method: str,
        points_required: int,
        prominence_required: float,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Limit Options")
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Ignored
        )
        self.options_widget = SPCalLimitOptionsWidget(
            options, accumulation_method, points_required, prominence_required
        )
        self.options_widget.optionsChanged.connect(self.optionsChanged)
        self.setWidget(self.options_widget)

    def limitOptions(self) -> SPCalLimitOptions:
        return self.options_widget.limitOptions()

    def accumulationMethod(self) -> str:
        return self.options_widget.limit_accumulation

    def pointsRequired(self) -> int:
        return self.options_widget.points_required

    def prominenceRequired(self) -> float:
        return self.options_widget.prominence_required

    def reset(self):
        settings = QtCore.QSettings()
        self.blockSignals(True)
        self.options_widget.setLimitOptions(SPCalLimitOptions())
        self.options_widget.setAdvancedOptions(
            str(settings.value("Threshold/AccumulationMethod", "signal mean")),
            int(settings.value("Threshold/PointsRequired", 1)),  # type: ignore
            float(settings.value("Threshold/ProminenceRequired", 0.2)),  # type: ignore
        )
        self.blockSignals(False)
        self.optionsChanged.emit()

    def setSignificantFigures(self, sf: int):
        self.options_widget.poisson.setSignificantFigures(sf)
        self.options_widget.gaussian.setSignificantFigures(sf)
        self.options_widget.compound.setSignificantFigures(sf)
