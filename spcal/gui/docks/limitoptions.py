import logging
from statistics import NormalDist

import numpy as np
from PySide6 import QtCore, QtWidgets

from spcal.gui.dialogs.singleion import SingleIonDialog
from spcal.gui.widgets.values import ValueWidget
from spcal.isotope import SPCalIsotopeBase
from spcal.processing.options import SPCalLimitOptions

logger = logging.getLogger(__name__)


class LimitOptionsBaseWidget(QtWidgets.QGroupBox):
    optionsChanged = QtCore.Signal()

    def __init__(
        self, name: str, alpha: float = 1e-7, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent=parent)
        sf = int(QtCore.QSettings().value("SigFigs", 4))  # type: ignore
        self.alpha = ValueWidget(
            alpha,
            min=1e-16,
            max=0.5,
            step=lambda x, s: x * 10**s,
            sigfigs=sf,
            allow_none=False,
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
        alpha: float = 1e-7,
        sigma: float = 0.6,
        single_ion_parameters: np.ndarray | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__("Compound Poisson", alpha, parent=parent)
        sf = int(QtCore.QSettings().value("SigFigs", 4))  # type: ignore

        # array of (..., [mz, mu,sigma] )
        self.single_ion_parameters = single_ion_parameters

        self.lognormal_sigma = ValueWidget(
            sigma, min=1e-9, max=10.0, step=0.05, sigfigs=sf, allow_none=False
        )
        self.lognormal_sigma.setToolTip(
            "Shape parameter for the log-normal approximation of the SIA. "
        )
        self.lognormal_sigma.valueChanged.connect(self.optionsChanged)
        self.lognormal_sigma.setEnabled(self.single_ion_parameters is None)

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
        self.sigma = ValueWidget(
            0.0, min=0.0, max=8.0, step=1.0, sigfigs=sf, allow_none=False
        )
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
        alpha: float = 1e-6,
        eta: float = 1.0,
        epsilon: float = 1.0,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__("Poisson", alpha, parent=parent)

        self.eta = eta
        self.epsilon = epsilon

    def isComplete(self) -> bool:
        return self.alpha.hasAcceptableInput()


class ManualLimitsOptions(QtWidgets.QGroupBox):
    optionsChanged = QtCore.Signal()
    requestManualLimitDialog = QtCore.Signal()

    def __init__(
        self,
        default_manual_limit: float = 100.0,
        manual_limits: dict[SPCalIsotopeBase, float] = {},
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__("Manual", parent)

        self.manual_limits = manual_limits

        sf = int(QtCore.QSettings().value("SigFigs", 4))  # type: ignore

        self.default_manual_limit = ValueWidget(
            default_manual_limit,
            min=0.0,
            max=1e6,
            step=1.0,
            sigfigs=sf,
            allow_none=False,
        )
        self.default_manual_limit.valueChanged.connect(self.optionsChanged)
        self.default_manual_limit.setToolTip(
            "Limit applied to all isotopes without a specific manual limit set."
        )

        self.button_set_limits = QtWidgets.QPushButton("Set Limits...")
        self.button_set_limits.pressed.connect(self.requestManualLimitDialog)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(
            self.button_set_limits, 0, QtCore.Qt.AlignmentFlag.AlignRight
        )

        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow("Default limit:", self.default_manual_limit)
        form_layout.addRow(button_layout)
        self.setLayout(form_layout)

    def setManualLimits(self, limits: dict[SPCalIsotopeBase, float]):
        self.manual_limits = limits
        self.optionsChanged.emit()

    def setSignificantFigures(self, sf: int):
        self.default_manual_limit.setSigFigs(sf)


class SPCalLimitOptionsWidget(QtWidgets.QWidget):
    optionsChanged = QtCore.Signal()
    requestManualLimitsDialog = QtCore.Signal()

    def __init__(
        self,
        limit_options: SPCalLimitOptions,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        self.window_size = QtWidgets.QSpinBox()
        self.window_size.setSingleStep(100)
        self.window_size.setRange(3, 9999999)
        self.window_size.setValue(limit_options.window_size or 1000)
        self.window_size.setToolTip("Size of window for moving thresholds.")
        self.window_size.setEnabled(limit_options.window_size > 0)
        self.check_window = QtWidgets.QCheckBox("Use window")
        self.check_window.setToolTip(
            "Calculate threhold for each point using data from surrounding points."
        )
        self.check_window.setChecked(limit_options.window_size > 0)
        self.check_window.toggled.connect(self.window_size.setEnabled)
        self.check_window.toggled.connect(self.optionsChanged)

        self.check_iterative = QtWidgets.QCheckBox("Iterative")
        self.check_iterative.setToolTip("Iteratively filter on non detections.")
        self.check_iterative.setChecked(limit_options.max_iterations > 1)

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
        self.limit_method.setCurrentText(limit_options.limit_method.title())
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
        self.gaussian = GaussianOptionsWidget(limit_options.gaussian_kws["alpha"])
        self.poisson = PoissonOptionsWidget(limit_options.poisson_kws["alpha"])
        self.compound = CompoundPoissonOptionsWidget(
            limit_options.compound_poisson_kws["alpha"],
            limit_options.compound_poisson_kws["sigma"],
            limit_options.single_ion_parameters,
        )
        self.manual = ManualLimitsOptions(
            limit_options.default_manual_limit, limit_options.manual_limits
        )
        self.manual.requestManualLimitDialog.connect(self.requestManualLimitsDialog)

        self.window_size.valueChanged.connect(self.optionsChanged)
        self.limit_method.currentIndexChanged.connect(self.optionsChanged)
        self.check_iterative.checkStateChanged.connect(self.optionsChanged)
        self.gaussian.optionsChanged.connect(self.optionsChanged)
        self.poisson.optionsChanged.connect(self.optionsChanged)
        self.compound.optionsChanged.connect(self.optionsChanged)
        self.manual.optionsChanged.connect(self.optionsChanged)

        layout_window_size = QtWidgets.QHBoxLayout()
        layout_window_size.addWidget(self.window_size, 1)
        layout_window_size.addWidget(self.check_window, 1)

        layout_method = QtWidgets.QHBoxLayout()
        layout_method.addWidget(self.limit_method, 1)
        layout_method.addWidget(
            self.check_iterative, 0, QtCore.Qt.AlignmentFlag.AlignRight
        )

        self.tab_options = QtWidgets.QTabWidget()
        self.tab_options.addTab(self.gaussian, "Gaussian")
        self.tab_options.addTab(self.poisson, "Poisson")
        self.tab_options.addTab(self.compound, "Compound")
        self.tab_options.addTab(self.manual, "Manual")

        layout = QtWidgets.QVBoxLayout()

        form_layout = QtWidgets.QFormLayout()

        form_layout.addRow("Window size:", layout_window_size)
        form_layout.addRow("Method:", layout_method)

        layout.addLayout(form_layout, 0)
        layout.addWidget(self.tab_options, 1)

        self.setLayout(layout)

    def setLimitOptions(self, options: SPCalLimitOptions):
        self.blockSignals(True)
        self.limit_method.setCurrentText(options.limit_method.capitalize())
        self.check_window.setChecked(options.window_size > 0)
        if options.window_size > 0:
            self.window_size.setValue(options.window_size)
        self.check_iterative.setChecked(options.max_iterations > 1)

        self.gaussian.setParameters(options.gaussian_kws)

        self.poisson.setParameters(options.poisson_kws)

        self.compound.setParameters(options.compound_poisson_kws)
        self.compound.single_ion_parameters = options.single_ion_parameters

        self.manual.default_manual_limit.setValue(options.default_manual_limit)
        self.manual.setManualLimits(options.manual_limits)

        self.blockSignals(False)
        self.optionsChanged.emit()

    def limitOptions(self) -> SPCalLimitOptions:
        options = SPCalLimitOptions(
            limit_method=self.limit_method.currentText().lower(),
            gaussian_kws=self.gaussian.parameters(),
            poisson_kws=self.poisson.parameters(),
            compound_poisson_kws=self.compound.parameters(),
            window_size=int(
                self.window_size.value() or 0 if self.check_window.isChecked() else 0
            ),
            max_iterations=100 if self.check_iterative.isChecked() else 1,
            single_ion_parameters=self.compound.single_ion_parameters,
            default_manual_limit=self.manual.default_manual_limit.value() or 100.0,
            manual_limits=self.manual.manual_limits,
        )
        return options


class SPCalLimitOptionsDock(QtWidgets.QDockWidget):
    optionsChanged = QtCore.Signal()

    def __init__(
        self,
        options: SPCalLimitOptions,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("spcal-limit-options-dock")
        self.setWindowTitle("Limit Options")
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Ignored
        )
        self.options_widget = SPCalLimitOptionsWidget(options)
        self.options_widget.optionsChanged.connect(self.optionsChanged)
        self.setWidget(self.options_widget)

    def limitOptions(self) -> SPCalLimitOptions:
        return self.options_widget.limitOptions()

    def setManualLimits(self, limits: dict[SPCalIsotopeBase, float]):
        self.options_widget.manual.setManualLimits(limits)

    def setSignificantFigures(self, sf: int):
        self.options_widget.poisson.setSignificantFigures(sf)
        self.options_widget.gaussian.setSignificantFigures(sf)
        self.options_widget.compound.setSignificantFigures(sf)
        self.options_widget.manual.setSignificantFigures(sf)

    def setLimitOptions(
        self,
        options: SPCalLimitOptions,
        accumlation_method: str | None = None,
        points_required: int | None = None,
        prominence_required: float | None = None,
    ):
        self.options_widget.setLimitOptions(options)
