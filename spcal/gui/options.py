from statistics import NormalDist

from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.widgets import UnitsWidget, ValueWidget
from spcal.siunits import signal_units, time_units

# Todo: add a tool to load an ionic standard and get mean / median signal


class OptionsWidget(QtWidgets.QWidget):
    optionsChanged = QtCore.Signal()
    limitOptionsChanged = QtCore.Signal()
    elementSelected = QtCore.Signal(str, float)
    useManualLimits = QtCore.Signal(bool)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        sf = int(QtCore.QSettings().value("sigfigs", 4))

        uptake_units = {
            "ml/min": 1e-3 / 60.0,
            "ml/s": 1e-3,
            "L/min": 1.0 / 60.0,
            "L/s": 1.0,
        }

        # Instrument wide options
        self.dwelltime = UnitsWidget(
            time_units,
            default_unit="ms",
            validator=QtGui.QDoubleValidator(0.0, 10.0, 10),
            significant_figures=sf,
        )
        self.dwelltime.setReadOnly(True)

        self.uptake = UnitsWidget(
            uptake_units,
            default_unit="ml/min",
            significant_figures=sf,
        )
        self.efficiency = ValueWidget(
            validator=QtGui.QDoubleValidator(0.0, 1.0, 10), significant_figures=sf
        )

        self.dwelltime.setToolTip(
            "ICP-MS dwell-time, updated from imported files if time column exists."
        )
        self.uptake.setToolTip("ICP-MS sample flow rate.")
        self.efficiency.setToolTip(
            "Transport efficiency. Can be calculated using a reference particle."
        )

        self.efficiency_method = QtWidgets.QComboBox()
        self.efficiency_method.addItems(
            ["Manual Input", "Reference Particle", "Mass Response"]
        )
        self.efficiency_method.currentTextChanged.connect(self.efficiencyMethodChanged)
        for i, tooltip in enumerate(
            [
                "Manually enter the transport efficiency.",
                "Calculate the efficiency using a reference particle.",
                "Use the mass response of a reference particle.",
            ]
        ):
            self.efficiency_method.setItemData(i, tooltip, QtCore.Qt.ToolTipRole)

        # Complete Changed
        self.uptake.valueChanged.connect(self.optionsChanged)
        self.efficiency.valueChanged.connect(self.optionsChanged)

        self.inputs = QtWidgets.QGroupBox("Instrument Options")
        self.inputs.setLayout(QtWidgets.QFormLayout())
        self.inputs.layout().addRow("Uptake:", self.uptake)
        self.inputs.layout().addRow("Dwell time:", self.dwelltime)
        self.inputs.layout().addRow("Trans. Efficiency:", self.efficiency)
        self.inputs.layout().addRow("", self.efficiency_method)

        self.window_size = ValueWidget(1000, validator=QtGui.QIntValidator(3, 1000000))
        self.window_size.setToolTip("Size of window for moving thresholds.")
        self.window_size.setEnabled(False)
        self.check_use_window = QtWidgets.QCheckBox("Use window")
        self.check_use_window.toggled.connect(self.window_size.setEnabled)

        layout_window_size = QtWidgets.QHBoxLayout()
        layout_window_size.addWidget(self.window_size, 1)
        layout_window_size.addWidget(self.check_use_window)

        self.method = QtWidgets.QComboBox()
        self.method.addItems(
            [
                "Automatic",
                "Highest",
                "Gaussian",
                "Poisson",
                "Manual Input",
            ]
        )
        self.method.setItemData(
            0,
            "Use Gaussian if signal mean is greater than 50, otherwise Poisson.",
            QtCore.Qt.ToolTipRole,
        )
        self.method.setItemData(
            1, "Use the highest of Gaussian and Poisson.", QtCore.Qt.ToolTipRole
        )
        self.method.setItemData(
            2, "Threshold using the mean and standard deviation.", QtCore.Qt.ToolTipRole
        )
        self.method.setItemData(
            3,
            "Threshold using Formula C from the MARLAP manual.",
            QtCore.Qt.ToolTipRole,
        )
        self.method.setItemData(
            4,
            "Manually defined limits in the sample / reference tab.",
            QtCore.Qt.ToolTipRole,
        )

        self.error_rate_poisson = ValueWidget(
            0.001,
            validator=QtGui.QDoubleValidator(1e-12, 0.5, 9),
            significant_figures=sf,
        )
        self.error_rate_poisson.setPlaceholderText("0.001")
        self.error_rate_poisson.setToolTip(
            "Type I (false positive) error rate for Poisson filters."
        )
        self.error_rate_gaussian = ValueWidget(
            1e-6,
            validator=QtGui.QDoubleValidator(1e-12, 0.5, 9),
            significant_figures=sf,
        )
        self.error_rate_gaussian.setPlaceholderText("1e-6")
        self.error_rate_gaussian.setToolTip(
            "Type I (false positive) error rate for Guassian filters."
        )
        self.error_rate_gaussian.editingFinished.connect(self.updateLabelSigma)

        self.label_sigma_gaussian = QtWidgets.QLabel("4.75 σ")
        self.label_sigma_gaussian.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.label_sigma_gaussian.setIndent(10)

        self.check_iterative = QtWidgets.QCheckBox("Iterative")
        self.check_iterative.setToolTip("Iteratively filter on non detections.")

        self.method.currentTextChanged.connect(self.limitMethodChanged)

        self.window_size.editingFinished.connect(self.limitOptionsChanged)
        self.check_use_window.toggled.connect(self.limitOptionsChanged)
        self.method.currentTextChanged.connect(self.limitOptionsChanged)
        self.error_rate_poisson.editingFinished.connect(self.limitOptionsChanged)
        self.error_rate_gaussian.editingFinished.connect(self.limitOptionsChanged)
        self.check_iterative.toggled.connect(self.limitOptionsChanged)

        layout_method = QtWidgets.QHBoxLayout()
        layout_method.addWidget(self.method)
        layout_method.addWidget(self.check_iterative)

        layout_gaussian = QtWidgets.QGridLayout()
        layout_gaussian.addWidget(self.error_rate_gaussian, 0, 0, 1, 1)
        align = (
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        layout_gaussian.addWidget(self.label_sigma_gaussian, 0, 0, 1, 1, align)

        self.limit_inputs = QtWidgets.QGroupBox("Threshold inputs")
        self.limit_inputs.setLayout(QtWidgets.QFormLayout())
        self.limit_inputs.layout().addRow("Window size:", layout_window_size)
        self.limit_inputs.layout().addRow("Filter method:", layout_method)
        self.limit_inputs.layout().addRow("Poisson α:", self.error_rate_poisson)
        self.limit_inputs.layout().addRow("Gaussian α:", layout_gaussian)

        self.celldiameter = UnitsWidget(
            units={"nm": 1e-9, "μm": 1e-6, "m": 1.0},
            default_unit="μm",
            color_invalid=QtGui.QColor(255, 255, 172),
        )
        self.celldiameter.setToolTip(
            "Sets the mean particle size and calculates intracellular concentrations."
        )

        self.misc_inputs = QtWidgets.QGroupBox("Cell Options")
        self.misc_inputs.setLayout(QtWidgets.QFormLayout())
        self.misc_inputs.layout().addRow("Cell diameter:", self.celldiameter)

        layout_left = QtWidgets.QVBoxLayout()
        layout_left.addWidget(self.limit_inputs)
        layout_left.addWidget(self.misc_inputs)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.inputs)
        layout.addLayout(layout_left)

        self.setLayout(layout)

    def updateLabelSigma(self) -> None:
        alpha = self.error_rate_gaussian.value()
        if alpha is None:
            alpha = 1e-6
        sigma = NormalDist().inv_cdf(1.0 - alpha)
        self.label_sigma_gaussian.setText(f"{sigma:.2f} σ")

    def efficiencyMethodChanged(self, method: str) -> None:
        if method == "Manual Input":
            self.uptake.setEnabled(True)
            self.efficiency.setEnabled(True)
        elif method == "Reference Particle":
            self.uptake.setEnabled(True)
            self.efficiency.setEnabled(False)
        elif method == "Mass Response":
            self.uptake.setEnabled(False)
            self.efficiency.setEnabled(False)

        self.optionsChanged.emit()

    def limitMethodChanged(self, method: str) -> None:
        self.useManualLimits.emit(method == "Manual Input")
        self.check_iterative.setEnabled(method != "Manual Input")
        self.error_rate_poisson.setEnabled(method != "Manual Input")
        self.error_rate_gaussian.setEnabled(method != "Manual Input")

    def isComplete(self) -> bool:
        if self.window_size.isEnabled() and not self.window_size.hasAcceptableInput():
            return False

        method = self.efficiency_method.currentText()
        if method == "Manual Input":
            return all(
                [
                    self.dwelltime.hasAcceptableInput(),
                    self.uptake.hasAcceptableInput(),
                    self.efficiency.hasAcceptableInput(),
                ]
            )
        elif method == "Reference Particle":
            return all(
                [
                    self.dwelltime.hasAcceptableInput(),
                    self.uptake.hasAcceptableInput(),
                ]
            )
        elif method == "Mass Response":
            return all(
                [
                    self.dwelltime.hasAcceptableInput(),
                ]
            )
        else:
            raise ValueError(f"Unknown method {method}.")

    def resetInputs(self) -> None:
        self.blockSignals(True)
        self.uptake.setValue(None)
        self.dwelltime.setValue(None)
        self.efficiency.setValue(None)
        self.celldiameter.setValue(None)
        self.blockSignals(False)
        self.optionsChanged.emit()

    def setSignificantFigures(self, num: int | None = None) -> None:
        if num is None:
            num = int(QtCore.QSettings().value("sigfigs", 4))
        for widget in self.findChildren(ValueWidget):
            widget.setSignificantFigures(num)
