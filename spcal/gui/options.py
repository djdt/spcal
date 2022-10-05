from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.units import UnitsWidget
from spcal.gui.widgets import ValidColorLineEdit

from typing import Optional


class OptionsWidget(QtWidgets.QWidget):
    optionsChanged = QtCore.Signal()
    elementSelected = QtCore.Signal(str, float)

    def __init__(self, parent: Optional[ QtWidgets.QWidget ] = None):
        super().__init__(parent)

        # density_units = {"g/cm³": 1e-3 * 1e6, "kg/m³": 1.0}
        response_units = {
            "counts/(pg/L)": 1e15,
            "counts/(ng/L)": 1e12,
            "counts/(μg/L)": 1e9,
            "counts/(mg/L)": 1e6,
        }
        uptake_units = {
            "ml/min": 1e-3 / 60.0,
            "ml/s": 1e-3,
            "L/min": 1.0 / 60.0,
            "L/s": 1.0,
        }

        # Instrument wide options
        self.dwelltime = UnitsWidget(
            {"ms": 1e-3, "s": 1.0}, default_unit="ms", validator=(0.0, 10.0, 10)
        )
        self.dwelltime.setReadOnly(True)

        self.uptake = UnitsWidget(
            uptake_units,
            default_unit="ml/min",
        )
        self.response = UnitsWidget(
            response_units,
            default_unit="counts/(μg/L)",
        )
        self.efficiency = ValidColorLineEdit()
        self.efficiency.setValidator(QtGui.QDoubleValidator(0.0, 1.0, 10))

        self.dwelltime.setToolTip(
            "ICP-MS dwell-time, updated from imported files if time column exists."
        )
        self.uptake.setToolTip("ICP-MS sample flowrate.")
        self.response.setToolTip("ICP-MS response for ionic standard.")
        self.efficiency.setToolTip(
            "Transport efficiency. Can be calculated using a reference particle."
        )

        self.efficiency_method = QtWidgets.QComboBox()
        self.efficiency_method.addItems(
            ["Manual Input", "Reference Particle", "Mass Response"]
        )
        self.efficiency_method.currentTextChanged.connect(self.efficiencyMethodChanged)
        self.efficiency_method.setItemData(
            0,
            "Manually enter the transport efficiency.",
            QtCore.Qt.ToolTipRole,
        )
        self.efficiency_method.setItemData(
            1,
            "Calculate the efficiency using a reference particle.",
            QtCore.Qt.ToolTipRole,
        )
        self.efficiency_method.setItemData(
            2,
            "Use the mass response of a reference particle.",
            QtCore.Qt.ToolTipRole,
        )

        # Complete Changed
        # self.dwelltime.valueChanged.connect(self.optionsChanged)
        self.uptake.valueChanged.connect(self.optionsChanged)
        self.response.valueChanged.connect(self.optionsChanged)
        self.efficiency.textChanged.connect(self.optionsChanged)

        self.inputs = QtWidgets.QGroupBox("Instrument Options")
        self.inputs.setLayout(QtWidgets.QFormLayout())
        self.inputs.layout().addRow("Uptake:", self.uptake)
        self.inputs.layout().addRow("Dwell time:", self.dwelltime)
        self.inputs.layout().addRow("Response:", self.response)
        self.inputs.layout().addRow("Trans. Efficiency:", self.efficiency)
        self.inputs.layout().addRow("", self.efficiency_method)

        self.window_size = ValidColorLineEdit("999")
        self.window_size.setValidator(QtGui.QIntValidator(3, 1000000))
        self.window_size.setToolTip("Size of window for moving thresholds.")
        self.window_size.setEnabled(False)
        self.check_use_window = QtWidgets.QCheckBox("Use window")
        self.check_use_window.toggled.connect(self.window_size.setEnabled)

        layout_window_size = QtWidgets.QHBoxLayout()
        layout_window_size.addWidget(self.window_size, 1)
        layout_window_size.addWidget(self.check_use_window)

        self.method = QtWidgets.QComboBox()
        self.method.addItems(
            ["Automatic", "Highest", "Gaussian", "Gaussian Median", "Poisson", "Manual Input"]
        )
        self.method.setItemData(
            0,
            "Use Gaussian if signal mean is greater than 50, otherwise Poisson.",
            QtCore.Qt.ToolTipRole,
        )
        self.method.setItemData(
            1,
            "Use the highest of Gaussian and Poisson.",
            QtCore.Qt.ToolTipRole,
        )
        self.method.setItemData(
            2,
            "Gaussian threshold with LOD of mean + 'simga' * σ.",
            QtCore.Qt.ToolTipRole,
        )
        self.method.setItemData(
            3,
            "Gaussian threshold with LOD of median + 'simga' * σ.",
            QtCore.Qt.ToolTipRole,
        )
        self.method.setItemData(
            4,
            "Regions of Lc with at least one value above Ld.",
            QtCore.Qt.ToolTipRole,
        )
        self.method.setItemData(
            5,
            "Use a manually defined limit for filtering.",
            QtCore.Qt.ToolTipRole,
        )

        # self.poisson_k = QtWidgets.QLineEdit("4.65")
        # self.poisson_k.setPlaceholderText("4.65")
        # self.poisson_k.setValidator(QtGui.QDoubleValidator(0.0, 1e9, 2))
        # self.poisson_k.setToolTip(
        #     "Z?. "
        #     "Default of 0.5 maintains 0.05 alpha / beta."
        # )
        self.error_rate_alpha = QtWidgets.QLineEdit("0.05")
        self.error_rate_alpha.setPlaceholderText("0.05")
        self.error_rate_alpha.setValidator(QtGui.QDoubleValidator(0.001, 0.5, 3))
        self.error_rate_alpha.setToolTip("Type I (false positive) error rate for Poisson filter.")

        self.error_rate_beta = QtWidgets.QLineEdit("0.05")
        self.error_rate_beta.setPlaceholderText("0.05")
        self.error_rate_beta.setValidator(QtGui.QDoubleValidator(0.001, 0.5, 3))
        self.error_rate_beta.setToolTip("Type II (false negative) error rate for Poisson filter.")

        self.sigma = QtWidgets.QLineEdit("5.0")
        self.sigma.setPlaceholderText("5.0")
        self.sigma.setValidator(QtGui.QDoubleValidator(0.0, 1e9, 2))
        self.sigma.setToolTip("LOD in number of standard deviations from mean.")
        self.manual = QtWidgets.QLineEdit("10.0")
        self.manual.setEnabled(False)
        self.manual.setValidator(QtGui.QDoubleValidator(1e-9, 1e9, 2))
        self.manual.setToolTip("Limit used when method is 'Manual Input'.")

        self.method.currentTextChanged.connect(self.limitMethodChanged)

        layout_error_rate = QtWidgets.QHBoxLayout()
        layout_error_rate.addWidget(self.error_rate_alpha, 1)
        layout_error_rate.addWidget(QtWidgets.QLabel(","), 0)
        layout_error_rate.addWidget(self.error_rate_beta, 1)

        self.limit_inputs = QtWidgets.QGroupBox("Threshold inputs")
        self.limit_inputs.setLayout(QtWidgets.QFormLayout())
        self.limit_inputs.layout().addRow("Window size:", layout_window_size)
        self.limit_inputs.layout().addRow("Filter method:", self.method)
        self.limit_inputs.layout().addRow("α, β:", layout_error_rate)
        self.limit_inputs.layout().addRow("Sigma:", self.sigma)
        self.limit_inputs.layout().addRow("Manual limit:", self.manual)

        self.celldiameter = UnitsWidget(
            units={"nm": 1e-9, "μm": 1e-6, "m": 1.0},
            default_unit="μm",
            invalid_color=QtGui.QColor(255, 255, 172),
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

    def efficiencyMethodChanged(self, method: str) -> None:
        if method == "Manual Input":
            self.response.setEnabled(True)
            self.uptake.setEnabled(True)
            self.efficiency.setEnabled(True)
        elif method == "Reference Particle":
            self.response.setEnabled(True)
            self.uptake.setEnabled(True)
            self.efficiency.setEnabled(False)
        elif method == "Mass Response":
            self.response.setEnabled(False)
            self.uptake.setEnabled(False)
            self.efficiency.setEnabled(False)

        self.optionsChanged.emit()

    def limitMethodChanged(self, method: str) -> None:
        if method == "Manual Input":
            self.manual.setEnabled(True)
        else:
            self.manual.setEnabled(False)

    def isComplete(self) -> bool:
        if self.window_size.isEnabled() and not self.window_size.hasAcceptableInput():
            return False

        method = self.efficiency_method.currentText()
        if method == "Manual Input":
            return all(
                [
                    self.dwelltime.hasAcceptableInput(),
                    self.response.hasAcceptableInput(),
                    self.uptake.hasAcceptableInput(),
                    self.efficiency.hasAcceptableInput(),
                ]
            )
        elif method == "Reference Particle":
            return all(
                [
                    self.dwelltime.hasAcceptableInput(),
                    self.response.hasAcceptableInput(),
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
        self.response.setValue(None)
        self.efficiency.setText("")
        self.celldiameter.setValue(None)
        self.blockSignals(False)
        self.optionsChanged.emit()
