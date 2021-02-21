from PySide2 import QtCore, QtGui, QtWidgets

from nanopart.gui.units import UnitsWidget
from nanopart.gui.widgets import ValidColorLineEdit


class OptionsWidget(QtWidgets.QWidget):
    optionsChanged = QtCore.Signal()
    elementSelected = QtCore.Signal(str, float)
    limitOptionsChanged = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget = None):
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
            {"ms": 1e-3, "s": 1.0},
            default_unit="ms",
        )
        self.uptake = UnitsWidget(
            uptake_units,
            default_unit="ml/min",
        )
        self.response = UnitsWidget(
            response_units,
            default_unit="counts/(μg/L)",
        )
        self.efficiency = ValidColorLineEdit()
        self.efficiency.setValidator(QtGui.QDoubleValidator(0.0, 1.0, 4))

        self.dwelltime.setToolTip(
            "ICP-MS dwell-time, updated from imported files if time column exists."
        )
        self.uptake.setToolTip("ICP-MS sample flowrate.")
        self.response.setToolTip("ICP-MS response for ionic standard.")
        self.efficiency.setToolTip(
            "Nebulisation efficiency. Can be calculated using a reference particle."
        )

        # Complete Changed
        self.dwelltime.valueChanged.connect(self.optionsChanged)
        self.uptake.valueChanged.connect(self.optionsChanged)
        self.response.valueChanged.connect(self.optionsChanged)
        self.efficiency.textChanged.connect(self.optionsChanged)

        self.inputs = QtWidgets.QGroupBox("Instrument Options")
        self.inputs.setLayout(QtWidgets.QFormLayout())
        self.inputs.layout().addRow("Uptake:", self.uptake)
        self.inputs.layout().addRow("Dwell time:", self.dwelltime)
        self.inputs.layout().addRow("Response:", self.response)
        self.inputs.layout().addRow("Neb. Efficiency:", self.efficiency)

        self.epsilon = QtWidgets.QLineEdit("0.5")
        self.epsilon.setValidator(QtGui.QDoubleValidator(0.0, 1e2, 2))

        self.sigma = QtWidgets.QLineEdit("3.0")
        self.sigma.setValidator(QtGui.QDoubleValidator(0.0, 1e2, 2))

        self.method = QtWidgets.QComboBox()
        self.method.addItems(["Automatic", "Highest", "Poisson", "Gaussian"])
        self.method.currentTextChanged.connect(self.limitOptionsChanged)

        self.epsilon.setToolTip(
            "Correction factor for low background counts. "
            "Default of 0.5 maintains 0.05 alpha / beta."
        )
        self.sigma.setToolTip("LOD in number of standard deviations from mean.")

        self.epsilon.textChanged.connect(self.limitOptionsChanged)
        self.sigma.textChanged.connect(self.limitOptionsChanged)

        self.limit_inputs = QtWidgets.QGroupBox("Limits inputs")
        self.limit_inputs.setLayout(QtWidgets.QFormLayout())
        self.limit_inputs.layout().addRow("Epsilon:", self.epsilon)
        self.limit_inputs.layout().addRow("Sigma:", self.sigma)
        self.limit_inputs.layout().addRow("LOD method:", self.method)

        self.diameter = UnitsWidget(
            units={"nm": 1e-9, "μm": 1e-6, "m": 1.0},
            default_unit="μm",
            invalid_color=QtGui.QColor(255, 255, 172)
        )

        self.cell_inputs = QtWidgets.QGroupBox("Single Cell Options")
        self.cell_inputs.setLayout(QtWidgets.QFormLayout())
        self.cell_inputs.layout().addRow("Hypothesised diamter:", self.diameter)

        layout_left = QtWidgets.QVBoxLayout()
        layout_left.addWidget(self.limit_inputs)
        layout_left.addWidget(self.cell_inputs)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.inputs)
        layout.addLayout(layout_left)

        self.setLayout(layout)

    def isComplete(self) -> bool:
        return all(
            [
                self.dwelltime.hasAcceptableInput(),
                self.efficiency.hasAcceptableInput(),
                self.response.hasAcceptableInput(),
                self.uptake.hasAcceptableInput(),
            ]
        ) or self.diameter.hasAcceptableInput()

    # def options(self) -> Dict[str, float]:
    #     return {
    #         "dwelltime": self.dwelltime.baseValue(),
    #         "efficiency": float(self.efficiency.text())
    #         if self.efficiency.hasAcceptableInput()
    #         else None,
    #         "response": self.response.baseValue(),
    #         "uptake": self.uptake.baseValue(),
    #     }

    # def setOptions(self, options: Dict[str, float]) -> None:
    #     self.blockSignals(True)
    #     if "dwelltime" in options:
    #         self.dwelltime.setBaseValue(options["dwelltime"])
    #     if "efficiency" in options:
    #         self.efficiency.setText(float(options["efficiency"]))
    #     if "response" in options:
    #         self.response.setBaseValue(options["response"])
    #     if "uptake" in options:
    #         self.uptake.setBaseValue(options["uptake"])
    #     self.blockSignals(False)

    def setEfficiency(self, text: str) -> None:
        self.efficiency.setEnabled(text == "")
        self.efficiency.setText(text)
