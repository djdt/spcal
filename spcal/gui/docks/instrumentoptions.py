from PySide6 import QtCore, QtWidgets

from spcal.gui.util import create_action
from spcal.gui.widgets import UnitsWidget, ValueWidget
from spcal.processing import SPCalInstrumentOptions
from spcal.siunits import time_units


class SPCalInstrumentOptionsDock(QtWidgets.QDockWidget):
    optionsChanged = QtCore.Signal()
    efficiencyDialogRequested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Instrument Options")

        uptake_units = {
            "ml/min": 1e-3 / 60.0,
            "ml/s": 1e-3,
            "L/min": 1.0 / 60.0,
            "L/s": 1.0,
        }

        # load stored options
        settings = QtCore.QSettings()
        sf = int(settings.value("SigFigs", 4))  # type: ignore

        # Actions
        self.action_efficiency = create_action(
            "folder-calculate",
            "Calculate Efficiency",
            "Open a dialog for calculating transport efficiency using the current file and isotope.",
            self.efficiencyDialogRequested,
        )
        self.button_efficiency = QtWidgets.QToolButton()
        self.button_efficiency.setDefaultAction(self.action_efficiency)
        self.button_efficiency.setEnabled(False)

        # Instrument wide options
        self.event_time = UnitsWidget(
            time_units,
            default_unit="ms",
            base_value_min=0.0,
            base_value_max=10.0,
            sigfigs=sf,
        )
        self.event_time.setReadOnly(True)

        self.uptake = UnitsWidget(uptake_units, step=0.1, default_unit="ml/min")
        self.efficiency = ValueWidget(min=0.0, max=1.0, step=0.01, sigfigs=sf)

        self.event_time.setToolTip(
            "ICP-MS time per event, updated from imported files if time column exists."
        )
        self.uptake.setToolTip("ICP-MS sample flow rate.")
        self.efficiency.setToolTip(
            "Transport efficiency. Can be calculated using a reference particle."
        )

        self.calibration_mode = QtWidgets.QComboBox()
        self.calibration_mode.addItems(["Efficiency", "Mass Response"])
        self.calibration_mode.currentTextChanged.connect(self.calibrationModeChanged)
        for i, tooltip in enumerate(
            [
                "Manually enter the transport efficiency.",
                "Calculate the efficiency using a reference particle.",
                "Use the mass response of a reference particle.",
            ]
        ):
            self.calibration_mode.setItemData(
                i, tooltip, QtCore.Qt.ItemDataRole.ToolTipRole
            )
        self.calibration_mode.setToolTip(
            self.calibration_mode.currentData(QtCore.Qt.ItemDataRole.ToolTipRole)
        )
        self.calibration_mode.currentIndexChanged.connect(
            lambda i: self.calibration_mode.setToolTip(
                self.calibration_mode.itemData(i, QtCore.Qt.ItemDataRole.ToolTipRole)
            )
        )

        # Complete Changed
        self.uptake.baseValueChanged.connect(self.optionsChanged)
        self.efficiency.valueChanged.connect(self.optionsChanged)
        self.uptake.baseValueChanged.connect(self.onUptakeChanged)

        layout_eff = QtWidgets.QHBoxLayout()
        layout_eff.addWidget(self.efficiency, 1)
        layout_eff.addWidget(self.button_efficiency, 0, QtCore.Qt.AlignmentFlag.AlignRight)

        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow("Uptake:", self.uptake)
        form_layout.addRow("Event time:", self.event_time)
        form_layout.addRow("Trans. Efficiency:", layout_eff)
        form_layout.addRow("Calibration mode:", self.calibration_mode)

        # layout = QtWidgets.QVBoxLayout()
        # layout.addLayout(form_layout)

        widget = QtWidgets.QWidget()
        widget.setLayout(form_layout)
        self.setWidget(widget)

    def onUptakeChanged(self):
        self.button_efficiency.setEnabled(self.uptake.baseValue() is not None)

    def asInstrumentOptions(self) -> SPCalInstrumentOptions:
        return SPCalInstrumentOptions(
            self.event_time.baseValue(),
            self.uptake.baseValue(),
            self.efficiency.value(),
        )

    def calibrationModeChanged(self, mode: str):
        if mode == "Efficiency":
            self.efficiency.setEnabled(False)
        elif mode == "Mass Response":
            self.efficiency.setEnabled(False)

        self.optionsChanged.emit()

    def isComplete(self) -> bool:
        mode = self.calibration_mode.currentText()
        if mode == "Efficiency":
            return all(
                [
                    self.event_time.hasAcceptableInput(),
                    self.uptake.hasAcceptableInput(),
                    self.efficiency.hasAcceptableInput(),
                ]
            )
        elif mode == "Mass Response":
            return all([self.event_time.hasAcceptableInput()])
        else:
            raise ValueError(f"Unknown method {mode}.")

    def resetInputs(self):
        self.blockSignals(True)
        self.uptake.setValue(None)
        self.event_time.setValue(None)
        self.efficiency.setValue(None)
        self.blockSignals(False)
        self.optionsChanged.emit()

    def setSignificantFigures(self, num: int):
        for widget in self.findChildren(ValueWidget):
            widget.setSigFigs(num)
