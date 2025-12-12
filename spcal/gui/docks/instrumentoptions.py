from PySide6 import QtCore, QtWidgets

from spcal.gui.util import create_action
from spcal.gui.widgets import UnitsWidget, ValueWidget
from spcal.processing import SPCalInstrumentOptions
from spcal.siunits import flowrate_units, time_units


class SPCalInstrumentOptionsWidget(QtWidgets.QWidget):
    optionsChanged = QtCore.Signal()
    efficiencyDialogRequested = QtCore.Signal()

    def __init__(
        self,
        options: SPCalInstrumentOptions,
        calibration_mode: str,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        settings = QtCore.QSettings()
        sf = int(settings.value("SigFigs", 4))  # type: ignore

        self.event_time = UnitsWidget(
            time_units,
            base_value=options.event_time,
            default_unit="ms",
            base_value_min=0.0,
            base_value_max=10.0,
            sigfigs=sf,
        )
        self.event_time.setReadOnly(True)
        self.event_time.setToolTip(
            "ICP-MS time per event, updated from imported files if time column exists."
        )

        self.uptake = UnitsWidget(
            flowrate_units, base_value=options.uptake, step=0.1, default_unit="ml/min"
        )
        self.uptake.setToolTip("ICP-MS sample flow rate.")

        self.efficiency = ValueWidget(
            options.efficiency, min=0.0, max=1.0, step=0.01, sigfigs=sf
        )
        self.efficiency.setToolTip(
            "Transport efficiency. Can be calculated using a reference particle."
        )

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
        self.calibration_mode = QtWidgets.QComboBox()
        self.calibration_mode.addItems(["Efficiency", "Mass Response"])
        self.calibration_mode.setCurrentText(calibration_mode.capitalize())
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
        layout_eff.addWidget(
            self.button_efficiency, 0, QtCore.Qt.AlignmentFlag.AlignRight
        )

        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow("Uptake:", self.uptake)
        form_layout.addRow("Event time:", self.event_time)
        form_layout.addRow("Trans. Efficiency:", layout_eff)
        form_layout.addRow("Calibration mode:", self.calibration_mode)

        self.setLayout(form_layout)

    def instrumentOptions(self) -> SPCalInstrumentOptions:
        return SPCalInstrumentOptions(
            self.event_time.baseValue(),
            self.uptake.baseValue(),
            self.efficiency.value(),
        )

    def setInstrumentOptions(self, instrument_options: SPCalInstrumentOptions):
        self.blockSignals(True)
        self.event_time.setBaseValue(instrument_options.event_time)
        self.uptake.setBaseValue(instrument_options.uptake)
        self.efficiency.setValue(instrument_options.efficiency)
        self.blockSignals(False)
        self.optionsChanged.emit()

    def onUptakeChanged(self):
        self.button_efficiency.setEnabled(self.uptake.baseValue() is not None)

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


class SPCalInstrumentOptionsDock(QtWidgets.QDockWidget):
    optionsChanged = QtCore.Signal()
    efficiencyDialogRequested = QtCore.Signal()

    def __init__(
        self,
        options: SPCalInstrumentOptions,
        calibration_mode: str,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Instrument Options")

        self.options_widget = SPCalInstrumentOptionsWidget(options, calibration_mode)
        self.options_widget.optionsChanged.connect(self.optionsChanged)
        self.options_widget.efficiencyDialogRequested.connect(
            self.efficiencyDialogRequested
        )

        self.setWidget(self.options_widget)

    def calibrationMode(self) -> str:
        return self.options_widget.calibration_mode.currentText().lower()

    def setCalibrationMode(self, mode: str):
        return self.options_widget.calibration_mode.setCurrentText(mode.capitalize())

    def instrumentOptions(self) -> SPCalInstrumentOptions:
        return self.options_widget.instrumentOptions()

    def setInstrumentOptions(
        self, options: SPCalInstrumentOptions, calibration_mode: str
    ):
        self.options_widget.setInstrumentOptions(options)

    def setEventTime(self, time: float | None):
        self.options_widget.event_time.setBaseValue(time)

    def reset(self):
        self.blockSignals(True)
        self.options_widget.setInstrumentOptions(
            SPCalInstrumentOptions(None, None, None)
        )
        self.setCalibrationMode("efficiency")
        self.blockSignals(False)
        self.optionsChanged.emit()

    def setSignificantFigures(self, num: int):
        for widget in self.options_widget.findChildren(ValueWidget):
            widget.setSigFigs(num)
