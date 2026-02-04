from PySide6 import QtCore, QtWidgets

from spcal.gui.util import create_action
from spcal.gui.widgets import UnitsWidget, ValueWidget
from spcal.processing.options import SPCalInstrumentOptions
from spcal.siunits import flowrate_units


class SPCalInstrumentOptionsWidget(QtWidgets.QWidget):
    optionsChanged = QtCore.Signal()
    clusterDistanceChanged = QtCore.Signal(float)
    efficiencyDialogRequested = QtCore.Signal()

    def __init__(
        self,
        options: SPCalInstrumentOptions,
        calibration_mode: str,
        cluster_distance: float = 0.03,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        self.cluster_distance = cluster_distance

        settings = QtCore.QSettings()
        sf = int(settings.value("SigFigs", 4))  # type: ignore

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

        self.button_advanced_options = QtWidgets.QPushButton("Advanced Options...")
        self.button_advanced_options.pressed.connect(self.dialogAdvancedOptions)

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
        form_layout.addRow("Trans. Efficiency:", layout_eff)
        form_layout.addRow("Calibration mode:", self.calibration_mode)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(form_layout, 0)
        layout.addWidget(
            self.button_advanced_options, 0, QtCore.Qt.AlignmentFlag.AlignRight
        )
        self.setLayout(layout)

    def instrumentOptions(self) -> SPCalInstrumentOptions:
        return SPCalInstrumentOptions(
            self.uptake.baseValue(),
            self.efficiency.value(),
        )

    def setInstrumentOptions(self, instrument_options: SPCalInstrumentOptions):
        self.blockSignals(True)
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
                    self.uptake.hasAcceptableInput(),
                    self.efficiency.hasAcceptableInput(),
                ]
            )
        elif mode == "Mass Response":
            return True
        else:
            raise ValueError(f"Unknown method {mode}.")

    def dialogAdvancedOptions(self):
        val, ok = QtWidgets.QInputDialog.getDouble(
            self,
            "Advanced Options",
            "Cluster distance:",
            value=self.cluster_distance * 100.0,
            minValue=0.1,
            maxValue=100.0,
            decimals=1,
            step=1.0,
        )
        if ok and val / 100.0 != self.cluster_distance:
            self.cluster_distance = val / 100.0
            self.optionsChanged.emit()


class SPCalInstrumentOptionsDock(QtWidgets.QDockWidget):
    optionsChanged = QtCore.Signal()
    efficiencyDialogRequested = QtCore.Signal()

    def __init__(
        self,
        options: SPCalInstrumentOptions,
        calibration_mode: str,
        cluster_distance: float,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Instrument Options")

        self.options_widget = SPCalInstrumentOptionsWidget(
            options, calibration_mode, cluster_distance
        )
        self.options_widget.optionsChanged.connect(self.optionsChanged)
        self.options_widget.efficiencyDialogRequested.connect(
            self.efficiencyDialogRequested
        )

        self.setWidget(self.options_widget)

    def clusterDistance(self) -> float:
        return self.options_widget.cluster_distance

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

    def reset(self):
        self.blockSignals(True)
        self.options_widget.setInstrumentOptions(SPCalInstrumentOptions(None, None))
        self.setCalibrationMode("efficiency")
        self.blockSignals(False)
        self.optionsChanged.emit()

    def setSignificantFigures(self, num: int):
        for widget in self.options_widget.findChildren(ValueWidget):
            widget.setSigFigs(num)
