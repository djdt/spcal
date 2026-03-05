from PySide6 import QtCore, QtWidgets

from spcal.gui.util import create_action
from spcal.gui.widgets.units import UnitsWidget
from spcal.gui.widgets.values import ValueWidget
from spcal.processing.options import SPCalInstrumentOptions
from spcal.siunits import flowrate_units


class SPCalInstrumentOptionsWidget(QtWidgets.QWidget):
    optionsChanged = QtCore.Signal()
    efficiencyDialogRequested = QtCore.Signal()

    def __init__(
        self,
        options: SPCalInstrumentOptions,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

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
        self.action_efficiency.setEnabled(False)

        self.button_efficiency = QtWidgets.QToolButton()
        self.button_efficiency.setDefaultAction(self.action_efficiency)

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

        self.setLayout(form_layout)

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
        self.action_efficiency.setEnabled(self.uptake.baseValue() is not None)


class SPCalInstrumentOptionsDock(QtWidgets.QDockWidget):
    optionsChanged = QtCore.Signal()
    efficiencyDialogRequested = QtCore.Signal()

    def __init__(
        self,
        options: SPCalInstrumentOptions,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("spcal-instrument-options-dock")
        self.setWindowTitle("Instrument Options")

        self.options_widget = SPCalInstrumentOptionsWidget(options)
        self.options_widget.optionsChanged.connect(self.optionsChanged)
        self.options_widget.efficiencyDialogRequested.connect(
            self.efficiencyDialogRequested
        )

        self.setWidget(self.options_widget)

    def instrumentOptions(self) -> SPCalInstrumentOptions:
        return self.options_widget.instrumentOptions()

    def setInstrumentOptions(self, options: SPCalInstrumentOptions):
        self.options_widget.setInstrumentOptions(options)

    def reset(self):
        self.blockSignals(True)
        self.options_widget.setInstrumentOptions(SPCalInstrumentOptions(None, None))
        self.blockSignals(False)
        self.optionsChanged.emit()

    def setSignificantFigures(self, num: int):
        for widget in self.options_widget.findChildren(ValueWidget):
            widget.setSigFigs(num)
