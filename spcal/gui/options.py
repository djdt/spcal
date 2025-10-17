from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.dialogs.advancedoptions import AdvancedThresholdOptions
from spcal.gui.docks.limitoptions import (
    CompoundPoissonOptions,
    GaussianOptions,
    PoissonOptions,
)
from spcal.gui.widgets import UnitsWidget, ValueWidget
from spcal.processing import SPCalInstrumentOptions, SPCalLimitOptions
from spcal.siunits import time_units


class OptionsWidget(QtWidgets.QWidget):
    optionsChanged = QtCore.Signal()
    limitOptionsChanged = QtCore.Signal()
    useManualLimits = QtCore.Signal(bool)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        uptake_units = {
            "ml/min": 1e-3 / 60.0,
            "ml/s": 1e-3,
            "L/min": 1.0 / 60.0,
            "L/s": 1.0,
        }

        # load stored options
        settings = QtCore.QSettings()
        self.limit_accumulation = str(
            settings.value("Threshold/AccumulationMethod", "Signal Mean")
        )
        self.points_required = int(settings.value("Threshold/PointsRequired", 1))  # type: ignore

        self.prominence_required = float(
            settings.value("Threshold/ProminenceRequired", 0.2)  # type: ignore
        )  # type: ignore

        sf = int(settings.value("SigFigs", 4))  # type: ignore

        # Instrument wide options
        self.event_time = UnitsWidget(
            time_units,
            default_unit="ms",
            validator=QtGui.QDoubleValidator(0.0, 10.0, 10),
            format=sf,
        )
        self.event_time.setReadOnly(True)

        self.uptake = UnitsWidget(
            uptake_units,
            default_unit="ml/min",
            format=sf,
        )
        self.efficiency = ValueWidget(
            validator=QtGui.QDoubleValidator(0.0, 1.0, 10), format=sf
        )

        self.event_time.setToolTip(
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
            self.efficiency_method.setItemData(
                i, tooltip, QtCore.Qt.ItemDataRole.ToolTipRole
            )
        self.efficiency_method.setToolTip(
            self.efficiency_method.currentData(QtCore.Qt.ItemDataRole.ToolTipRole)
        )
        self.efficiency_method.currentIndexChanged.connect(
            lambda i: self.efficiency_method.setToolTip(
                self.efficiency_method.itemData(i, QtCore.Qt.ItemDataRole.ToolTipRole)
            )
        )

        # Complete Changed
        self.uptake.baseValueChanged.connect(self.optionsChanged)
        self.efficiency.valueChanged.connect(self.optionsChanged)

        self.inputs = QtWidgets.QGroupBox("Instrument Options")
        input_layout = QtWidgets.QFormLayout()
        input_layout.addRow("Uptake:", self.uptake)
        input_layout.addRow("Event time:", self.event_time)
        input_layout.addRow("Trans. Efficiency:", self.efficiency)
        input_layout.addRow("", self.efficiency_method)
        self.inputs.setLayout(input_layout)

        self.window_size = ValueWidget(
            1000, format=("f", 0), validator=QtGui.QIntValidator(3, 1000000)
        )
        self.window_size.setEditFormat(0, format="f")
        self.window_size.setToolTip("Size of window for moving thresholds.")
        self.window_size.setEnabled(False)
        self.check_window = QtWidgets.QCheckBox("Use window")
        self.check_window.setToolTip(
            "Calculate threhold for each point using data from surrounding points."
        )
        self.check_window.toggled.connect(self.window_size.setEnabled)

        layout_window_size = QtWidgets.QHBoxLayout()
        layout_window_size.addWidget(self.window_size, 1)
        layout_window_size.addWidget(self.check_window, 1)

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
        self.limit_method.setItemData(
            0,
            "Automatically determine the best method.",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.limit_method.setItemData(
            1,
            "Use the highest of Gaussian and Poisson.",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.limit_method.setItemData(
            2,
            "Estimate ToF limits using a compound distribution based on the "
            "number of accumulations and the single ion distribution..",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.limit_method.setItemData(
            3,
            "Threshold using the mean and standard deviation.",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.limit_method.setItemData(
            4,
            "Threshold using Formula C from the MARLAP manual.",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.limit_method.setItemData(
            5,
            "Manually define limits in the sample and reference tabs.",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.limit_method.setToolTip(
            self.limit_method.currentData(QtCore.Qt.ItemDataRole.ToolTipRole)
        )
        self.limit_method.currentIndexChanged.connect(
            lambda i: self.limit_method.setToolTip(
                self.limit_method.itemData(i, QtCore.Qt.ItemDataRole.ToolTipRole)
            )
        )

        self.check_iterative = QtWidgets.QCheckBox("Iterative")
        self.check_iterative.setToolTip("Iteratively filter on non detections.")

        self.button_advanced_options = QtWidgets.QPushButton("Advanced Options...")
        self.button_advanced_options.pressed.connect(self.dialogAdvancedOptions)

        self.limit_method.currentTextChanged.connect(self.limitMethodChanged)

        self.window_size.editingFinished.connect(self.limitOptionsChanged)
        self.check_window.toggled.connect(self.limitOptionsChanged)
        self.limit_method.currentTextChanged.connect(self.limitOptionsChanged)
        self.check_iterative.toggled.connect(self.limitOptionsChanged)

        self.compound_poisson = CompoundPoissonOptions()
        self.compound_poisson.method.currentTextChanged.connect(
            lambda: self.limitMethodChanged("Compound Poisson")
        )
        self.compound_poisson.limitOptionsChanged.connect(self.limitOptionsChanged)
        self.poisson = PoissonOptions()
        self.poisson.limitOptionsChanged.connect(self.limitOptionsChanged)
        self.gaussian = GaussianOptions()
        self.gaussian.limitOptionsChanged.connect(self.limitOptionsChanged)

        layout_method = QtWidgets.QHBoxLayout()
        layout_method.addWidget(self.limit_method, 1)
        layout_method.addWidget(self.check_iterative, 1)

        layout_button = QtWidgets.QHBoxLayout()
        layout_button.addWidget(
            self.button_advanced_options, 0, QtCore.Qt.AlignmentFlag.AlignRight
        )

        self.limit_inputs = QtWidgets.QGroupBox("Threshold Options")
        limit_layout = QtWidgets.QFormLayout()
        limit_layout.addRow("Window size:", layout_window_size)
        limit_layout.addRow("Threshold method:", layout_method)
        limit_layout.addRow(layout_button)
        limit_layout.addRow(self.compound_poisson)
        limit_layout.addRow(self.gaussian)
        limit_layout.addRow(self.poisson)
        self.limit_inputs.setLayout(limit_layout)

        layout_left = QtWidgets.QVBoxLayout()
        layout_left.addWidget(self.limit_inputs)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.inputs)
        layout.addLayout(layout_left)

        self.setLayout(layout)

    def instrumentOptions(self) -> SPCalInstrumentOptions:
        return SPCalInstrumentOptions(
            event_time=self.event_time.baseValue(),
            uptake=self.uptake.baseValue(),
            efficiency=self.efficiency.value(),
            # mass_response=self.mass_response,
        )

    def limitOptions(self) -> SPCalLimitOptions:
        return SPCalLimitOptions(
            method=self.limit_method.currentText().lower(),
            gaussian_kws=self.gaussian.state(),
            poisson_kws=self.poisson.state(),
            compound_poisson_kws=self.compound_poisson.state(),
            window_size=int(self.window_size.value() or 0),
            max_iterations=100 if self.check_iterative.isChecked() else 1,
        )

    def state(self) -> dict:
        state_dict = {
            "uptake": self.uptake.baseValue(),
            "dwelltime": self.event_time.baseValue(),
            "efficiency": self.efficiency.value(),
            "efficiency method": self.efficiency_method.currentText(),
            "window size": self.window_size.value(),
            "use window": self.check_window.isChecked(),
            "limit method": self.limit_method.currentText(),
            "accumulation method": self.limit_accumulation,
            "points required": self.points_required,
            "prominence required": self.prominence_required,
            "iterative": self.check_iterative.isChecked(),
            "compound poisson": self.compound_poisson.state(),
            "gaussian": self.gaussian.state(),
            "poisson": self.poisson.state(),
        }
        return {k: v for k, v in state_dict.items() if v is not None}

    def setState(self, state: dict) -> None:
        self.blockSignals(True)
        if "uptake" in state:
            self.uptake.setBaseValue(state["uptake"])
            self.uptake.setBestUnit()
        if "dwelltime" in state:
            self.event_time.setBaseValue(state["dwelltime"])
            self.event_time.setBestUnit()
        if "efficiency" in state:
            self.efficiency.setValue(state["efficiency"])

        self.efficiency_method.setCurrentText(state["efficiency method"])

        if "window size" in state:
            self.window_size.setValue(state["window size"])
        self.check_window.setChecked(bool(state["use window"]))
        self.check_iterative.setChecked(bool(state["iterative"]))

        if "accumulation method" in state:
            self.limit_accumulation = state["accumulation method"]
        if "points required" in state:
            self.points_required = state["points required"]
        if "prominence required" in state:
            self.prominence_required = state["prominence required"]

        self.compound_poisson.setState(state["compound poisson"])
        self.gaussian.setState(state["gaussian"])
        self.poisson.setState(state["poisson"])

        self.blockSignals(False)

        # Separate for useManualLimits signal
        self.limit_method.setCurrentText(state["limit method"])

        self.optionsChanged.emit()
        self.limitOptionsChanged.emit()

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
        manual = method == "Manual Input"

        nowindow = manual
        if (
            method == "Compound Poisson"
            and not self.compound_poisson.method.currentText() == "Lookup Table"
        ):
            nowindow = True

        self.useManualLimits.emit(manual)
        self.compound_poisson.setEnabled(not manual)
        self.gaussian.setEnabled(not manual)
        self.poisson.setEnabled(not manual)

        self.check_iterative.setEnabled(not manual)
        self.check_window.setEnabled(not nowindow)

        if nowindow:
            self.check_window.setChecked(False)
        if manual:
            self.check_iterative.setChecked(False)
        self.blockSignals(False)

    def isComplete(self) -> bool:
        if self.window_size.isEnabled() and not self.window_size.hasAcceptableInput():
            return False

        method = self.efficiency_method.currentText()
        if method == "Manual Input":
            return all(
                [
                    self.event_time.hasAcceptableInput(),
                    self.uptake.hasAcceptableInput(),
                    self.efficiency.hasAcceptableInput(),
                ]
            )
        elif method == "Reference Particle":
            return all(
                [
                    self.event_time.hasAcceptableInput(),
                    self.uptake.hasAcceptableInput(),
                ]
            )
        elif method == "Mass Response":
            return all(
                [
                    self.event_time.hasAcceptableInput(),
                ]
            )
        else:
            raise ValueError(f"Unknown method {method}.")

    def resetInputs(self) -> None:
        self.blockSignals(True)
        self.uptake.setValue(None)
        self.event_time.setValue(None)
        self.efficiency.setValue(None)
        self.blockSignals(False)
        self.optionsChanged.emit()

    def setSignificantFigures(self, num: int | None = None) -> None:
        if num is None:
            num = int(QtCore.QSettings().value("SigFigs", 4))  # type: ignore
        for widget in self.findChildren(ValueWidget):
            if widget.view_format[1] == "g":
                widget.setViewFormat(num)

    def setAdvancedOptions(
        self, accumlation_method: str, points_required: int, prominence_required: float
    ) -> None:
        self.limit_accumulation = accumlation_method
        self.points_required = points_required
        self.prominence_required = prominence_required

        settings = QtCore.QSettings()
        settings.setValue("Threshold/PointsRequired", points_required)
        settings.setValue("Threshold/ProminenceRequired", prominence_required)

        self.limitOptionsChanged.emit()

    def dialogAdvancedOptions(self) -> None:
        dlg = AdvancedThresholdOptions(
            self.limit_accumulation,
            self.points_required,
            self.prominence_required,
            parent=self,
        )
        dlg.optionsSelected.connect(self.setAdvancedOptions)
        dlg.open()
