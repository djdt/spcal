import logging
from typing import Iterator

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import spcal.particle
from spcal.calc import mode
from spcal.gui.dialogs.tools import MassFractionCalculatorDialog, ParticleDatabaseDialog
from spcal.gui.util import create_action
from spcal.gui.widgets import EditableComboBox, OverLabel, UnitsWidget, ValueWidget
from spcal.siunits import mass_concentration_units, size_units

logger = logging.getLogger(__name__)


class SampleIOWidget(QtWidgets.QWidget):
    optionsChanged = QtCore.Signal()
    request = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        sf = int(QtCore.QSettings().value("SigFigs", 4))  # type: ignore

        self.action_density = create_action(
            "folder-database",
            "Lookup Density",
            "Search for compound densities.",
            self.dialogParticleDatabase,
        )
        self.action_mass_fraction = create_action(
            "folder-calculate",
            "Calculate Mass Fraction",
            "Calculate the mass fraction and MW for a given formula.",
            self.dialogMassFractionCalculator,
        )
        self.action_ionic_response = create_action(
            "document-open",
            "Ionic Response Tool",
            "Read ionic responses from a file and apply to sample and reference.",
            lambda: self.request.emit("ionic response"),
        )

        self.density = UnitsWidget(
            {"g/cm³": 1e-3 * 1e6, "kg/m³": 1.0},
            default_unit="g/cm³",
            format=sf,
        )
        self.density.lineedit.addAction(
            self.action_density, QtWidgets.QLineEdit.ActionPosition.TrailingPosition
        )
        self.response = UnitsWidget(
            {
                "counts/(pg/L)": 1e15,
                "counts/(ng/L)": 1e12,
                "counts/(μg/L)": 1e9,
                "counts/(mg/L)": 1e6,
            },
            default_unit="counts/(μg/L)",
            format=sf,
        )
        self.response.lineedit.addAction(
            self.action_ionic_response,
            QtWidgets.QLineEdit.ActionPosition.TrailingPosition,
        )

        self.massfraction = ValueWidget(
            1.0,
            validator=QtGui.QDoubleValidator(0.0, 1.0, 16),
            format=sf,
        )
        self.massfraction.addAction(
            self.action_mass_fraction,
            QtWidgets.QLineEdit.ActionPosition.TrailingPosition,
        )

        self.density.setToolTip("Sample particle density.")
        self.response.setToolTip(
            "ICP-MS response for an ionic standard of this element."
        )
        self.massfraction.setToolTip(
            "Ratio of the mass of the analyte over the mass of the particle."
        )

        self.density.baseValueChanged.connect(self.optionsChanged)
        self.response.baseValueChanged.connect(self.optionsChanged)
        self.massfraction.valueChanged.connect(self.optionsChanged)

        self.inputs = QtWidgets.QGroupBox("Inputs")
        input_layout = QtWidgets.QFormLayout()
        input_layout.addRow("Density:", self.density)
        input_layout.addRow("Ionic response:", self.response)
        input_layout.addRow("Mass fraction:", self.massfraction)
        self.inputs.setLayout(input_layout)

        self.count = ValueWidget(0, format=("f", 0))
        self.count.setReadOnly(True)
        self.background_count = ValueWidget(format=sf)
        self.background_count.setReadOnly(True)
        self.lod_count = ValueWidget(format=sf)
        self.lod_count.setReadOnly(True)
        self.lod_label = OverLabel(self.lod_count, "")

        self.outputs = QtWidgets.QGroupBox("Outputs")
        output_layout = QtWidgets.QFormLayout()
        output_layout.addRow("Particle count:", self.count)
        output_layout.addRow("Background count:", self.background_count)
        output_layout.addRow("Detection threshold:", self.lod_label)
        self.outputs.setLayout(output_layout)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.inputs)
        layout.addWidget(self.outputs)

        self.setLayout(layout)

    def state(self) -> dict:
        state_dict = {
            "density": self.density.baseValue(),
            "response": self.response.baseValue(),
            "mass fraction": self.massfraction.value(),
        }
        if not self.lod_count.isReadOnly():  # Editable lod, save
            state_dict["lod"] = self.lod_count.value()
        return {k: v for k, v in state_dict.items() if v is not None}

    def setSignificantFigures(self, num: int | None = None):
        if num is None:
            num = int(QtCore.QSettings().value("SigFigs", 4))  # type: ignore
        for widget in self.findChildren(ValueWidget):
            if widget.view_format[1] == "g":
                widget.setViewFormat(num)

    def setState(self, state: dict):
        self.blockSignals(True)
        if "density" in state:
            self.density.setBaseValue(state["density"])
        if "response" in state:
            self.response.setBaseValue(state["response"])
        if "mass fraction" in state:
            self.massfraction.setValue(state["mass fraction"])
        self.blockSignals(False)
        self.optionsChanged.emit()

    def clearInputs(self):
        self.blockSignals(True)
        self.density.setValue(None)
        self.response.setValue(None)
        self.massfraction.setValue(1.0)
        self.blockSignals(False)

    def clearOutputs(self):
        self.count.setValue(None)
        self.count.setError(None)
        self.background_count.setValue(None)
        self.background_count.setError(None)
        self.lod_count.setValue(None)
        self.lod_label.setText("")

    def dialogMassFractionCalculator(self) -> QtWidgets.QDialog:
        def set_mass_fraction(ratios: list[tuple[str, float]]):
            self.massfraction.setValue(ratios[0][1])

        dlg = MassFractionCalculatorDialog(parent=self)
        dlg.ratiosSelected.connect(set_mass_fraction)
        dlg.open()
        return dlg

    def dialogParticleDatabase(self) -> QtWidgets.QDialog:
        dlg = ParticleDatabaseDialog(parent=self)
        dlg.densitySelected.connect(
            lambda x: self.density.setBaseValue(x * 1000.0)
        )  # to kg/m3
        dlg.open()
        return dlg

    def isComplete(self) -> bool:
        return (
            self.density.hasAcceptableInput()
            and (self.response.hasAcceptableInput() or not self.response.isEnabled())
            and self.massfraction.hasAcceptableInput()
        )

    def updateOutputs(
        self,
        responses: np.ndarray,
        detections: np.ndarray,
        labels: np.ndarray,
        lod: float,
        limit_str: str,
    ):
        bg_values = responses[labels == 0]
        if np.count_nonzero(np.isfinite(bg_values)) > 0:
            background, background_std = (
                float(np.nanmean(bg_values)),
                float(np.nanstd(bg_values)),
            )
        else:
            background, background_std = None, None

        count = np.count_nonzero(detections)

        self.count.setValue(int(count))
        self.count.setError(int(np.round(np.sqrt(count), 0)))
        self.background_count.setValue(background)
        self.background_count.setError(background_std)
        self.lod_count.setValue(lod)
        self.lod_label.setText(limit_str)

    def syncOutput(self, other: "SampleIOWidget", output: str):
        widget = getattr(self, output)
        if not isinstance(widget, UnitsWidget):
            raise ValueError("linkOutputs: output must be a UnitsWidget")
        widget.sync(getattr(other, output))


class ReferenceIOWidget(SampleIOWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent)

        sf = int(QtCore.QSettings().value("SigFigs", 4))  # type: ignore

        self.concentration = UnitsWidget(
            units=mass_concentration_units,
            default_unit="ng/L",
            color_invalid=QtGui.QColor(255, 255, 172),
            format=sf,
        )
        self.diameter = UnitsWidget(size_units, default_unit="nm", format=sf)

        self.concentration.setToolTip("Reference particle concentration.")
        self.diameter.setToolTip("Reference particle diameter.")

        self.concentration.baseValueChanged.connect(self.optionsChanged)
        self.diameter.baseValueChanged.connect(self.optionsChanged)

        input_layout = self.inputs.layout()
        assert isinstance(input_layout, QtWidgets.QFormLayout)
        input_layout.insertRow(0, "Concentration:", self.concentration)
        input_layout.insertRow(1, "Diameter:", self.diameter)

        self.check_use_efficiency_for_all = QtWidgets.QCheckBox(
            "Calibrate for all elements."
        )
        self.check_use_efficiency_for_all.setToolTip(
            "Use this element to calculate transport efficiency for all other elements,"
            " otherwise each element is calculated individually."
        )
        self.check_use_efficiency_for_all.setTristate(True)

        self.efficiency = ValueWidget(
            validator=QtGui.QDoubleValidator(0.0, 1.0, 10), format=sf
        )
        self.efficiency.setReadOnly(True)
        self.efficiency_label = OverLabel(self.efficiency, "")

        self.massresponse = UnitsWidget(
            {
                "ag/count": 1e-21,
                "fg/count": 1e-18,
                "pg/count": 1e-15,
                "ng/count": 1e-12,
                "μg/count": 1e-9,
                "mg/count": 1e-6,
                "g/count": 1e-3,
                "kg/count": 1.0,
            },
            default_unit="ag/count",
            format=sf,
        )
        self.massresponse.setReadOnly(True)

        output_layout = self.outputs.layout()
        assert isinstance(output_layout, QtWidgets.QFormLayout)
        output_layout.addRow("Trans. Efficiency:", self.efficiency_label)
        output_layout.addRow("", self.check_use_efficiency_for_all)
        output_layout.addRow("Mass Response:", self.massresponse)

    def state(self) -> dict:
        state_dict = super().state()
        state_dict.update(
            {
                "diameter": self.diameter.baseValue(),
                "concentration": self.concentration.baseValue(),
                "efficiency for all": self.check_use_efficiency_for_all.checkState()
                == QtCore.Qt.CheckState.Checked,
            }
        )
        return {k: v for k, v in state_dict.items() if v is not None}

    def setState(self, state: dict):
        self.blockSignals(True)
        if "diameter" in state:
            self.diameter.setBaseValue(state["diameter"])
        if "concentration" in state:
            self.concentration.setBaseValue(state["concentration"])
        self.blockSignals(False)
        # Outside for external signals
        if state["efficiency for all"]:
            self.check_use_efficiency_for_all.setCheckState(
                QtCore.Qt.CheckState.Checked
            )
        super().setState(state)

    def clearInputs(self):
        super().clearInputs()
        self.blockSignals(True)
        self.diameter.setValue(None)
        self.concentration.setValue(None)
        self.check_use_efficiency_for_all.setChecked(False)
        self.blockSignals(False)

    def clearOutputs(self):
        super().clearOutputs()
        self.efficiency.clear()
        self.massresponse.setValue(None)

    def updateEfficiency(
        self,
        detections: np.ndarray,
        dwell: float,
        time: float,
        uptake: float,
    ):
        # Make these delegates
        self.efficiency.setValue(None)
        self.efficiency_label.setText("")
        self.massresponse.setBaseValue(None)

        density = self.density.baseValue()
        diameter = self.diameter.baseValue()
        if density is None or diameter is None:
            return

        mass = spcal.particle.reference_particle_mass(density, diameter)
        mass_fraction = self.massfraction.value()
        if mass_fraction is not None:
            self.massresponse.setBaseValue(
                float(mass * mass_fraction / np.mean(detections[detections > 0]))
            )

        # If concentration defined use conc method
        concentration = self.concentration.baseValue()
        response = self.response.baseValue()
        if concentration is not None and uptake is not None:
            efficiency = spcal.particle.nebulisation_efficiency_from_concentration(
                int(np.count_nonzero(detections)),
                concentration=concentration,
                mass=mass,
                flow_rate=uptake,
                time=time,
            )
            self.efficiency.setValue(efficiency)
            self.efficiency_label.setText("(Concentration)")
        elif mass_fraction is not None and uptake is not None and response is not None:
            efficiency = spcal.particle.nebulisation_efficiency_from_mass(
                detections,
                dwell=dwell,
                mass=mass,
                flow_rate=uptake,
                response_factor=response,
                mass_fraction=mass_fraction,
            )
            self.efficiency.setValue(efficiency)
            self.efficiency_label.setText("(Mass)")

    def isComplete(self) -> bool:
        return super().isComplete() and self.diameter.hasAcceptableInput()


class ResultIOWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent)

        sf = int(QtCore.QSettings().value("SigFigs", 4))  # type: ignore

        self.count = ValueWidget(format=("f", 0))
        self.count.setReadOnly(True)
        self.count_label = OverLabel(self.count, "")
        self.number = UnitsWidget(
            {"#/L": 1.0, "#/ml": 1e3}, default_unit="#/L", format=sf
        )
        self.number.setReadOnly(True)
        self.conc = UnitsWidget(
            mass_concentration_units, default_unit="ng/L", format=sf
        )
        self.conc.setReadOnly(True)
        self.background = UnitsWidget(
            mass_concentration_units, default_unit="ng/L", format=sf
        )
        self.background.setReadOnly(True)

        self.mean = UnitsWidget(size_units, default_unit="nm", format=sf)
        self.mean.setReadOnly(True)
        self.median = UnitsWidget(size_units, default_unit="nm", format=sf)
        self.median.setReadOnly(True)
        self.mode = UnitsWidget(size_units, default_unit="nm", format=sf)
        self.mode.setReadOnly(True)
        self.lod = UnitsWidget(size_units, default_unit="nm", format=sf)
        self.lod.setReadOnly(True)

        layout_outputs_left = QtWidgets.QFormLayout()
        layout_outputs_left.addRow("No. Detections:", self.count_label)
        layout_outputs_left.addRow("No. Concentration:", self.number)
        layout_outputs_left.addRow("Concentration:", self.conc)
        layout_outputs_left.addRow("Ionic Background:", self.background)

        layout_outputs_right = QtWidgets.QFormLayout()
        layout_outputs_right.addRow("Mean:", self.mean)
        layout_outputs_right.addRow("Median:", self.median)
        layout_outputs_right.addRow("Mode:", self.mode)
        layout_outputs_right.addRow("LOD:", self.lod)

        self.outputs = QtWidgets.QGroupBox("Outputs")
        output_layout = QtWidgets.QHBoxLayout()
        output_layout.addLayout(layout_outputs_left)
        output_layout.addLayout(layout_outputs_right)
        self.outputs.setLayout(output_layout)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.outputs)

        self.setLayout(layout)

    def setSignificantFigures(self, num: int | None = None):
        if num is None:
            num = int(QtCore.QSettings().value("SigFigs", 4))  # type: ignore
        for widget in self.findChildren(ValueWidget):
            if widget.view_format[1] == "g":
                widget.setViewFormat(num)

    def clearOutputs(self):
        self.mean.setBaseValue(None)
        self.mean.setBaseError(None)
        self.median.setBaseValue(None)
        self.mode.setBaseValue(None)
        self.lod.setBaseValue(None)

        self.count.setValue(0)
        self.count_label.setText("")
        self.number.setBaseValue(None)
        self.number.setBaseError(None)
        self.conc.setBaseValue(None)
        self.conc.setBaseError(None)
        self.background.setBaseValue(None)
        self.background.setBaseError(None)

    def updateOutputs(
        self,
        values: np.ndarray,
        units: dict[str, float],
        lod: np.ndarray,
        count: float,
        count_percent: float,
        count_error: float,
        conc: float | None = None,
        number_conc: float | None = None,
        background_conc: float | None = None,
        background_error: float | None = None,
    ):
        if values.size == 0:  # will never be visible / enabled
            self.clearOutputs()
            return

        mean = np.mean(values)
        median = np.median(values)
        std = np.std(values, mean=mean)
        _mode = mode(values)

        mean_lod = np.mean(lod)

        relative_error = count_error / count

        for te in [self.mean, self.median, self.mode, self.lod]:
            te.setUnits(units)

        self.mean.setBaseValue(float(mean))
        self.mean.setBaseError(float(std))
        self.median.setBaseValue(float(median))
        self.mode.setBaseValue(_mode)
        self.lod.setBaseValue(float(mean_lod))

        unit = self.mean.setBestUnit()
        self.median.setUnit(unit)
        self.mode.setUnit(unit)
        self.lod.setUnit(unit)

        self.count.setValue(int(count))
        self.count.setError(int(np.round(count_error, 0)))
        self.count_label.setText(f"({count_percent:.0f} %)")
        self.number.setBaseValue(number_conc)
        if number_conc is not None:
            self.number.setBaseError(number_conc * relative_error)
        else:
            self.number.setBaseError(None)
        self.number.setBestUnit()

        self.conc.setBaseValue(conc)
        if conc is not None:
            self.conc.setBaseError(conc * relative_error)
        else:
            self.conc.setBaseError(None)
        unit = self.conc.setBestUnit()

        self.background.setBaseValue(background_conc)
        if background_conc is not None and background_error is not None:
            self.background.setBaseError(background_conc * background_error)
        else:
            self.background.setBaseError(None)
        self.background.setUnit(unit)


class SampleIOStack(QtWidgets.QWidget):
    nameChanged = QtCore.Signal(str)
    namesEdited = QtCore.Signal(dict)
    enabledNamesChanged = QtCore.Signal()

    optionsChanged = QtCore.Signal(str)

    requestIonicResponseTool = QtCore.Signal()
    limitsChanged = QtCore.Signal()

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        self.combo_name = EditableComboBox(self)
        self.combo_name.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.combo_name.setValidator(QtGui.QRegularExpressionValidator("[^\\s]+"))

        self.stack = QtWidgets.QStackedWidget()
        self.combo_name.currentIndexChanged.connect(self.stack.setCurrentIndex)
        self.combo_name.currentIndexChanged.connect(  # Otherwise emitted when 1 item and edit
            lambda i: self.nameChanged.emit(self.combo_name.itemText(i))
        )
        self.combo_name.enabledTextsChanged.connect(self.enabledNamesChanged)
        self.combo_name.textsEdited.connect(self.namesEdited)

        self.repopulate([""])

        self.layout_top = QtWidgets.QHBoxLayout()
        self.layout_top.addWidget(
            self.combo_name, 0, QtCore.Qt.AlignmentFlag.AlignRight
        )

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.layout_top)
        layout.addWidget(self.stack, 1)
        self.setLayout(layout)

    def __contains__(self, name: str) -> bool:
        return self.combo_name.findText(name) != -1

    def __getitem__(self, name: str) -> SampleIOWidget:
        return self.stack.widget(self.combo_name.findText(name))  # type: ignore

    def __iter__(self) -> Iterator[SampleIOWidget]:
        for i in range(self.stack.count()):
            yield self.stack.widget(i)  # type: ignore

    def names(self) -> list[str]:
        return [self.combo_name.itemText(i) for i in range(self.combo_name.count())]

    def enabledNames(self) -> list[str]:
        names = []
        model = self.combo_name.model()
        for i in range(self.combo_name.count()):
            if model.flags(model.index(i, 0)) & QtCore.Qt.ItemFlag.ItemIsEnabled:
                names.append(self.combo_name.itemText(i))
        return names

    def widgets(self) -> list[SampleIOWidget]:
        return [self.stack.widget(i) for i in range(self.stack.count())]  # type: ignore

    def handleRequest(self, request: str, value: None = None):
        if request == "ionic response":
            self.requestIonicResponseTool.emit()

    def clear(self):
        for widget in self.widgets():
            widget.clearInputs()
            widget.clearOutputs()

    def repopulate(self, names: list[str], widget_type: type = SampleIOWidget):
        self.blockSignals(True)
        old_widgets = {
            name: widget for name, widget in zip(self.names(), self.widgets())
        }

        self.combo_name.clear()
        while self.stack.count() > 0:
            self.stack.removeWidget(self.stack.widget(0))

        for i, name in enumerate(names):
            self.combo_name.addItem(name)
            if name in old_widgets:
                widget = old_widgets[name]
            else:
                widget = widget_type()
                widget.optionsChanged.connect(self.onWidgetOptionChanged)
                widget.request.connect(self.handleRequest)
            self.stack.addWidget(widget)
        self.blockSignals(False)

    def onWidgetOptionChanged(self):
        widget = self.sender()
        if not isinstance(widget, SampleIOWidget):
            raise ValueError("invalid widget triggered onWidgetOptionsChanged")
        name = self.combo_name.itemText(self.stack.indexOf(widget))
        self.optionsChanged.emit(name)

    def setSignificantFigures(self, num: int | None = None):
        for widget in self.widgets():
            widget.setSignificantFigures(num)

    def setResponses(self, responses: dict[str, float]):
        for name, response in responses.items():
            if name in self:
                self[name].response.setBaseValue(response)

    def setLimitsEditable(self, editable: bool):
        for widget in self.widgets():
            if editable:
                widget.lod_count.valueEdited.connect(self.limitsChanged)
            else:
                try:
                    widget.lod_count.valueEdited.disconnect(self.limitsChanged)
                except RuntimeError:
                    pass
            widget.lod_count.setReadOnly(not editable)


class ReferenceIOStack(SampleIOStack):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        self.button_group_check_efficiency = QtWidgets.QButtonGroup()
        self.button_group_check_efficiency.setExclusive(False)
        self.button_group_check_efficiency.buttonClicked.connect(self.buttonClicked)
        super().__init__(parent=parent)

    def buttonClicked(self, button: QtWidgets.QAbstractButton):
        assert isinstance(button, QtWidgets.QCheckBox)
        self.button_group_check_efficiency.blockSignals(True)
        if button.checkState() == QtCore.Qt.CheckState.PartiallyChecked:
            # Don't allow partial
            button.setCheckState(QtCore.Qt.CheckState.Checked)
        for b in self.button_group_check_efficiency.buttons():
            assert isinstance(b, QtWidgets.QCheckBox)
            if b == button:
                continue
            if button.checkState() == QtCore.Qt.CheckState.Checked:
                b.setCheckState(QtCore.Qt.CheckState.PartiallyChecked)
            else:
                b.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.button_group_check_efficiency.blockSignals(False)

    def repopulate(self, names: list[str], widget_type: type = ReferenceIOWidget):
        super().repopulate(names, ReferenceIOWidget)
        for name in names:
            io = self.stack.widget(self.combo_name.findText(name))
            if not isinstance(io, ReferenceIOWidget):
                raise TypeError("ReferenceIOStack must use ReferenceIOWidget")
            self.button_group_check_efficiency.addButton(
                io.check_use_efficiency_for_all
            )


class ResultIOStack(QtWidgets.QWidget):
    nameChanged = QtCore.Signal(str)
    modeChanged = QtCore.Signal(str)

    namesEdited = QtCore.Signal(dict)
    enabledNamesChanged = QtCore.Signal()

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        self.combo_name = EditableComboBox(self)
        self.combo_name.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.combo_name.setValidator(QtGui.QRegularExpressionValidator("[^\\s]+"))
        self.combo_name.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.NoContextMenu)

        self.stack = QtWidgets.QStackedWidget()
        self.combo_name.currentIndexChanged.connect(self.stack.setCurrentIndex)
        self.combo_name.currentIndexChanged.connect(  # Otherwise emitted when 1 item and edit
            lambda i: self.nameChanged.emit(self.combo_name.itemText(i))
        )
        self.combo_name.enabledTextsChanged.connect(self.enabledNamesChanged)
        self.combo_name.textsEdited.connect(self.namesEdited)

        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["Signal", "Mass", "Size", "Volume"])
        self.mode.setItemData(
            0, "Accumulated detection signal.", QtCore.Qt.ItemDataRole.ToolTipRole
        )
        self.mode.setItemData(
            1,
            "Particle mass, requires calibration.",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.mode.setItemData(
            2,
            "Particle size, requires calibration.",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.mode.setItemData(
            3,
            "Particle volume, requires calibration.",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.mode.setCurrentText("Signal")
        self.mode.currentTextChanged.connect(self.modeChanged)

        self.repopulate([""])

        self.layout_top = QtWidgets.QHBoxLayout()
        self.layout_top.addWidget(
            self.combo_name, 0, QtCore.Qt.AlignmentFlag.AlignRight
        )
        self.layout_top.insertWidget(
            0, QtWidgets.QLabel("Mode:"), 0, QtCore.Qt.AlignmentFlag.AlignLeft
        )
        self.layout_top.insertWidget(1, self.mode, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        self.layout_top.insertStretch(2, 1)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.layout_top)
        layout.addWidget(self.stack, 1)
        self.setLayout(layout)

    def __contains__(self, name: str) -> bool:
        return self.combo_name.findText(name) != -1

    def __getitem__(self, name: str) -> ResultIOWidget:
        return self.stack.widget(self.combo_name.findText(name))  # type: ignore

    def __iter__(self) -> Iterator[ResultIOWidget]:
        for i in range(self.stack.count()):
            yield self.stack.widget(i)  # type: ignore

    def names(self) -> list[str]:
        return [self.combo_name.itemText(i) for i in range(self.combo_name.count())]

    def enabledNames(self) -> list[str]:
        names = []
        model = self.combo_name.model()
        for i in range(self.combo_name.count()):
            if model.flags(model.index(i, 0)) & QtCore.Qt.ItemFlag.ItemIsEnabled:
                names.append(self.combo_name.itemText(i))
        return names

    def widgets(self) -> list[ResultIOWidget]:
        return [self.stack.widget(i) for i in range(self.stack.count())]  # type: ignore

    def repopulate(self, names: list[str]):
        self.blockSignals(True)
        old_widgets = {
            name: widget for name, widget in zip(self.names(), self.widgets())
        }

        self.combo_name.clear()
        while self.stack.count() > 0:
            self.stack.removeWidget(self.stack.widget(0))

        for i, name in enumerate(names):
            self.combo_name.addItem(name)
            if name in old_widgets:
                widget = old_widgets[name]
            else:
                widget = ResultIOWidget()
            self.stack.addWidget(widget)
        self.blockSignals(False)

    def setSignificantFigures(self, num: int | None = None):
        for widget in self.widgets():
            widget.setSignificantFigures(num)

    def clear(self):
        for widget in self.widgets():
            widget.clearOutputs()
