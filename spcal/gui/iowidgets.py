import logging
from typing import Iterator, Type

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import spcal.particle
from spcal.gui.dialogs.tools import MassFractionCalculatorDialog, ParticleDatabaseDialog
from spcal.gui.util import create_action
from spcal.gui.widgets import EditableComboBox, OverLabel, UnitsWidget, ValueWidget
from spcal.siunits import mass_concentration_units, size_units

logger = logging.getLogger(__name__)


class IOWidget(QtWidgets.QWidget):
    optionsChanged = QtCore.Signal()
    request = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

    def clearInputs(self) -> None:
        raise NotImplementedError

    def clearOutputs(self) -> None:
        raise NotImplementedError

    def updateInputs(self) -> None:
        raise NotImplementedError

    def updateOutputs(self) -> None:
        raise NotImplementedError

    def setSignificantFigures(self, num: int | None = None) -> None:
        if num is None:
            num = int(QtCore.QSettings().value("SigFigs", 4))
        for widget in self.findChildren(ValueWidget):
            if widget.view_format.endswith("g"):
                widget.setViewFormat(num)

    def isComplete(self) -> bool:
        return True


class SampleIOWidget(IOWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        sf = int(QtCore.QSettings().value("SigFigs", 4))

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

        self.inputs = QtWidgets.QGroupBox("Inputs")
        self.inputs.setLayout(QtWidgets.QFormLayout())
        self.density = UnitsWidget(
            {"g/cm³": 1e-3 * 1e6, "kg/m³": 1.0},
            default_unit="g/cm³",
            format=sf,
        )
        self.density.lineedit.addAction(
            self.action_density, QtWidgets.QLineEdit.ActionPosition.TrailingPosition
        )
        self.molarmass = UnitsWidget(
            {"g/mol": 1e-3, "kg/mol": 1.0},
            default_unit="g/mol",
            color_invalid=QtGui.QColor(255, 255, 172),
            format=sf,
        )
        self.molarmass.lineedit.addAction(
            self.action_mass_fraction,
            QtWidgets.QLineEdit.ActionPosition.TrailingPosition,
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
        self.molarmass.setToolTip(
            "Molecular weight, required to calculate intracellular concentrations."
        )
        self.response.setToolTip(
            "ICP-MS response for an ionic standard of this element."
        )
        self.massfraction.setToolTip(
            "Ratio of the mass of the analyte over the mass of the particle."
        )

        self.density.baseValueChanged.connect(self.optionsChanged)
        self.molarmass.baseValueChanged.connect(self.optionsChanged)
        self.response.baseValueChanged.connect(self.optionsChanged)
        self.massfraction.valueChanged.connect(self.optionsChanged)

        self.inputs.layout().addRow("Density:", self.density)
        self.inputs.layout().addRow("Molar mass:", self.molarmass)
        self.inputs.layout().addRow("Ionic response:", self.response)
        self.inputs.layout().addRow("Mass fraction:", self.massfraction)

        self.count = ValueWidget(0, format="d")
        self.count.setReadOnly(True)
        self.background_count = ValueWidget(format=sf)
        self.background_count.setReadOnly(True)
        self.lod_count = ValueWidget(format=sf)
        self.lod_count.setReadOnly(True)
        self.lod_label = OverLabel(self.lod_count, "")

        self.outputs = QtWidgets.QGroupBox("Outputs")
        self.outputs.setLayout(QtWidgets.QFormLayout())
        self.outputs.layout().addRow("Particle count:", self.count)
        self.outputs.layout().addRow("Background count:", self.background_count)
        self.outputs.layout().addRow("Detection threshold:", self.lod_label)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.inputs)
        layout.addWidget(self.outputs)

        self.setLayout(layout)

    def state(self) -> dict:
        state_dict = {
            "density": self.density.baseValue(),
            "molar mass": self.molarmass.baseValue(),
            "response": self.response.baseValue(),
            "mass fraction": self.massfraction.value(),
        }
        if not self.lod_count.isReadOnly():  # Editable lod, save
            state_dict["lod"] = self.lod_count.value()
        return {k: v for k, v in state_dict.items() if v is not None}

    def setState(self, state: dict) -> None:
        self.blockSignals(True)
        if "density" in state:
            self.density.setBaseValue(state["density"])
        if "molar mass" in state:
            self.molarmass.setBaseValue(state["molar mass"])
        if "response" in state:
            self.response.setBaseValue(state["response"])
        if "mass fraction" in state:
            self.massfraction.setValue(state["mass fraction"])
        self.blockSignals(False)
        self.optionsChanged.emit()

    def clearInputs(self) -> None:
        self.blockSignals(True)
        self.density.setValue(None)
        self.molarmass.setValue(None)
        self.response.setValue(None)
        self.massfraction.setValue(1.0)
        self.blockSignals(False)

    def clearOutputs(self) -> None:
        self.count.setValue(None)
        self.background_count.setValue(None)
        self.lod_count.setValue(None)
        self.lod_label.setText("")

    def dialogMassFractionCalculator(self) -> QtWidgets.QDialog:
        def set_mass_fraction(ratios: list[tuple[str, float]]):
            self.massfraction.setValue(ratios[0][1])

        dlg = MassFractionCalculatorDialog(parent=self)
        dlg.ratiosSelected.connect(set_mass_fraction)
        dlg.molarMassSelected.connect(lambda x: self.molarmass.setBaseValue(x / 1000.0))
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
    ) -> None:
        background = np.nanmean(responses[labels == 0])
        background_std = np.nanstd(responses[labels == 0])

        count = np.count_nonzero(detections)

        self.count.setValue(int(count))
        self.count.setError(int(np.round(np.sqrt(count), 0)))
        self.background_count.setValue(background)
        self.background_count.setError(background_std)
        self.lod_count.setValue(lod)
        self.lod_label.setText(limit_str)

    def syncOutput(self, other: "SampleIOWidget", output: str) -> None:
        widget = getattr(self, output)
        if not isinstance(widget, UnitsWidget):
            raise ValueError("linkOutputs: output must be a UnitsWidget")
        widget.sync(getattr(other, output))


class ReferenceIOWidget(SampleIOWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent)

        sf = int(QtCore.QSettings().value("SigFigs", 4))

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

        self.inputs.layout().setRowVisible(self.molarmass, False)
        self.inputs.layout().insertRow(0, "Concentration:", self.concentration)
        self.inputs.layout().insertRow(1, "Diameter:", self.diameter)

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

        self.outputs.layout().addRow("Trans. Efficiency:", self.efficiency_label)
        self.outputs.layout().addRow("", self.check_use_efficiency_for_all)
        self.outputs.layout().addRow("Mass Response:", self.massresponse)

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

    def setState(self, state: dict) -> None:
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

    def clearInputs(self) -> None:
        super().clearInputs()
        self.blockSignals(True)
        self.diameter.setValue(None)
        self.concentration.setValue(None)
        self.check_use_efficiency_for_all.setChecked(False)
        self.blockSignals(False)

    def clearOutputs(self) -> None:
        super().clearOutputs()
        self.efficiency.clear()
        self.massresponse.setValue(None)

    def updateEfficiency(
        self,
        detections: np.ndarray,
        dwell: float,
        time: float,
        uptake: float,
    ) -> None:
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
                mass * mass_fraction / np.mean(detections[detections > 0])
            )

        # If concentration defined use conc method
        concentration = self.concentration.baseValue()
        response = self.response.baseValue()
        if concentration is not None and uptake is not None:
            efficiency = spcal.particle.nebulisation_efficiency_from_concentration(
                np.count_nonzero(detections),
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


class ResultIOWidget(IOWidget):
    optionsChanged = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent)

        sf = int(QtCore.QSettings().value("SigFigs", 4))

        self.outputs = QtWidgets.QGroupBox("Outputs")
        self.outputs.setLayout(QtWidgets.QHBoxLayout())

        self.count = ValueWidget(format=".0f")
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

        self.lod = UnitsWidget(size_units, default_unit="nm", format=sf)
        self.lod.setReadOnly(True)
        self.mean = UnitsWidget(size_units, default_unit="nm", format=sf)
        self.mean.setReadOnly(True)
        self.median = UnitsWidget(size_units, default_unit="nm", format=sf)
        self.median.setReadOnly(True)

        layout_outputs_left = QtWidgets.QFormLayout()
        layout_outputs_left.addRow("No. Detections:", self.count_label)
        layout_outputs_left.addRow("No. Concentration:", self.number)
        layout_outputs_left.addRow("Concentration:", self.conc)
        layout_outputs_left.addRow("Ionic Background:", self.background)

        layout_outputs_right = QtWidgets.QFormLayout()
        layout_outputs_right.addRow("Mean:", self.mean)
        layout_outputs_right.addRow("Median:", self.median)
        layout_outputs_right.addRow("LOD:", self.lod)

        self.outputs.layout().addLayout(layout_outputs_left)
        self.outputs.layout().addLayout(layout_outputs_right)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.outputs)

        self.setLayout(layout)

    def clearOutputs(self) -> None:
        self.mean.setBaseValue(None)
        self.mean.setBaseError(None)
        self.median.setBaseValue(None)
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
    ) -> None:
        mean = np.mean(values)
        median = np.median(values)
        std = np.std(values)
        mean_lod = np.mean(lod)

        for te in [self.mean, self.median, self.lod]:
            te.setUnits(units)

        self.mean.setBaseValue(mean)
        self.mean.setBaseError(std)
        self.median.setBaseValue(median)
        self.lod.setBaseValue(mean_lod)

        unit = self.mean.setBestUnit()
        self.median.setUnit(unit)
        self.lod.setUnit(unit)

        relative_error = count_error / count
        self.count.setValue(int(count))
        self.count.setError(int(np.round(count_error, 0)))
        self.count_label.setText(f"({count_percent:{self.count.view_format}} %)")
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


class IOStack(QtWidgets.QWidget):
    nameChanged = QtCore.Signal(str)
    namesEdited = QtCore.Signal(dict)
    enabledNamesChanged = QtCore.Signal()

    optionsChanged = QtCore.Signal(str)

    def __init__(
        self,
        io_widget_type: Type[IOWidget],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        self.io_widget_type = io_widget_type

        self.combo_name = EditableComboBox(self)
        self.combo_name.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
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
        self.layout_top.addWidget(self.combo_name, 0, QtCore.Qt.AlignRight)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.layout_top)
        layout.addWidget(self.stack, 1)
        self.setLayout(layout)

    def __contains__(self, name: str) -> bool:
        return self.combo_name.findText(name) != -1

    def __getitem__(self, name: str) -> IOWidget:
        return self.stack.widget(self.combo_name.findText(name))  # type: ignore

    def __iter__(self) -> Iterator[IOWidget]:
        for i in range(self.stack.count()):
            yield self.stack.widget(i)

    def names(self) -> list[str]:
        return [self.combo_name.itemText(i) for i in range(self.combo_name.count())]

    def enabledNames(self) -> list[str]:
        return [
            self.combo_name.itemText(i)
            for i in range(self.combo_name.count())
            if self.combo_name.model().item(i).isEnabled()
        ]

    def widgets(self) -> list[IOWidget]:
        return [self.stack.widget(i) for i in range(self.stack.count())]  # type: ignore

    def handleRequest(self, request: str, value: None = None) -> None:
        raise NotImplementedError

    def repopulate(self, names: list[str]) -> None:
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
                widget = self.io_widget_type()
                widget.optionsChanged.connect(
                    lambda: self.onWidgetOptionChanged(widget)
                )
                widget.request.connect(self.handleRequest)
            self.stack.addWidget(widget)
        self.blockSignals(False)

    def onWidgetOptionChanged(self, widget: IOWidget) -> None:
        name = self.combo_name.itemText(self.stack.indexOf(widget))
        self.optionsChanged.emit(name)

    def setSignificantFigures(self, num: int | None = None) -> None:
        for widget in self.widgets():
            widget.setSignificantFigures(num)

    def resetInputs(self) -> None:
        for widget in self.widgets():
            widget.resetInputs()


class SampleIOStack(IOStack):
    requestIonicResponseTool = QtCore.Signal()
    limitsChanged = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(SampleIOWidget, parent=parent)

    def handleRequest(self, request: str, value: None = None) -> None:
        if request == "ionic response":
            self.requestIonicResponseTool.emit()

    def setResponses(self, responses: dict[str, float]) -> None:
        for name, response in responses.items():
            if name in self:
                self[name].response.setBaseValue(response)

    def setLimitsEditable(self, editable: bool) -> None:
        for widget in self.widgets():
            if editable:
                widget.lod_count.valueEdited.connect(self.limitsChanged)
            else:
                try:
                    widget.lod_count.valueEdited.disconnect(self.limitsChanged)
                except RuntimeError:
                    pass
            widget.lod_count.setReadOnly(not editable)


class ReferenceIOStack(IOStack):
    requestIonicResponseTool = QtCore.Signal()
    limitsChanged = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        self.button_group_check_efficiency = QtWidgets.QButtonGroup()
        self.button_group_check_efficiency.setExclusive(False)
        self.button_group_check_efficiency.buttonClicked.connect(self.buttonClicked)
        super().__init__(ReferenceIOWidget, parent=parent)

    def handleRequest(self, request: str, value: None = None) -> None:
        if request == "ionic response":
            self.requestIonicResponseTool.emit()

    def buttonClicked(self, button: QtWidgets.QAbstractButton) -> None:
        assert isinstance(button, QtWidgets.QCheckBox)
        self.button_group_check_efficiency.blockSignals(True)
        if button.checkState() == QtCore.Qt.CheckState.PartiallyChecked:
            # Don't allow partial
            button.setCheckState(QtCore.Qt.CheckState.Checked)
        for b in self.button_group_check_efficiency.buttons():
            if b == button:
                continue
            if button.checkState() == QtCore.Qt.CheckState.Checked:
                b.setCheckState(QtCore.Qt.CheckState.PartiallyChecked)
            else:
                b.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.button_group_check_efficiency.blockSignals(False)

    def repopulate(self, names: list[str]) -> None:
        super().repopulate(names)
        for name in names:
            self.button_group_check_efficiency.addButton(
                self[name].check_use_efficiency_for_all
            )

    def setResponses(self, responses: dict[str, float]) -> None:
        for name, response in responses.items():
            if name in self:
                self[name].response.setBaseValue(response)

    def setLimitsEditable(self, editable: bool) -> None:
        for widget in self.widgets():
            if editable:
                widget.lod_count.valueChanged.connect(self.limitsChanged)
            else:
                try:
                    widget.lod_count.valueChanged.disconnect(self.limitsChanged)
                except RuntimeError:
                    pass
            widget.lod_count.setReadOnly(not editable)


class ResultIOStack(IOStack):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(ResultIOWidget, parent=parent)
        self.combo_name.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.NoContextMenu)
