import logging
from typing import Dict, Generic, Iterator, List, Type, TypeVar

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import spcal

# from spcal import npdata
from spcal.gui.dialogs.tools import MassFractionCalculatorDialog, ParticleDatabaseDialog
from spcal.gui.util import create_action
from spcal.gui.widgets import UnitsWidget, ValidColorLineEdit
from spcal.siunits import mass_concentration_units, size_units

logger = logging.getLogger(__name__)


class IOWidget(QtWidgets.QWidget):
    optionsChanged = QtCore.Signal(str)

    def __init__(self, name: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.name = name

    def clearInputs(self) -> None:
        raise NotImplementedError

    def clearOutputs(self) -> None:
        raise NotImplementedError

    def updateInputs(self) -> None:
        raise NotImplementedError

    def updateOutputs(self) -> None:
        raise NotImplementedError

    def isComplete(self) -> bool:
        return True


class SampleIOWidget(IOWidget):
    def __init__(self, name: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(name, parent)

        self.action_density = create_action(
            "folder-database",
            "Lookup Density",
            "Search for compound densities.",
            self.dialogParticleDatabase,
        )
        self.action_mass_fraction = create_action(
            "folder-calculate",
            "Calculate Molar Ratio",
            "Calculate the molar ratio and MW for a given formula.",
            self.dialogMassFractionCalculator,
        )

        self.inputs = QtWidgets.QGroupBox("Inputs")
        self.inputs.setLayout(QtWidgets.QFormLayout())

        # self.element = ValidColorLineEdit(color_bad=QtGui.QColor(255, 255, 172))
        # self.element.setValid(False)
        # self.element.setCompleter(QtWidgets.QCompleter(list(npdata.data.keys())))
        # self.element.textChanged.connect(self.elementChanged)

        self.density = UnitsWidget(
            {"g/cm³": 1e-3 * 1e6, "kg/m³": 1.0},
            default_unit="g/cm³",
        )
        self.density.lineedit.addAction(
            self.action_density, QtWidgets.QLineEdit.TrailingPosition
        )
        self.molarmass = UnitsWidget(
            {"g/mol": 1e-3, "kg/mol": 1.0},
            default_unit="g/mol",
            color_invalid=QtGui.QColor(255, 255, 172),
        )
        self.response = UnitsWidget(
            {
                "counts/(pg/L)": 1e15,
                "counts/(ng/L)": 1e12,
                "counts/(μg/L)": 1e9,
                "counts/(mg/L)": 1e6,
            },
            default_unit="counts/(μg/L)",
        )
        self.massfraction = ValidColorLineEdit("1.0")
        self.massfraction.setValidator(QtGui.QDoubleValidator(0.0, 1.0, 4))
        self.massfraction.addAction(
            self.action_mass_fraction, QtWidgets.QLineEdit.TrailingPosition
        )

        # self.element.setToolTip(
        #     "Input formula for density, molarmass and massfraction."
        # )
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

        self.density.valueChanged.connect(lambda: self.optionsChanged.emit(self.name))
        self.molarmass.valueChanged.connect(lambda: self.optionsChanged.emit(self.name))
        self.response.valueChanged.connect(lambda: self.optionsChanged.emit(self.name))
        self.massfraction.textChanged.connect(
            lambda: self.optionsChanged.emit(self.name)
        )

        # self.inputs.layout().addRow("Formula:", self.element)
        self.inputs.layout().addRow("Density:", self.density)
        self.inputs.layout().addRow("Molar mass:", self.molarmass)
        self.inputs.layout().addRow("Ionic response:", self.response)
        self.inputs.layout().addRow("Molar ratio:", self.massfraction)

        self.count = QtWidgets.QLineEdit("0")
        self.count.setReadOnly(True)
        self.background_count = QtWidgets.QLineEdit()
        self.background_count.setReadOnly(True)
        self.lod_count = QtWidgets.QLineEdit()
        self.lod_count.setReadOnly(True)

        self.outputs = QtWidgets.QGroupBox("Outputs")
        self.outputs.setLayout(QtWidgets.QFormLayout())
        self.outputs.layout().addRow("Particle count:", self.count)
        self.outputs.layout().addRow("Background count:", self.background_count)
        self.outputs.layout().addRow("LOD count:", self.lod_count)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.inputs)
        layout.addWidget(self.outputs)

        self.setLayout(layout)

    def clearInputs(self) -> None:
        self.blockSignals(True)
        # self.element.setText("")
        self.density.setValue(None)
        self.molarmass.setValue(None)
        self.response.setValue(None)
        self.massfraction.setText("1.0")
        self.blockSignals(False)

    def clearOutputs(self) -> None:
        self.count.setText("")
        self.background_count.setText("")
        self.lod_count.setText("")

    def dialogMassFractionCalculator(self) -> QtWidgets.QDialog:
        def set_mass_fraction(ratios: Dict[str, float]):
            first = next(iter(ratios.values()))
            self.massfraction.setText(f"{first:.4f}")

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

    # def elementChanged(self, text: str) -> None:
    #     if text in npdata.data:
    #         density, mw, mr = npdata.data[text]
    #         # self.element.setValid(True)
    #         self.density.setValue(density)
    #         self.density.setUnit("g/cm³")
    #         self.density.setEnabled(False)
    #         self.molarmass.setValue(mw)
    #         self.molarmass.setUnit("g/mol")
    #         self.molarmass.setEnabled(False)
    #         self.massfraction.setText(str(mr))
    #         self.massfraction.setEnabled(False)
    #     else:
    #         # self.element.setValid(False)
    #         self.density.setEnabled(True)
    #         self.molarmass.setEnabled(True)
    #         self.massfraction.setEnabled(True)

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
        limit_name: str,
        limit_params: Dict[str, float],
    ) -> None:
        background = np.mean(responses[labels == 0])
        background_std = np.std(responses[labels == 0])

        count = np.count_nonzero(detections)

        self.count.setText(f"{count} ± {np.sqrt(count):.1f}")
        self.background_count.setText(f"{background:.4g} ± {background_std:.4g}")
        self.lod_count.setText(
            f"{lod:.4g} ({limit_name}, {','.join(f'{k}={v}' for k,v in limit_params.items())})"
        )

    def syncOutput(self, other: "SampleIOWidget", output: str) -> None:
        widget = getattr(self, output)
        if not isinstance(widget, UnitsWidget):
            raise ValueError("linkOutputs: output must be a UnitsWidget")
        widget.sync(getattr(other, output))


class ReferenceIOWidget(SampleIOWidget):
    def __init__(self, name: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(name, parent=parent)

        self.concentration = UnitsWidget(
            units=mass_concentration_units,
            default_unit="ng/L",
            color_invalid=QtGui.QColor(255, 255, 172),
        )
        self.diameter = UnitsWidget(size_units, default_unit="nm")

        self.concentration.setToolTip("Reference particle concentration.")
        self.diameter.setToolTip("Reference particle diameter.")

        self.concentration.valueChanged.connect(
            lambda: self.optionsChanged.emit(self.name)
        )
        self.diameter.valueChanged.connect(lambda: self.optionsChanged.emit(self.name))

        self.inputs.layout().setRowVisible(self.molarmass, False)
        self.inputs.layout().insertRow(0, "Concentration:", self.concentration)
        self.inputs.layout().insertRow(1, "Diameter:", self.diameter)

        self.check_use_efficiency_for_all = QtWidgets.QCheckBox(
            "Calibrate for all elements."
        )
        self.check_use_efficiency_for_all.setToolTip(
            "Use this element to calculate transport efficiency for all other elements, "
            "otherwise each element is calculated individually."
        )

        self.efficiency = QtWidgets.QLineEdit()
        self.efficiency.setValidator(QtGui.QDoubleValidator(0.0, 1.0, 10))
        self.efficiency.setReadOnly(True)

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
        )
        self.massresponse.setReadOnly(True)

        self.outputs.layout().addRow("Trans. Efficiency:", self.efficiency)
        self.outputs.layout().addRow("", self.check_use_efficiency_for_all)
        self.outputs.layout().addRow("Mass Response:", self.massresponse)

    def clearInputs(self) -> None:
        super().clearInputs()
        self.blockSignals(True)
        self.diameter.setValue(None)
        self.concentration.setValue(None)
        self.efficiency.setText("")
        self.massresponse.setValue(None)
        self.blockSignals(False)

    def updateEfficiency(
        self,
        detections: np.ndarray,
        dwell: float,
        time: float,
        uptake: float,
    ) -> None:
        self.efficiency.setText("")
        self.massresponse.setValue("")

        density = self.density.baseValue()
        diameter = self.diameter.baseValue()
        response = self.response.baseValue()
        if density is None or diameter is None or response is None:
            return

        mass = spcal.reference_particle_mass(density, diameter)
        mass_fraction = (
            float(self.massfraction.text())
            if self.massfraction.hasAcceptableInput()
            else None
        )
        if mass_fraction is not None:
            self.massresponse.setBaseValue(mass * mass_fraction / np.mean(detections))

        # If concentration defined use conc method
        concentration = self.concentration.baseValue()
        if concentration is not None and uptake is not None:
            efficiency = spcal.nebulisation_efficiency_from_concentration(
                detections.size,
                concentration=concentration,
                mass=mass,
                flow_rate=uptake,
                time=time,
            )
            self.efficiency.setText(f"{efficiency:.4g}")
        elif mass_fraction is not None and uptake is not None and response is not None:
            efficiency = spcal.nebulisation_efficiency_from_mass(
                detections,
                dwell=dwell,
                mass=mass,
                flow_rate=uptake,
                response_factor=response,
                mass_fraction=mass_fraction,
            )
            self.efficiency.setText(f"{efficiency:.4g}")

    def isComplete(self) -> bool:
        return super().isComplete() and self.diameter.hasAcceptableInput()


class ResultIOWidget(IOWidget):
    optionsChanged = QtCore.Signal(str)

    def __init__(self, name: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(name, parent=parent)
        self.outputs = QtWidgets.QGroupBox("Outputs")
        self.outputs.setLayout(QtWidgets.QHBoxLayout())

        self.count = QtWidgets.QLineEdit()
        self.count.setReadOnly(True)
        self.number = UnitsWidget(
            {"#/L": 1.0, "#/ml": 1e3},
            default_unit="#/L",
            formatter=".0f",
        )
        self.number.setReadOnly(True)
        self.conc = UnitsWidget(
            mass_concentration_units,
            default_unit="ng/L",
        )
        self.conc.setReadOnly(True)
        self.background = UnitsWidget(
            mass_concentration_units,
            default_unit="ng/L",
        )
        self.background.setReadOnly(True)

        self.lod = UnitsWidget(
            size_units,
            default_unit="nm",
        )
        self.lod.setReadOnly(True)
        self.mean = UnitsWidget(
            size_units,
            default_unit="nm",
        )
        self.mean.setReadOnly(True)
        self.median = UnitsWidget(
            size_units,
            default_unit="nm",
        )
        self.median.setReadOnly(True)

        layout_outputs_left = QtWidgets.QFormLayout()
        layout_outputs_left.addRow("No. Detections:", self.count)
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

        self.count.setText("")
        self.number.setBaseValue(None)
        self.number.setBaseError(None)
        self.conc.setBaseValue(None)
        self.conc.setBaseError(None)
        self.background.setBaseValue(None)
        self.background.setBaseError(None)

    def updateOutputs(
        self,
        values: np.ndarray,
        units: Dict[str, float],
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

        relative_error = count / count_error
        self.count.setText(f"{count} ± {count_error:.1f} ({count_percent:.1f} %)")
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


IOType = TypeVar("IOType", bound=IOWidget)


class IOStack(QtWidgets.QWidget, Generic[IOType]):
    nameChanged = QtCore.Signal(str)
    optionsChanged = QtCore.Signal(str)

    def __init__(
        self,
        io_widget_type: Type[IOType],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        self.io_widget_type = io_widget_type

        self.combo_name = QtWidgets.QComboBox()
        self.combo_name.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)

        self.stack = QtWidgets.QStackedWidget()
        self.combo_name.currentIndexChanged.connect(self.stack.setCurrentIndex)
        self.combo_name.currentTextChanged.connect(self.nameChanged)

        self.repopulate(["<element>"])

        self.layout_top = QtWidgets.QHBoxLayout()
        self.layout_top.addWidget(self.combo_name, 0, QtCore.Qt.AlignRight)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.layout_top)
        layout.addWidget(self.stack, 1)
        self.setLayout(layout)

    def __contains__(self, name: str) -> bool:
        return self.combo_name.findText(name) != -1

    def __getitem__(self, name: str) -> IOType:
        return self.stack.widget(self.combo_name.findText(name))  # type: ignore

    def __iter__(self) -> Iterator[IOType]:
        for i in range(self.stack.count()):
            yield self.stack.widget(i)

    def names(self) -> List[str]:
        return [self.combo_name.itemText(i) for i in range(self.combo_name.count())]

    def widgets(self) -> List[IOType]:
        return [self.stack.widget(i) for i in range(self.stack.count())]  # type: ignore

    def repopulate(self, names: List[str]) -> None:
        self.blockSignals(True)
        self.combo_name.clear()
        while self.stack.count() > 0:
            self.stack.removeWidget(self.stack.widget(0))

        for name in names:
            self.combo_name.addItem(name)
            widget = self.io_widget_type(name)
            widget.optionsChanged.connect(self.optionsChanged)
            self.stack.addWidget(widget)
        self.blockSignals(False)

    def resetInputs(self) -> None:
        for widget in self.widgets():
            widget.resetInputs()


class SampleIOStack(IOStack[SampleIOWidget]):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(SampleIOWidget, parent=parent)


class ReferenceIOStack(IOStack[ReferenceIOWidget]):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        self.button_group_check_efficiency = QtWidgets.QButtonGroup()
        self.button_group_check_efficiency.buttonClicked.connect(self.buttonClicked)
        self.last_button_checked: QtWidgets.QAbstractButton | None = None
        super().__init__(ReferenceIOWidget, parent=parent)

    def buttonClicked(self, button: QtWidgets.QAbstractButton) -> None:
        if button == self.last_button_checked:
            self.button_group_check_efficiency.setExclusive(False)
            self.button_group_check_efficiency.checkedButton().setChecked(False)
            self.button_group_check_efficiency.setExclusive(True)
            self.last_button_checked = None
        else:
            self.last_button_checked = button

    def repopulate(self, names: List[str]) -> None:
        super().repopulate(names)
        for name in names:
            self.button_group_check_efficiency.addButton(
                self[name].check_use_efficiency_for_all
            )


class ResultIOStack(IOStack[ResultIOWidget]):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(ResultIOWidget, parent=parent)
