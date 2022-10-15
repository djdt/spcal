from PySide6 import QtCore, QtGui, QtWidgets

import numpy as np
import logging

import spcal
from spcal import npdata

from spcal.gui.units import UnitsWidget
from spcal.gui.widgets import ValidColorLineEdit

from typing import Dict, Generic, List, Optional, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)


class IOWidget(QtWidgets.QWidget):
    optionsChanged = QtCore.Signal(str)

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
    def __init__(self, name: str, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.name = name

        self.inputs = QtWidgets.QGroupBox("Inputs")
        self.inputs.setLayout(QtWidgets.QFormLayout())

        self.element = ValidColorLineEdit(color_bad=QtGui.QColor(255, 255, 172))
        self.element.setValid(False)
        self.element.setCompleter(QtWidgets.QCompleter(list(npdata.data.keys())))
        self.element.textChanged.connect(self.elementChanged)

        self.density = UnitsWidget(
            {"g/cm³": 1e-3 * 1e6, "kg/m³": 1.0},
            default_unit="g/cm³",
        )
        self.molarmass = UnitsWidget(
            {"g/mol": 1e-3, "kg/mol": 1.0},
            default_unit="g/mol",
            invalid_color=QtGui.QColor(255, 255, 172),
        )
        self.massfraction = ValidColorLineEdit("1.0")
        self.massfraction.setValidator(QtGui.QDoubleValidator(0.0, 1.0, 4))

        self.element.setToolTip(
            "Input formula for density, molarmass and massfraction."
        )
        self.density.setToolTip("Sample particle density.")
        self.molarmass.setToolTip(
            "Molecular weight, required to calculate intracellular concentrations."
        )
        self.massfraction.setToolTip(
            "Ratio of the mass of the analyte over the mass of the particle."
        )

        self.density.valueChanged.connect(lambda: self.optionsChanged.emit(self.name))
        self.molarmass.valueChanged.connect(lambda: self.optionsChanged.emit(self.name))
        self.massfraction.textChanged.connect(
            lambda: self.optionsChanged.emit(self.name)
        )

        self.inputs.layout().addRow("Formula:", self.element)
        self.inputs.layout().addRow("Density:", self.density)
        self.inputs.layout().addRow("Molar mass:", self.molarmass)
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

    def elementChanged(self, text: str) -> None:
        if text in npdata.data:
            density, mw, mr = npdata.data[text]
            self.element.setValid(True)
            self.density.setValue(density)
            self.density.setUnit("g/cm³")
            self.density.setEnabled(False)
            self.molarmass.setValue(mw)
            self.molarmass.setUnit("g/mol")
            self.molarmass.setEnabled(False)
            self.massfraction.setText(str(mr))
            self.massfraction.setEnabled(False)
        else:
            self.element.setValid(False)
            self.density.setEnabled(True)
            self.molarmass.setEnabled(True)
            self.massfraction.setEnabled(True)

    def clearInputs(self) -> None:
        self.blockSignals(True)
        self.element.setText("")
        self.density.setValue(None)
        self.molarmass.setValue(None)
        self.massfraction.setText("1.0")
        self.blockSignals(False)

    def clearOutputs(self) -> None:
        self.count.setText("")
        self.background_count.setText("")
        self.lod_count.setText("")

    def updateOutputs(
        self,
        responses: np.ndarray,
        detections: np.ndarray,
        labels: np.ndarray,
        limits: Tuple[str, Dict[str, float], np.ndarray],
    ) -> None:
        background = np.mean(responses[labels == 0])
        background_std = np.std(responses[labels == 0])
        lod = np.mean(limits[2]["ld"])  # + self.background

        self.count.setText(f"{detections.size} ± {np.sqrt(detections.size):.1f}")
        self.background_count.setText(f"{background:.4g} ± {background_std:.4g}")
        self.lod_count.setText(
            f"{lod:.4g} ({limits[0]}, {','.join(f'{k}={v}' for k,v in limits[1].items())})"
        )

    def isComplete(self) -> bool:
        return (
            self.density.hasAcceptableInput() and self.massfraction.hasAcceptableInput()
        )


class ReferenceIOWidget(SampleIOWidget):
    def __init__(self, name: str, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(name, parent=parent)

        concentration_units = {
            "fg/L": 1e-18,
            "pg/L": 1e-15,
            "ng/L": 1e-12,
            "μg/L": 1e-9,
            "mg/L": 1e-6,
            "g/L": 1e-3,
            "kg/L": 1.0,
        }
        self.concentration = UnitsWidget(
            units=concentration_units,
            default_unit="ng/L",
            invalid_color=QtGui.QColor(255, 255, 172),
        )
        self.diameter = UnitsWidget(
            {"nm": 1e-9, "μm": 1e-6, "m": 1.0},
            default_unit="nm",
        )

        self.concentration.setToolTip("Reference particle concentration.")
        self.diameter.setToolTip("Reference particle diameter.")

        self.concentration.valueChanged.connect(
            lambda: self.optionsChanged.emit(self.name)
        )
        self.diameter.valueChanged.connect(lambda: self.optionsChanged.emit(self.name))

        self.inputs.layout().removeRow(self.molarmass)
        self.inputs.layout().insertRow(0, "Concentration:", self.concentration)
        self.inputs.layout().insertRow(1, "Diameter:", self.diameter)

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
        response: Optional[float],
        time: float,
        uptake: Optional[float],
    ) -> None:
        self.efficiency.setText("")
        self.massresponse.setValue("")

        density = self.density.baseValue()
        diameter = self.diameter.baseValue()
        if density is None or diameter is None:
            return

        mass = spcal.reference_particle_mass(density, diameter)
        massfraction = (
            float(self.massfraction.text())
            if self.massfraction.hasAcceptableInput()
            else None
        )
        if massfraction is not None:
            self.massresponse.setBaseValue(mass * massfraction / np.mean(detections))

        # If concentration defined use conc method
        concentration = self.concentration.baseValue()
        if concentration is not None and uptake is not None:
            efficiency = spcal.nebulisation_efficiency_from_concentration(
                detections.size,
                concentration=concentration,
                mass=mass,
                flowrate=uptake,
                time=time,
            )
            self.efficiency.setText(f"{efficiency:.4g}")
        elif massfraction is not None and uptake is not None and response is not None:
            efficiency = spcal.nebulisation_efficiency_from_mass(
                detections,
                dwell=dwell,
                mass=mass,
                flowrate=uptake,
                response_factor=response,
                mass_fraction=massfraction,
            )
            self.efficiency.setText(f"{efficiency:.4g}")

    def isComplete(self) -> bool:
        return super().isComplete() and self.diameter.hasAcceptableInput()


class ResultIOWidget(IOWidget):
    optionsChanged = QtCore.Signal(str)

    signal_units = {"counts": 1.0}
    size_units = {"nm": 1e-9, "μm": 1e-6, "m": 1.0}
    mass_units = {
        "ag": 1e-21,
        "fg": 1e-18,
        "pg": 1e-15,
        "ng": 1e-12,
        "μg": 1e-9,
        "g": 1e-3,
        "kg": 1.0,
    }
    molar_concentration_units = {
        "amol/L": 1e-18,
        "fmol/L": 1e-15,
        "pmol/L": 1e-12,
        "nmol/L": 1e-9,
        "μmol/L": 1e-6,
        "mmol/L": 1e-3,
        "mol/L": 1.0,
    }
    concentration_units = {
        "fg/L": 1e-18,
        "pg/L": 1e-15,
        "ng/L": 1e-12,
        "μg/L": 1e-9,
        "mg/L": 1e-6,
        "g/L": 1e-3,
        "kg/L": 1.0,
    }

    def __init__(self, name: str, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.name = name

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
            self.concentration_units,
            default_unit="ng/L",
        )
        self.conc.setReadOnly(True)
        self.background = UnitsWidget(
            self.concentration_units,
            default_unit="ng/L",
        )
        self.background.setReadOnly(True)

        self.lod = UnitsWidget(
            self.size_units,
            default_unit="nm",
        )
        self.lod.setReadOnly(True)
        self.mean = UnitsWidget(
            self.size_units,
            default_unit="nm",
        )
        self.mean.setReadOnly(True)
        self.median = UnitsWidget(
            self.size_units,
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
        count_error: float,
        conc: Optional[float] = None,
        number_conc: Optional[float] = None,
        background_conc: Optional[float] = None,
        background_error: Optional[float] = None,
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
        self.count.setText(f"{count} ± {count_error:.1f}")
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
    optionsChanged = QtCore.Signal(str)

    def __init__(
        self,
        io_widget_type: Type[IOType],
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)

        self.io_widget_type = io_widget_type

        self.combo_name = QtWidgets.QComboBox()
        self.stack = QtWidgets.QStackedWidget()

        self.combo_name.currentIndexChanged.connect(self.stack.setCurrentIndex)

        self.layout_top = QtWidgets.QHBoxLayout()
        self.layout_top.addWidget(self.combo_name, 0, QtCore.Qt.AlignRight)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.layout_top)
        layout.addWidget(self.stack, 1)
        self.setLayout(layout)

    def __getitem__(self, name: str) -> IOType:
        return self.stack.widget(self.combo_name.findText(name))  # type: ignore

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
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(SampleIOWidget, parent=parent)


class ReferenceIOStack(IOStack[ReferenceIOWidget]):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(ReferenceIOWidget, parent=parent)


class ResultIOStack(IOStack[ResultIOWidget]):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(ResultIOWidget, parent=parent)