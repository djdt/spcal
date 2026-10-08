import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.datafile import SPCalDataFile
from spcal.gui.dialogs.tools import MassFractionCalculatorDialog, ParticleDatabaseDialog
from spcal.gui.modelviews.massfraction import MassFractionValidator
from spcal.gui.objects import ContextMenuRedirectFilter
from spcal.gui.util import create_action
from spcal.gui.widgets.units import UnitsWidget
from spcal.gui.widgets.values import ValueWidget
from spcal.isotope import SPCalIsotopeBase
from spcal.particle import (
    nebulisation_efficiency_from_mass,
    nebulisation_efficiency_from_mass_concentration,
    nebulisation_efficiency_from_number_concentration,
    reference_particle_mass,
)
from spcal.processing.options import SPCalIsotopeOptions
from spcal.processing.result import SPCalProcessingResult
from spcal.siunits import (
    density_units,
    flowrate_units,
    mass_concentration_units,
    mass_units,
    number_concentration_units,
    response_units,
    size_units,
)


class DensityWidget(UnitsWidget):
    def __init__(
        self,
        density: float | None,
        sigfigs: int = 4,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(
            density_units, "g/cm³", density, sigfigs=sigfigs, parent=parent
        )
        self._value.lineEdit().installEventFilter(ContextMenuRedirectFilter(self))
        self.action_density_lookup = create_action(
            "folder-database",
            "Lookup Density",
            "Lookup and select a density from the database",
            self.dialogParticleDatabase,
        )

    def dialogParticleDatabase(self):
        dlg = ParticleDatabaseDialog(parent=self)
        dlg.densitySelected.connect(self.setBaseValue)
        dlg.open()
        return dlg

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        menu = self._value.lineEdit().createStandardContextMenu()
        menu.insertAction(menu.actions()[0], self.action_density_lookup)
        menu.insertSeparator(menu.actions()[1])
        menu.popup(event.globalPos())


class MassFractionWidget(ValueWidget):
    def __init__(
        self,
        mass_fraction: float | None,
        sigfigs: int = 4,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(
            mass_fraction, step=0.1, max=1.0, sigfigs=sigfigs, parent=parent
        )
        self.lineEdit().setValidator(MassFractionValidator(sigfigs))
        self.lineEdit().installEventFilter(ContextMenuRedirectFilter(self))
        self.action_fraction_calc = create_action(
            "folder-calculate",
            "Calculate Mass Fraction",
            "Input a molecular formula to calculate the mass fraction",
            self.dialogMassFractionCalculator,
        )

    def dialogMassFractionCalculator(self) -> QtWidgets.QDialog:
        def set_major_ratio(ratios: list):
            self.setValue(float(ratios[0][1]))

        dlg = MassFractionCalculatorDialog(parent=self)
        dlg.ratiosSelected.connect(set_major_ratio)
        dlg.open()
        return dlg

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        menu = self.lineEdit().createStandardContextMenu()
        menu.insertAction(menu.actions()[0], self.action_fraction_calc)
        menu.insertSeparator(menu.actions()[1])
        menu.popup(event.globalPos())


class TransportEfficiencyDialog(QtWidgets.QDialog):
    efficencyChanged = QtCore.Signal(object)
    massResponseChanged = QtCore.Signal(object)
    efficiencySelected = QtCore.Signal(object)
    massResponseSelected = QtCore.Signal(object)

    uptakeChanged = QtCore.Signal(object)
    isotopeOptionsChanged = QtCore.Signal(SPCalIsotopeBase, SPCalIsotopeOptions)

    def __init__(
        self,
        data_file: SPCalDataFile,
        isotope: SPCalIsotopeBase,
        result: SPCalProcessingResult,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Transport Efficiency Calculator")
        self.setMinimumWidth(600)

        if result.number == 0:
            raise ValueError("unable to calculate efficiency, no particles detected")

        self.proc_result = result
        options = result.method.isotope_options[result.isotope]

        sf = int(QtCore.QSettings().value("SigFigs", 4))  # type: ignore

        self.uptake = UnitsWidget(
            flowrate_units,
            "ml/min",
            self.proc_result.method.instrument_options.uptake,
            sigfigs=sf,
        )

        self.diameter = UnitsWidget(size_units, "nm", options.diameter, sigfigs=sf)
        self.density = DensityWidget(options.density, sigfigs=sf)
        self.response = UnitsWidget(
            response_units, "L/µg", options.response, sigfigs=sf
        )
        self.mass_fraction = MassFractionWidget(options.mass_fraction, sigfigs=sf)

        self.mass_concentration = UnitsWidget(
            mass_concentration_units, "µg/L", options.concentration, sigfigs=sf
        )
        self.number_concentration = UnitsWidget(
            number_concentration_units, default_unit="#/ml", base_value=None, sigfigs=sf
        )

        self.diameter.baseValueChanged.connect(self.onOptionChanged)
        self.density.baseValueChanged.connect(self.onOptionChanged)
        self.response.baseValueChanged.connect(self.onOptionChanged)
        self.mass_fraction.valueChanged.connect(self.onOptionChanged)
        self.uptake.baseValueChanged.connect(self.onOptionChanged)

        self.mass_concentration.baseValueChanged.connect(self.onOptionChanged)
        self.number_concentration.baseValueChanged.connect(self.onOptionChanged)

        self.efficiency = ValueWidget(sigfigs=sf)
        self.efficiency.setReadOnly(True)
        self.mass_response = UnitsWidget(mass_units, "ag", sigfigs=sf)
        self.mass_response.setReadOnly(True)

        self.efficencyChanged.connect(self.efficiency.setValue)
        self.massResponseChanged.connect(self.mass_response.setBaseValue)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        gbox_info = QtWidgets.QGroupBox("Data file")
        gbox_info_layout = QtWidgets.QFormLayout()
        gbox_info_layout.addRow("Name:", QtWidgets.QLabel(str(data_file.path.name)))
        gbox_info_layout.addRow("Isotope:", QtWidgets.QLabel(str(isotope)))
        gbox_info_layout.addRow("No events:", QtWidgets.QLabel(str(result.number)))
        gbox_info.setLayout(gbox_info_layout)

        gbox_inst = QtWidgets.QGroupBox("Instrument options")
        gbox_inst_layout = QtWidgets.QFormLayout()
        gbox_inst_layout.addRow("Uptake", self.uptake)
        gbox_inst.setLayout(gbox_inst_layout)

        gbox_mass = QtWidgets.QGroupBox("Reference properties")
        gbox_mass_layout = QtWidgets.QFormLayout()
        gbox_mass_layout.addRow("Density", self.density)
        gbox_mass_layout.addRow("Response", self.response)
        gbox_mass_layout.addRow("Mass Fraction", self.mass_fraction)
        gbox_mass_layout.addRow("Diameter", self.diameter)
        gbox_mass.setLayout(gbox_mass_layout)

        self.gbox_conc = QtWidgets.QGroupBox("Number method")
        self.gbox_conc.setCheckable(True)
        self.gbox_conc.setChecked(False)
        self.gbox_conc.clicked.connect(self.onNumberMethodEnabled)
        gbox_conc_layout = QtWidgets.QFormLayout()
        gbox_conc_layout.addRow("Mass Conc.", self.mass_concentration)
        gbox_conc_layout.addRow("Number Conc.", self.number_concentration)
        self.gbox_conc.setLayout(gbox_conc_layout)

        gbox_output = QtWidgets.QGroupBox("Calculated")
        gbox_ouput_layout = QtWidgets.QFormLayout()
        gbox_ouput_layout.addRow("Efficiency", self.efficiency)
        gbox_ouput_layout.addRow("Mass Response", self.mass_response)
        gbox_output.setLayout(gbox_ouput_layout)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(gbox_info, 0, 0, 1, 2)
        layout.addWidget(gbox_inst, 1, 0, 1, 2)
        layout.addWidget(gbox_mass, 2, 0, 1, 1)
        layout.addWidget(self.gbox_conc, 3, 0, 1, 1)
        layout.addWidget(gbox_output, 2, 1, 2, 1)
        layout.addWidget(self.button_box, 4, 0, 1, 2)

        self.setLayout(layout)
        self.onOptionChanged()

    def onOptionChanged(self):
        self.updateEfficiency()
        self.completeChanged()

    def onNumberMethodEnabled(self):
        if self.gbox_conc.isChecked():
            button = QtWidgets.QMessageBox.warning(
                self,
                "Use Number Method?",
                "Calibration using the number method requires a reference particle solution with a certified number or mass concentration.\n"
                "Are you sure use wish to use this method?",
                buttons=QtWidgets.QMessageBox.StandardButton.Ok
                | QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if button == QtWidgets.QMessageBox.StandardButton.Cancel:
                self.gbox_conc.setChecked(False)

    def updateEfficiency(self):
        density = self.density.baseValue()
        diameter = self.diameter.baseValue()
        mass_fraction = self.mass_fraction.value()

        mass_conc = self.mass_concentration.baseValue()
        number_conc = self.number_concentration.baseValue()
        uptake = self.uptake.baseValue()
        response = self.response.baseValue()

        if number_conc is not None and uptake is not None:
            eff = nebulisation_efficiency_from_number_concentration(
                self.proc_result.number,
                number_concentration=number_conc,
                flow_rate=uptake,
                time=self.proc_result.total_time,
            )
        elif (
            mass_conc is not None
            and uptake is not None
            and density is not None
            and diameter is not None
        ):
            reference_mass = reference_particle_mass(density, diameter)
            eff = nebulisation_efficiency_from_mass_concentration(
                self.proc_result.number,
                mass_concentration=mass_conc,
                mass=reference_mass,
                flow_rate=uptake,
                time=self.proc_result.total_time,
            )
        elif (
            mass_fraction is not None
            and response is not None
            and uptake is not None
            and diameter is not None
            and density is not None
        ):
            reference_mass = reference_particle_mass(density, diameter)
            eff = nebulisation_efficiency_from_mass(
                self.proc_result.calibrated("signal"),
                dwell=self.proc_result.event_time,
                mass=reference_mass,
                flow_rate=uptake,
                response_factor=response,
                mass_fraction=mass_fraction,
            )
        else:
            eff = None
        self.efficencyChanged.emit(eff)

        if density is not None and diameter is not None and mass_fraction is not None:
            mass_response = float(
                reference_particle_mass(density, diameter)
                * mass_fraction
                / np.mean(self.proc_result.calibrated("signal"))
            )
        else:
            mass_response = None
        self.massResponseChanged.emit(mass_response)

    def isComplete(self) -> bool:
        efficiency = self.efficiency.value()
        return (
            bool(efficiency is not None and efficiency < 1.0)
            or self.mass_response.baseValue() is not None
        )

    def completeChanged(self):
        complete = self.isComplete()
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            complete
        )

    def accept(self):
        self.efficiencySelected.emit(self.efficiency.value())
        self.massResponseSelected.emit(self.mass_response.baseValue())

        # Set any changed
        options = self.proc_result.method.isotope_options[self.proc_result.isotope]
        new_options = SPCalIsotopeOptions(
            self.density.baseValue(),
            self.response.baseValue(),
            self.mass_fraction.value(),
            self.mass_concentration.baseValue(),
            self.diameter.baseValue(),
            self.mass_response.baseValue(),
        )
        if options != new_options:
            self.proc_result.method.isotope_options[self.proc_result.isotope] = (
                new_options
            )
            self.isotopeOptionsChanged.emit(self.proc_result.isotope, new_options)
        if self.uptake.baseValue() != self.proc_result.method.instrument_options.uptake:
            self.uptakeChanged.emit(self.uptake.baseValue())

        super().accept()
