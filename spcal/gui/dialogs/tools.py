from typing import Generator

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.isotope import SPCalIsotopeBase
from spcal.particle import (
    nebulisation_efficiency_from_concentration,
    nebulisation_efficiency_from_mass,
    reference_particle_mass,
)
from spcal.gui.modelviews.models import NumpyRecArrayTableModel, SearchColumnsProxyModel
from spcal.gui.widgets.values import ValueWidget
from spcal.gui.widgets.units import UnitsWidget
from spcal.npdb import db
from spcal.processing.options import SPCalIsotopeOptions
from spcal.processing.result import SPCalProcessingResult
from spcal.siunits import (
    density_units,
    mass_units,
    mass_concentration_units,
    size_units,
    response_units,
)


class FormulaValidator(QtGui.QValidator):
    def __init__(
        self, regex: QtCore.QRegularExpression, parent: QtCore.QObject | None = None
    ):
        super().__init__(parent)
        self.regex = regex

    def validate(self, input: str, _: int) -> QtGui.QValidator.State:
        iter = self.regex.globalMatch(input)
        if len(input) == 0:
            return QtGui.QValidator.State.Acceptable
        if not str.isalnum(input.replace(".", "")):
            return QtGui.QValidator.State.Invalid
        if not iter.hasNext():  # no match
            return QtGui.QValidator.State.Intermediate
        while iter.hasNext():
            match = iter.next()
            if match.captured(1) not in db["elements"]["Symbol"]:
                return QtGui.QValidator.State.Intermediate
        return QtGui.QValidator.State.Acceptable


class MassFractionCalculatorDialog(QtWidgets.QDialog):
    ratiosChanged = QtCore.Signal()
    ratiosSelected = QtCore.Signal(list)
    molarMassSelected = QtCore.Signal(float)

    def __init__(self, formula: str = "", parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Mass Fraction Calculator")
        self.resize(300, 120)

        self.regex = QtCore.QRegularExpression("([A-Z][a-z]?)([0-9\\.]*)")
        self.ratios: dict[str, float] = {}
        self.mw = 0.0

        self.lineedit_formula = QtWidgets.QLineEdit(formula)
        self.lineedit_formula.setPlaceholderText("Molecular Formula")
        self.lineedit_formula.setValidator(FormulaValidator(self.regex))
        self.lineedit_formula.textChanged.connect(self.recalculate)

        self.label_mw = QtWidgets.QLabel("MW = 0 g/mol")

        self.textedit_ratios = QtWidgets.QTextEdit()
        self.textedit_ratios.setReadOnly(True)
        self.textedit_ratios.setFont(QtGui.QFont("Courier"))

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )

        self.ratiosChanged.connect(self.updateLabels)
        self.ratiosChanged.connect(self.completeChanged)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.lineedit_formula, 0)
        layout.addWidget(self.label_mw, 0)
        layout.addWidget(self.textedit_ratios, 1)
        layout.addWidget(self.button_box, 0)

        self.setLayout(layout)
        self.completeChanged()

    def accept(self):
        # dict order messed up during signal, send as list of tuples
        ratios = [(k, v) for k, v in self.ratios.items()]
        self.ratiosSelected.emit(ratios)
        self.molarMassSelected.emit(self.mw)
        super().accept()

    def completeChanged(self):
        complete = self.isComplete()
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            complete
        )

    def isComplete(self) -> bool:
        return len(self.ratios) > 0

    def recalculate(self):
        """Calculates the mass fraction of each valid element in the formula."""
        self.ratios = {}
        elements = db["elements"]
        for element, number in self.searchFormula():
            idx = np.flatnonzero(elements["Symbol"] == element)
            if idx.size > 0:
                ratio = elements["MW"][idx[0]] * float(number or 1.0)
                self.ratios[element] = self.ratios.get(element, 0.0) + ratio
        self.mw = sum(self.ratios.values())
        for element in self.ratios:
            self.ratios[element] = self.ratios[element] / self.mw
        self.ratiosChanged.emit()

    def searchFormula(self) -> Generator[tuple[str, float], None, None]:
        iter = self.regex.globalMatch(self.lineedit_formula.text())
        while iter.hasNext():
            match = iter.next()
            yield match.captured(1), float(match.captured(2) or 1.0)

    def updateLabels(self):
        self.textedit_ratios.setPlainText("")
        if len(self.ratios) == 0:
            return
        text = "<html>"
        for i, (element, ratio) in enumerate(self.ratios.items()):
            if i == 0:
                text += "<b>"
            text += f"{element:<2}&nbsp;{ratio:.4f}&nbsp;&nbsp;"
            if i == 0:
                text += "</b>"
            if i % 3 == 2:
                text += "<br>"
        text += "</html>"
        self.textedit_ratios.setText(text)

        self.label_mw.setText(f"MW = {self.mw:.4g} g/mol")


class ParticleDatabaseDialog(QtWidgets.QDialog):
    densitySelected = QtCore.Signal(float)
    formulaSelected = QtCore.Signal(str)

    def __init__(self, formula: str = "", parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Density Database")
        self.resize(800, 600)

        self.lineedit_search = QtWidgets.QLineEdit(formula)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.model = NumpyRecArrayTableModel(
            np.concatenate((db["inorganic"], db["polymer"])),
            name_formats={"Density": "{:.4g}"},
        )
        self.proxy = SearchColumnsProxyModel([0, 1])
        self.proxy.setSourceModel(self.model)

        self.table = QtWidgets.QTableView()
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustToContentsOnFirstShow
        )
        self.table.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table.setModel(self.proxy)
        self.table.setColumnHidden(4, True)

        self.lineedit_search.textChanged.connect(self.searchDatabase)
        self.lineedit_search.textChanged.connect(self.table.clearSelection)
        self.table.pressed.connect(self.completeChanged)
        self.table.doubleClicked.connect(self.accept)
        self.proxy.rowsInserted.connect(self.completeChanged)
        self.proxy.rowsRemoved.connect(self.completeChanged)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Search"), 0)
        layout.addWidget(self.lineedit_search, 0)
        layout.addWidget(self.table)
        layout.addWidget(self.button_box, 0)

        self.setLayout(layout)
        self.completeChanged()

    def searchDatabase(self, string: str):
        self.proxy.setSearchString(string)

    def isComplete(self) -> bool:
        return len(self.table.selectedIndexes()) > 0

    def completeChanged(self):
        complete = self.isComplete()
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            complete
        )

    def accept(self):
        idx = self.table.selectedIndexes()
        self.densitySelected.emit(1000.0 * float(self.proxy.data(idx[3])))
        self.formulaSelected.emit(self.proxy.data(idx[0]))
        super().accept()


class TransportEfficiencyDialog(QtWidgets.QDialog):
    efficencyChanged = QtCore.Signal(object)
    massResponseChanged = QtCore.Signal(object)
    efficencySelected = QtCore.Signal(float)
    massResponseSelected = QtCore.Signal(float)

    isotopeOptionsChanged = QtCore.Signal(SPCalIsotopeBase, SPCalIsotopeOptions)

    def __init__(
        self,
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

        self.diameter = UnitsWidget(size_units, "nm", options.diameter, sigfigs=sf)
        self.density = UnitsWidget(density_units, "g/cm³", options.density, sigfigs=sf)
        self.concentration = UnitsWidget(
            mass_concentration_units, "µg/L", options.concentration, sigfigs=sf
        )
        self.response = UnitsWidget(
            response_units, "L/µg", options.response, sigfigs=sf
        )
        self.mass_fraction = ValueWidget(
            options.mass_fraction, step=0.1, max=1.0, sigfigs=sf
        )

        self.diameter.baseValueChanged.connect(self.onOptionChanged)
        self.density.baseValueChanged.connect(self.onOptionChanged)
        self.concentration.baseValueChanged.connect(self.onOptionChanged)
        self.response.baseValueChanged.connect(self.onOptionChanged)
        self.mass_fraction.valueChanged.connect(self.onOptionChanged)

        self.efficiency = ValueWidget(sigfigs=sf)
        self.efficiency.setReadOnly(True)
        self.mass_response = UnitsWidget(mass_units, "ag", sigfigs=sf)
        self.mass_response.setReadOnly(True)

        self.efficencyChanged.connect(self.efficiency.setValue)
        self.massResponseChanged.connect(self.mass_response.setValue)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        gbox_mass = QtWidgets.QGroupBox("Reference properties")
        gbox_mass_layout = QtWidgets.QFormLayout()
        gbox_mass_layout.addRow("Density", self.density)
        gbox_mass_layout.addRow("Response", self.response)
        gbox_mass_layout.addRow("Mass Fraction", self.mass_fraction)
        gbox_mass_layout.addRow("Diameter", self.diameter)
        gbox_mass.setLayout(gbox_mass_layout)

        gbox_conc = QtWidgets.QGroupBox("")
        gbox_conc_layout = QtWidgets.QFormLayout()
        gbox_conc_layout.addRow("Concentration", self.concentration)
        gbox_conc.setLayout(gbox_conc_layout)

        gbox_output = QtWidgets.QGroupBox("Calculated")
        gbox_ouput_layout = QtWidgets.QFormLayout()
        gbox_ouput_layout.addRow("Efficiency", self.efficiency)
        gbox_ouput_layout.addRow("Mass Response", self.mass_response)
        gbox_output.setLayout(gbox_ouput_layout)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(gbox_mass, 0, 0, 1, 1)
        layout.addWidget(gbox_conc, 1, 0, 1, 1)
        layout.addWidget(gbox_output, 0, 1, 2, 1)
        layout.addWidget(self.button_box, 2, 0, 1, 2)

        self.setLayout(layout)
        self.onOptionChanged()

    def onOptionChanged(self):
        self.updateEfficiency()
        self.completeChanged()

    def updateEfficiency(self):
        density = self.density.baseValue()
        diameter = self.diameter.baseValue()

        if density is None or diameter is None:
            return None

        reference_mass = reference_particle_mass(density, diameter)

        mass_fraction = self.mass_fraction.value()
        if mass_fraction is not None:
            mass_response = float(
                reference_mass
                * mass_fraction
                / np.mean(self.proc_result.calibrated("signal"))
            )
            self.massResponseChanged.emit(mass_response)

        concentration = self.concentration.baseValue()
        uptake = self.proc_result.method.instrument_options.uptake
        response = self.response.baseValue()

        if concentration is not None and uptake is not None:
            eff = nebulisation_efficiency_from_concentration(
                self.proc_result.number,
                concentration=concentration,
                mass=reference_mass,
                flow_rate=uptake,
                time=self.proc_result.total_time,
            )
        elif mass_fraction is not None and response is not None and uptake is not None:
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

    def isComplete(self) -> bool:
        return (
            self.efficiency.value() is not None
            or self.mass_response.baseValue() is not None
        )

    def completeChanged(self):
        complete = self.isComplete()
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            complete
        )

    def accept(self):
        self.efficencySelected.emit(self.efficiency.value())
        self.massResponseSelected.emit(self.mass_response.baseValue())

        # Set any changed
        options = self.proc_result.method.isotope_options[self.proc_result.isotope]
        new_options = SPCalIsotopeOptions(
            self.density.baseValue(),
            self.response.baseValue(),
            self.mass_fraction.value(),
            self.concentration.baseValue(),
            self.diameter.baseValue(),
            self.mass_response.baseValue(),
        )
        if options != new_options:
            self.proc_result.method.isotope_options[self.proc_result.isotope] = new_options
            self.isotopeOptionsChanged.emit(self.proc_result.isotope, new_options)

        super().accept()
