from pathlib import Path
from typing import Dict, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.units import mass_units, molar_concentration_units, size_units
from spcal.io.text import export_single_particle_results
from spcal.result import SPCalResult


class ExportDialog(QtWidgets.QDialog):
    invalid_chars = '<>:"/\\|?*'

    def __init__(
        self,
        path: str | Path,
        results: Dict[str, SPCalResult],
        units: Dict[str, Tuple[str, float]] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Results Export Options")

        self.results = results

        _units = {"mass": "kg", "size": "m", "cell_concentration": "mol/L"}
        if units is not None:
            _units.update({k: v[0] for k, v in units.items()})

        filename_regexp = QtCore.QRegularExpression(f"[^{self.invalid_chars}]+")
        self.lineedit_path = QtWidgets.QLineEdit(str(Path(path).absolute()))
        self.lineedit_path.setMinimumWidth(300)
        self.lineedit_path.setValidator(
            QtGui.QRegularExpressionValidator(filename_regexp)
        )
        self.button_path = QtWidgets.QPushButton("Select File")
        self.button_path.clicked.connect(self.dialogFilePath)

        file_box = QtWidgets.QGroupBox("Save File")
        file_box.setLayout(QtWidgets.QHBoxLayout())
        file_box.layout().addWidget(self.lineedit_path, 1)
        file_box.layout().addWidget(self.button_path, 0)

        self.mass_units = QtWidgets.QComboBox()
        self.mass_units.addItems(mass_units.keys())
        self.mass_units.setCurrentText(_units["mass"])
        self.size_units = QtWidgets.QComboBox()
        self.size_units.addItems(size_units.keys())
        self.size_units.setCurrentText(_units["size"])
        self.conc_units = QtWidgets.QComboBox()
        self.conc_units.addItems(molar_concentration_units.keys())
        self.conc_units.setCurrentText(_units["cell_concentration"])

        units_box = QtWidgets.QGroupBox("Output Units")
        units_box.setLayout(QtWidgets.QFormLayout())
        units_box.layout().addRow("Mass units", self.mass_units)
        units_box.layout().addRow("Size units", self.size_units)
        units_box.layout().addRow("Conc. units", self.conc_units)

        self.check_export_inputs = QtWidgets.QCheckBox("Export options and inputs.")
        self.check_export_inputs.setChecked(True)
        self.check_export_arrays = QtWidgets.QCheckBox(
            "Export detected particle arrays."
        )
        self.check_export_arrays.setChecked(True)
        self.check_export_compositions = QtWidgets.QCheckBox(
            "Export peak compositions."
        )

        switches_box = QtWidgets.QGroupBox("Export options")
        switches_box.setLayout(QtWidgets.QVBoxLayout())
        switches_box.layout().addWidget(self.check_export_inputs)
        switches_box.layout().addWidget(self.check_export_arrays)
        switches_box.layout().addWidget(self.check_export_compositions)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(file_box)
        layout.addWidget(units_box)
        layout.addWidget(switches_box)
        layout.addWidget(self.button_box, 0)

        self.setLayout(layout)

    def dialogFilePath(self) -> QtWidgets.QFileDialog:
        dlg = QtWidgets.QFileDialog(
            self,
            "Save Results",
            self.lineedit_path.text(),
            "CSV Documents (*.csv);;All files (*)",
        )
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dlg.setFileMode(QtWidgets.QFileDialog.FileMode.AnyFile)
        dlg.fileSelected.connect(self.lineedit_path.setText)
        dlg.open()

    def accept(self) -> None:
        units = {
            "mass": (
                self.mass_units.currentText(),
                mass_units[self.mass_units.currentText()],
            ),
            "size": (
                self.size_units.currentText(),
                size_units[self.size_units.currentText()],
            ),
            "conc": (
                self.conc_units.currentText(),
                molar_concentration_units[self.conc_units.currentText()],
            ),
        }

        export_single_particle_results(
            self.lineedit_path.text(),
            self.results,
            units_for_results=units,
            output_inputs=self.check_export_inputs.isChecked(),
            output_compositions=self.check_export_compositions.isChecked(),
            output_arrays=self.check_export_arrays.isChecked(),
        )
        super().accept()
