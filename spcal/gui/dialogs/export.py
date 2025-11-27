import logging
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.datafile import SPCalDataFile
from spcal.gui.io import get_save_spcal_path
from spcal.io.export import export_single_particle_results
from spcal.isotope import SPCalIsotope
from spcal.processing.result import SPCalProcessingResult
from spcal.siunits import mass_units, size_units

logger = logging.getLogger(__name__)


class ExportDialog(QtWidgets.QDialog):
    invalid_chars = '<>:"/\\|?*'

    def __init__(
        self,
        data_files: list[SPCalDataFile],
        results: dict[SPCalDataFile, dict[SPCalIsotope, SPCalProcessingResult]],
        clusters: dict[SPCalDataFile, np.ndarray],
        path: Path | None = None,
        units: dict[str, tuple[str, float]] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Results Export Options")

        self.data_files = data_files
        self.results = results
        self.clusters = clusters

        _units = {"mass": "kg", "size": "m", "volume": "L"}
        if units is not None:
            _units.update({k: v[0] for k, v in units.items()})

        filename_regexp = QtCore.QRegularExpression(f"[^{self.invalid_chars}]+")

        self.lineedit_name = QtWidgets.QLineEdit("<DataFile>_results.csv")
        self.lineedit_name.setMinimumWidth(300)
        self.lineedit_name.setValidator(
            QtGui.QRegularExpressionValidator(filename_regexp)
        )

        self.lineedit_dir = QtWidgets.QLineEdit(str(self.data_files[0].path.parent))
        self.lineedit_dir.setMinimumWidth(300)

        self.button_path = QtWidgets.QPushButton("Directory")
        self.button_path.clicked.connect(self.dialogDirectory)

        file_box = QtWidgets.QGroupBox("Save File")
        file_box_layout = QtWidgets.QHBoxLayout()
        file_box_layout.addWidget(self.lineedit_name, 1)
        file_box_layout.addWidget(self.button_path, 0)
        file_box.setLayout(file_box_layout)

        self.mass_units = QtWidgets.QComboBox()
        self.mass_units.addItems(list(mass_units.keys()))
        self.mass_units.setCurrentText(_units["mass"])
        self.size_units = QtWidgets.QComboBox()
        self.size_units.addItems(list(size_units.keys()))
        self.size_units.setCurrentText(_units["size"])

        units_box = QtWidgets.QGroupBox("Output Units")
        units_box_layout = QtWidgets.QFormLayout()
        units_box_layout.addRow("Mass units", self.mass_units)
        units_box_layout.addRow("Size units", self.size_units)
        units_box.setLayout(units_box_layout)

        self.check_export_inputs = QtWidgets.QCheckBox("Export options and inputs.")
        self.check_export_inputs.setChecked(True)
        self.check_export_arrays = QtWidgets.QCheckBox(
            "Export particle detection data arrays."
        )
        self.check_export_arrays.setChecked(True)
        self.check_export_compositions = QtWidgets.QCheckBox(
            "Export peak compositions."
        )

        switches_box = QtWidgets.QGroupBox("Export options")
        switches_box_layout = QtWidgets.QVBoxLayout()
        switches_box_layout.addWidget(self.check_export_inputs)
        switches_box_layout.addWidget(self.check_export_arrays)
        switches_box_layout.addWidget(self.check_export_compositions)
        switches_box.setLayout(switches_box_layout)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(file_box)
        layout.addWidget(units_box)
        layout.addWidget(switches_box)
        layout.addWidget(self.button_box, 0)

        self.setLayout(layout)

    def dialogDirectory(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Directory", self.lineedit_dir.text()
        )
        if dir == "":
            return
        self.lineedit_dir.setText(dir)

    def accept(self):
        units = {
            "mass": (
                self.mass_units.currentText(),
                mass_units[self.mass_units.currentText()],
            ),
            "size": (
                self.size_units.currentText(),
                size_units[self.size_units.currentText()],
            ),
        }

        # path = Path(self.lineedit_path.text())
        # first_result = next(iter(self.results.values()))
        path = get_save_spcal_path(self, [("CSV Documents", ".csv")])

        # Check if overwriting
        if path.exists():
            button = QtWidgets.QMessageBox.warning(
                self,
                "Overwrite File?",
                f"The file '{path.name}' already exists, do you want to overwrite it?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if button == QtWidgets.QMessageBox.StandardButton.No:
                return

        # export_single_particle_results(
        #     path,
        #     self.results,
        #     self.clusters,
        #     self.times,
        #     units_for_results=units,
        #     output_inputs=self.check_export_inputs.isChecked(),
        #     output_compositions=self.check_export_compositions.isChecked(),
        #     output_arrays=self.check_export_arrays.isChecked(),
        # )
        logger.info(f"Data for {first_result.file} exported to {path}.")
        super().accept()
