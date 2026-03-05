import logging
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.datafile import SPCalDataFile
from spcal.io.export import export_spcal_processing_results
from spcal.isotope import SPCalIsotopeBase
from spcal.processing.result import SPCalProcessingResult
from spcal.siunits import mass_units, size_units

logger = logging.getLogger(__name__)


class ExportDialog(QtWidgets.QDialog):
    INVALID_CHARS = '<>:"|?*'

    def __init__(
        self,
        data_file: SPCalDataFile,
        results: dict[SPCalIsotopeBase, SPCalProcessingResult],
        clusters: dict[str, np.ndarray],
        path: Path | None = None,
        units: dict[str, tuple[str, float]] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Export Results")

        self.data_file = data_file
        self.results = results
        self.clusters = clusters

        if path is None:
            path = data_file.path.with_name(data_file.path.stem + "_spcal_results.csv")

        _units = {"mass": "fg", "size": "nm"}
        if units is not None:
            _units.update({k: v[0] for k, v in units.items()})

        filename_regexp = QtCore.QRegularExpression(
            f"[^{ExportDialog.INVALID_CHARS}]+.csv"
        )

        self.lineedit_path = QtWidgets.QLineEdit(str(path))
        self.lineedit_path.setMinimumWidth(300)
        self.lineedit_path.setValidator(
            QtGui.QRegularExpressionValidator(filename_regexp)
        )
        self.lineedit_path.textChanged.connect(self.completeChanged)

        self.button_path = QtWidgets.QPushButton("Select...")
        self.button_path.clicked.connect(self.dialogExportPath)

        path_box = QtWidgets.QGroupBox("Export path")
        path_box_layout = QtWidgets.QVBoxLayout()
        path_box_layout.addWidget(self.lineedit_path)
        path_box_layout.addWidget(
            self.button_path, 0, QtCore.Qt.AlignmentFlag.AlignRight
        )
        path_box.setLayout(path_box_layout)

        self.mass_units = QtWidgets.QComboBox()
        self.mass_units.addItems(list(mass_units.keys()))
        self.mass_units.setCurrentText(_units["mass"])
        self.size_units = QtWidgets.QComboBox()
        self.size_units.addItems(list(size_units.keys()))
        self.size_units.setCurrentText(_units["size"])

        units_box = QtWidgets.QGroupBox("Units")
        units_box_layout = QtWidgets.QFormLayout()
        units_box_layout.addRow("Mass units", self.mass_units)
        units_box_layout.addRow("Size units", self.size_units)
        units_box.setLayout(units_box_layout)

        self.check_export_inputs = QtWidgets.QCheckBox(
            "Instrument, limit and isotope options"
        )
        self.check_export_inputs.setChecked(True)
        self.check_export_arrays = QtWidgets.QCheckBox("Particle data arrays")
        self.check_export_arrays.setChecked(True)
        self.check_export_compositions = QtWidgets.QCheckBox("Particle compositions")

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

        layout = QtWidgets.QGridLayout()
        layout.addWidget(
            QtWidgets.QLabel(
                f"Exporting <b>{data_file.path.name}</b>: {len(data_file.selected_isotopes)} isotopes"
            ),
            0,
            0,
            1,
            2,
        )
        layout.addWidget(path_box, 1, 0, 1, 2)
        layout.addWidget(units_box, 2, 0)
        layout.addWidget(switches_box, 2, 1)
        layout.addWidget(self.button_box, 3, 0, 1, 2)

        self.setLayout(layout)

    def completeChanged(self):
        self.button_box.button(
            QtWidgets.QDialogButtonBox.StandardButton.Save
        ).setEnabled(self.isComplete())

    def isComplete(self) -> bool:
        if not self.lineedit_path.hasAcceptableInput():
            return False
        return True

    def togglePressedItem(self, item: QtWidgets.QListWidgetItem):
        state = QtCore.Qt.CheckState.Checked
        if item.checkState() == state:
            state = QtCore.Qt.CheckState.Unchecked
        item.setCheckState(state)

    def dialogExportPath(self):
        file, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Path",
            self.lineedit_path.text(),
            "CSV Documents (*.csv);;All files (*)",
        )

        if any(x in file for x in ExportDialog.INVALID_CHARS):
            QtWidgets.QMessageBox.information(
                self,
                "Invalid Name",
                f"Files may not contain any of these characters: {ExportDialog.INVALID_CHARS}",
            )
            for char in ExportDialog.INVALID_CHARS:
                file = file.replace(char, "")

        if file != "":
            self.lineedit_path.setText(file)

    def accept(self):
        units = {
            "signal": ("cts", 1.0),
            "mass": (
                self.mass_units.currentText(),
                mass_units[self.mass_units.currentText()],
            ),
            "size": (
                self.size_units.currentText(),
                size_units[self.size_units.currentText()],
            ),
        }

        path = Path(self.lineedit_path.text())

        # Check if overwriting
        if path.exists():
            button = QtWidgets.QMessageBox.warning(
                self,
                "Overwrite File?",
                f"The file '{path.stem}' already exists, do you want to overwrite it?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if button == QtWidgets.QMessageBox.StandardButton.No:
                return

        export_spcal_processing_results(
            path,
            self.data_file,
            list(self.results.values()),
            self.clusters,
            units=units,
            export_options=self.check_export_inputs.isChecked(),
            export_compositions=self.check_export_compositions.isChecked(),
            export_arrays=self.check_export_arrays.isChecked(),
        )
        logger.info(f"Results for {self.data_file.path.stem} exported to {path}.")
        super().accept()
