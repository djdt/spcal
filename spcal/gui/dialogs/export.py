import logging
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.datafile import SPCalDataFile, SPCalNuDataFile
from spcal.io.export import export_spcal_processing_results
from spcal.isotope import SPCalIsotopeBase
from spcal.processing.result import SPCalProcessingResult
from spcal.siunits import mass_units, size_units, volume_units

logger = logging.getLogger(__name__)


class ExportDialog(QtWidgets.QDialog):
    INVALID_CHARS = '<>:"/\\|?*'

    def __init__(
        self,
        data_files: list[SPCalDataFile],
        results: dict[SPCalDataFile, dict[SPCalIsotopeBase, SPCalProcessingResult]],
        clusters: dict[SPCalDataFile, dict[str, np.ndarray]],
        path: Path | None = None,
        units: dict[str, tuple[str, float]] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Export Results")

        self.results = results
        self.clusters = clusters

        self.list = QtWidgets.QListWidget()
        for data_file in data_files:
            item = QtWidgets.QListWidgetItem()
            item.setText(str(data_file.path.stem))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, data_file)
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            self.list.addItem(item)
        self.list.itemPressed.connect(self.togglePressedItem)
        self.list.itemChanged.connect(self.completeChanged)
        self.list.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents
        )

        _units = {"mass": "fg", "size": "nm", "volume": "nm³"}
        if units is not None:
            _units.update({k: v[0] for k, v in units.items()})

        filename_regexp = QtCore.QRegularExpression(
            f"(%DataFile%)?[^{ExportDialog.INVALID_CHARS}]+(%DataFile%)?"
        )

        self.lineedit_name = QtWidgets.QLineEdit("%DataFile%_spcal_results.csv")
        self.lineedit_name.setMinimumWidth(300)
        self.lineedit_name.setValidator(
            QtGui.QRegularExpressionValidator(filename_regexp)
        )
        self.lineedit_name.textChanged.connect(self.completeChanged)

        self.lineedit_dir = QtWidgets.QLineEdit(str(data_files[0].path.parent))
        self.lineedit_dir.setMinimumWidth(300)
        self.lineedit_dir.textChanged.connect(self.completeChanged)

        self.button_path = QtWidgets.QPushButton("Directory")
        self.button_path.clicked.connect(self.dialogDirectory)

        file_box = QtWidgets.QGroupBox("Data files")
        file_box_layout = QtWidgets.QGridLayout()
        file_box_layout.addWidget(self.list, 0, 0, 1, 3)
        file_box_layout.addWidget(QtWidgets.QLabel("File Name:"), 1, 0)
        file_box_layout.addWidget(QtWidgets.QLabel("Directory:"), 2, 0)
        file_box_layout.addWidget(self.lineedit_name, 1, 1)
        file_box_layout.addWidget(self.lineedit_dir, 2, 1)
        file_box_layout.addWidget(self.button_path, 2, 2)
        file_box.setLayout(file_box_layout)

        self.mass_units = QtWidgets.QComboBox()
        self.mass_units.addItems(list(mass_units.keys()))
        self.mass_units.setCurrentText(_units["mass"])
        self.size_units = QtWidgets.QComboBox()
        self.size_units.addItems(list(size_units.keys()))
        self.size_units.setCurrentText(_units["size"])
        self.volume_units = QtWidgets.QComboBox()
        self.volume_units.addItems(list(volume_units.keys()))
        self.volume_units.setCurrentText(_units["volume"])

        units_box = QtWidgets.QGroupBox("Units")
        units_box_layout = QtWidgets.QFormLayout()
        units_box_layout.addRow("Mass units", self.mass_units)
        units_box_layout.addRow("Size units", self.size_units)
        units_box_layout.addRow("Volume units", self.volume_units)
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

        layout = QtWidgets.QGridLayout()
        layout.addWidget(file_box, 0, 0)
        layout.addWidget(units_box, 0, 1)
        layout.addWidget(switches_box, 1, 1)
        layout.addWidget(self.button_box, 2, 0, 1, 2)

        self.setLayout(layout)

    def completeChanged(self):
        self.button_box.button(
            QtWidgets.QDialogButtonBox.StandardButton.Save
        ).setEnabled(self.isComplete())

    def isComplete(self) -> bool:
        if not self.lineedit_name.hasAcceptableInput():
            return False
        if not Path(self.lineedit_dir.text()).exists():
            return False

        for i in range(self.list.count()):
            if self.list.item(i).checkState() == QtCore.Qt.CheckState.Checked:
                return True
        return False

    def togglePressedItem(self, item: QtWidgets.QListWidgetItem):
        state = QtCore.Qt.CheckState.Checked
        if item.checkState() == state:
            state = QtCore.Qt.CheckState.Unchecked
        item.setCheckState(state)

    def dialogDirectory(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Directory", self.lineedit_dir.text()
        )
        if dir == "":
            return
        self.lineedit_dir.setText(dir)

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
            "volume": (
                self.volume_units.currentText(),
                volume_units[self.volume_units.currentText()],
            ),
        }

        for i in range(self.list.count()):
            if self.list.item(i).checkState() != QtCore.Qt.CheckState.Checked:
                continue
            data_file = self.list.item(i).data(QtCore.Qt.ItemDataRole.UserRole)

            path = Path(self.lineedit_dir.text())
            if not path.exists():
                raise ValueError(f"Directory '{path}' does not exist.")
            file_name = self.lineedit_name.text().replace(
                "%DataFile%", data_file.path.stem
            )
            path = path.joinpath(file_name).with_suffix(".csv")

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

            export_spcal_processing_results(
                path,
                data_file,
                list(self.results[data_file].values()),
                self.clusters[data_file],
                units=units,
                export_options=self.check_export_inputs.isChecked(),
                export_compositions=self.check_export_compositions.isChecked(),
                export_arrays=self.check_export_arrays.isChecked(),
            )
            logger.info(f"Results for {data_file.path.stem} exported to {path}.")
        super().accept()


if __name__ == "__main__":
    from spcal.processing import SPCalProcessingMethod
    from spcal.processing.options import SPCalIsotopeOptions
    from spcal.isotope import ISOTOPE_TABLE

    method = SPCalProcessingMethod()
    df1 = SPCalNuDataFile.load(
        Path("/home/tom/Downloads/15-03-56 5.0ppb + 80nm Au + UCNP")
    )
    df1.selected_isotopes = [
        ISOTOPE_TABLE[("Yb", 171)],
        ISOTOPE_TABLE[("Yb", 172)],
        ISOTOPE_TABLE[("Yb", 173)],
    ]

    df2 = SPCalNuDataFile.load(Path("/home/tom/Downloads/NT032/14-36-31 10 ppb att/"))
    df2.selected_isotopes = [
        ISOTOPE_TABLE[("Yb", 171)],
        ISOTOPE_TABLE[("Yb", 172)],
        ISOTOPE_TABLE[("Yb", 173)],
        ISOTOPE_TABLE[("Au", 197)],
    ]

    for isotope in [
        ISOTOPE_TABLE[("Yb", 171)],
        ISOTOPE_TABLE[("Yb", 172)],
        ISOTOPE_TABLE[("Yb", 173)],
        ISOTOPE_TABLE[("Au", 197)],
    ]:
        method.isotope_options[isotope] = SPCalIsotopeOptions(None, None, None)

    results1 = method.processDataFile(df1)
    results2 = method.processDataFile(df2)

    method.filterResults(results1)
    method.filterResults(results2)

    app = QtWidgets.QApplication()

    dlg = ExportDialog([df1, df2], {df1: results1, df2: results2}, {df1: {}, df2: {}})
    dlg.open()
    app.exec()
