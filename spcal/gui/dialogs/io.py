from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.units import (
    UnitsWidget,
    mass_units,
    molar_concentration_units,
    size_units,
    time_units,
)
from spcal.io.text import export_single_particle_results, import_single_particle_file
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


class ImportDialog(QtWidgets.QDialog):
    dataImported = QtCore.Signal(np.ndarray, dict)

    forbidden_names = ["Overlay"]

    def __init__(self, path: str | Path, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        header_row_count = 10

        self.file_path = Path(path)
        self.file_header = [
            x for _, x in zip(range(header_row_count), self.file_path.open("r"))
        ]
        self.setWindowTitle(f"SPCal File Import: {self.file_path.name}")

        first_data_line = 0
        for line in self.file_header:
            try:
                float(line.split(",")[-1])
                break
            except ValueError:
                pass
            first_data_line += 1

        column_count = max([line.count(",") for line in self.file_header]) + 1

        self.table = QtWidgets.QTableWidget()
        self.table.itemChanged.connect(self.completeChanged)
        self.table.setMinimumSize(800, 400)
        self.table.setColumnCount(column_count)
        self.table.setRowCount(header_row_count)
        self.table.setFont(QtGui.QFont("Courier"))

        self.dwelltime = UnitsWidget(
            time_units,
            default_unit="ms",
            validator=QtGui.QDoubleValidator(0.0, 10.0, 10),
        )
        self.dwelltime.valueChanged.connect(self.completeChanged)

        self.combo_intensity_units = QtWidgets.QComboBox()
        self.combo_intensity_units.addItems(["Counts", "CPS"])
        if any("cps" in line.lower() for line in self.file_header):
            self.combo_intensity_units.setCurrentText("CPS")

        self.combo_delimiter = QtWidgets.QComboBox()
        self.combo_delimiter.addItems([",", ";", "Space", "Tab"])
        self.combo_delimiter.currentIndexChanged.connect(self.fillTable)

        self.spinbox_first_line = QtWidgets.QSpinBox()
        self.spinbox_first_line.setRange(1, header_row_count - 1)
        self.spinbox_first_line.setValue(first_data_line)
        self.spinbox_first_line.valueChanged.connect(self.updateTableIgnores)

        # self.combo_ignore_columns = QtWidgets.QComboBox()
        # self.combo_ignore_columns.addItems(["Use", "Ignore"])
        # self.combo_ignore_columns.setCurrentIndex(1)
        # self.combo_ignore_columns.currentTextChanged.connect(self.updateLEIgnores)

        self.le_ignore_columns = QtWidgets.QLineEdit()
        self.le_ignore_columns.setText("1;")
        self.le_ignore_columns.setValidator(
            QtGui.QRegularExpressionValidator(QtCore.QRegularExpression("[0-9;]+"))
        )
        self.le_ignore_columns.textChanged.connect(self.updateTableIgnores)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        import_form = QtWidgets.QFormLayout()
        import_form.addRow("Delimiter:", self.combo_delimiter)
        import_form.addRow("Import From Row:", self.spinbox_first_line)
        import_form.addRow("Ignore Columns:", self.le_ignore_columns)

        import_box = QtWidgets.QGroupBox("Import Options")
        import_box.setLayout(import_form)

        data_form = QtWidgets.QFormLayout()
        data_form.addRow("Dwell Time:", self.dwelltime)
        data_form.addRow("Intensity Units:", self.combo_intensity_units)

        data_box = QtWidgets.QGroupBox("Data Options")
        data_box.setLayout(data_form)

        box_layout = QtWidgets.QHBoxLayout()
        box_layout.addWidget(import_box, 1)
        box_layout.addWidget(data_box, 1)

        self.fillTable()

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(box_layout)
        layout.addWidget(self.table)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def isComplete(self) -> bool:
        return self.dwelltime.hasAcceptableInput() and not any(
            x in self.forbidden_names for x in self.names()
        )

    def completeChanged(self) -> None:
        complete = self.isComplete()
        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(complete)

    def delimiter(self) -> str:
        delimiter = self.combo_delimiter.currentText()
        if delimiter == "Space":
            delimiter = " "
        elif delimiter == "Tab":
            delimiter = "\t"
        return delimiter

    def ignoreColumns(self) -> List[int]:
        return [int(i or 0) - 1 for i in self.le_ignore_columns.text().split(";")]

    def useColumns(self) -> List[int]:
        return [
            c for c in range(self.table.columnCount()) if c not in self.ignoreColumns()
        ]

    def names(self) -> List[str]:
        return [
            self.table.item(self.spinbox_first_line.value() - 1, c).text()
            for c in self.useColumns()
        ]

    def fillTable(self) -> None:
        lines = [line.split(self.delimiter()) for line in self.file_header]
        col_count = max(len(line) for line in lines)
        self.table.setColumnCount(col_count)

        for row, line in enumerate(lines):
            line.extend([""] * (col_count - len(line)))
            for col, text in enumerate(line):
                item = QtWidgets.QTableWidgetItem(text.strip())
                self.table.setItem(row, col, item)

        self.table.resizeColumnsToContents()
        self.updateTableIgnores()

        if self.dwelltime.value() is None:
            self.readDwelltimeFromTable()

    def updateTableIgnores(self) -> None:
        header_row = self.spinbox_first_line.value() - 1
        for row in range(self.table.rowCount()):
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item is None:
                    continue
                if row != header_row:
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                else:
                    item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
                if row < header_row or col in self.ignoreColumns():
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEnabled)
                else:
                    item.setFlags(item.flags() | QtCore.Qt.ItemIsEnabled)

    # def updateLEIgnores(self, text: str) -> None:
    #     if text == "Ignore":
    #         pass
    #     elif text == "Use":
    #         pass

    def readDwelltimeFromTable(self) -> None:
        header_row = self.spinbox_first_line.value() - 1
        for col in range(self.table.columnCount()):
            text = self.table.item(header_row, col).text().lower()
            if "time" in text:
                try:
                    times = [
                        float(self.table.item(row, col).text().replace(",", "."))
                        for row in range(header_row + 1, self.table.rowCount())
                    ]
                except ValueError:
                    continue
                if "ms" in text:
                    factor = 1e-3
                elif "us" in text or "μs" in text:
                    factor = 1e-6
                else:  # assume that value is in seconds
                    factor = 1.0
                self.dwelltime.setBaseValue(
                    np.round(np.mean(np.diff(times)), 6) * factor  # type: ignore
                )
                break

    def importOptions(self) -> dict:
        return {
            "path": self.file_path,
            "dwelltime": self.dwelltime.baseValue(),
            "delimiter": self.delimiter(),
            "ignores": self.ignoreColumns(),
            "columns": self.useColumns(),
            "first line": self.spinbox_first_line.value(),
            "names": self.names(),
            "cps": self.combo_intensity_units.currentText() == "CPS",
        }

    def accept(self) -> None:
        options = self.importOptions()

        data, old_names = import_single_particle_file(
            options["path"],
            delimiter=options["delimiter"],
            columns=options["columns"],
            first_line=options["first line"],
            new_names=options["names"],
            convert_cps=options["dwelltime"] if options["cps"] else None,
        )
        # Save original names
        assert data.dtype.names is not None
        options["old names"] = old_names

        self.dataImported.emit(data, options)
        super().accept()
