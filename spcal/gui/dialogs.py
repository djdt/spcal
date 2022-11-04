from PySide6 import QtCore, QtGui, QtWidgets
import numpy as np
from pathlib import Path

from spcal.gui.units import UnitsWidget

from typing import Dict, List


class BinWidthDialog(QtWidgets.QDialog):
    binWidthsChanged = QtCore.Signal(dict)

    def __init__(
        self,
        bin_widths: Dict[str, float | None],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Histogram Bin Widths")

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
        concentration_units = {
            "amol/L": 1e-18,
            "fmol/L": 1e-15,
            "pmol/L": 1e-12,
            "nmol/L": 1e-9,
            "μmol/L": 1e-6,
            "mmol/L": 1e-3,
            "mol/L": 1.0,
        }

        self.width_signal = QtWidgets.QLineEdit(str((bin_widths.get("signal", ""))))
        self.width_signal.setPlaceholderText("auto")
        self.width_signal.setValidator(QtGui.QIntValidator(0, 999999999))

        color = self.palette().color(QtGui.QPalette.Base)

        self.width_mass = UnitsWidget(
            mass_units, value=bin_widths.get("mass", None), invalid_color=color
        )
        self.width_size = UnitsWidget(
            size_units, value=bin_widths.get("size", None), invalid_color=color
        )
        self.width_conc = UnitsWidget(
            concentration_units,
            value=bin_widths.get("concentration", None),
            invalid_color=color,
        )

        for widget in [self.width_mass, self.width_size, self.width_conc]:
            widget.setBestUnit()
            widget.lineedit.setPlaceholderText("auto")

        layout_form = QtWidgets.QFormLayout()
        layout_form.addRow("Signal:", self.width_signal)
        layout_form.addRow("Mass:", self.width_mass)
        layout_form.addRow("Size:", self.width_size)
        layout_form.addRow("Concentration:", self.width_conc)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.RestoreDefaults
            | QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Cancel
        )
        self.button_box.clicked.connect(self.buttonBoxClicked)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(layout_form)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

    def buttonBoxClicked(self, button: QtWidgets.QAbstractButton) -> None:
        sbutton = self.button_box.standardButton(button)
        if sbutton == QtWidgets.QDialogButtonBox.RestoreDefaults:
            self.reset()
        elif sbutton == QtWidgets.QDialogButtonBox.Ok:
            self.accept()
        else:
            self.reject()

    def accept(self) -> None:
        bin_widths = {
            "mass": self.width_mass.baseValue(),
            "size": self.width_size.baseValue(),
            "concentration": self.width_conc.baseValue(),
        }
        try:
            bin_widths["signal"] = int(self.width_signal.text())
        except ValueError:
            bin_widths["signal"] = None
        self.binWidthsChanged.emit(bin_widths)

        super().accept()

    def reset(self) -> None:
        self.width_signal.setText("")
        for widget in [self.width_mass, self.width_size, self.width_conc]:
            widget.setBaseValue(None)


class ImportDialog(QtWidgets.QDialog):
    dataImported = QtCore.Signal(np.ndarray, dict)

    forbidden_names = ["Overlay"]

    def __init__(self, file: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        header_row_count = 10

        self.file_path = Path(file)
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
            {"ms": 1e-3, "s": 1.0}, default_unit="ms", validator=(0.0, 10.0, 10)
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
        self.spinbox_first_line.valueChanged.connect(self.fillTable)

        self.le_ignore_columns = QtWidgets.QLineEdit()
        self.le_ignore_columns.setText("1;")
        self.le_ignore_columns.setValidator(
            QtGui.QRegularExpressionValidator(QtCore.QRegularExpression("[0-9;]+"))
        )
        self.le_ignore_columns.textChanged.connect(self.fillTable)

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
            x in self.forbidden_names for x in self.headers()
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

    def headers(self) -> List[str]:
        return [
            self.table.item(self.spinbox_first_line.value() - 1, c).text()
            for c in self.useColumns()
        ]

    def fillTable(self) -> None:
        lines = [line.split(self.delimiter()) for line in self.file_header]
        header_row = self.spinbox_first_line.value() - 1
        self.table.setColumnCount(max(len(line) for line in lines))

        for row, line in enumerate(lines):
            for col, text in enumerate(line):
                item = QtWidgets.QTableWidgetItem(text.strip())
                if row != header_row:
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                if row < header_row or col in self.ignoreColumns():
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEnabled)

                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()

        if self.dwelltime.value() is None:
            self.readDwelltimeFromTable()

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
            "headers": self.headers(),
            "cps": self.combo_intensity_units.currentText() == "CPS",
        }

    # def setImportOptions(self, options: dict) -> None:
    #     pass

    def accept(self) -> None:
        options = self.importOptions()

        data = np.genfromtxt(
            options["path"],
            delimiter=options["delimiter"],
            usecols=options["columns"],
            names=options["headers"],
            skip_header=options["first line"],
            converters={0: lambda s: float(s.replace(",", "."))},
            invalid_raise=False,
        )

        if options["cps"]:
            dwell = options["dwelltime"]
            for name in data.dtype.names:
                data[name] *= dwell  # type: ignore

        self.dataImported.emit(data, options)
        super().accept()
