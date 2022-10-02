from PySide2 import QtCore, QtGui, QtWidgets
import numpy as np

from typing import List, Optional


class ImportDialog(QtWidgets.QDialog):
    dataImported = QtCore.Signal(np.ndarray)

    def __init__(self, file: str, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.resize(800, 800)
        self.setWindowTitle("SPCal File Import")

        header_row_count = 10

        # self.file_data = self.delimited_translated_columns(file)
        self.file_pointer = open(file, "r")
        self.file_header = [
            x for _, x in zip(range(header_row_count), self.file_pointer)
        ]

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
        self.table.setColumnCount(column_count)
        self.table.setRowCount(header_row_count)
        self.table.setFont(QtGui.QFont("Courier"))

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
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        controls = QtWidgets.QFormLayout()
        controls.addRow("Delimiter:", self.combo_delimiter)
        controls.addRow("Import From Row:", self.spinbox_first_line)
        controls.addRow("Ignore Columns:", self.le_ignore_columns)

        self.fillTable()

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(self.table)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def delimiter(self) -> str:
        delimiter = self.combo_delimiter.currentText()
        if delimiter == "Space":
            delimiter = " "
        elif delimiter == "Tab":
            delimiter = "\t"
        return delimiter

    def ignoreColumns(self) -> List[int]:
        return [int(i or 0) - 1 for i in self.le_ignore_columns.text().split(";")]

    def fillTable(self) -> None:
        lines = [line.split(self.delimiter()) for line in self.file_header]
        self.table.setColumnCount(max(len(line) for line in lines))
        for row, line in enumerate(lines):
            for col, text in enumerate(line):
                item = QtWidgets.QTableWidgetItem(text.strip())
                if row != self.spinbox_first_line.value() - 1:
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                if (
                    row < self.spinbox_first_line.value() - 1
                    or col in self.ignoreColumns()
                ):
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEnabled)

                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()

    def accept(self) -> None:
        self.file_pointer.seek(0)

        ignores = self.ignoreColumns()
        cols = [col for col in range(self.table.columnCount()) if col not in ignores]
        header_row = self.spinbox_first_line.value() - 1
        headers = [self.table.item(header_row, col).text() for col in cols]

        data = np.genfromtxt(
            self.file_pointer,
            delimiter=self.delimiter(),
            usecols=cols,
            names=headers,
            skip_header=header_row + 1,
            converters={0: lambda s: float(s.replace(",", "."))},
            invalid_raise=False,
        )
        self.dataImported.emit(data)
        super().accept()
