from PySide2 import QtCore, QtGui, QtWidgets

from typing import Generator, Optional


class ImportDialog(QtWidgets.QDialog):
    def __init__(self, file: str, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.resize(800, 800)
        self.setWindowTitle("SPCal File Import")

        header_row_count = 10

        self.file_data = self.delimited_translated_columns(file)
        self.file_header = [x for _, x in zip(range(header_row_count), self.file_data)]

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

        self.spinbox_first_line = QtWidgets.QSpinBox()
        self.spinbox_first_line.setRange(0, header_row_count - 1)
        self.spinbox_first_line.setValue(first_data_line)
        self.spinbox_first_line.valueChanged.connect(self.fill_table)

        self.le_ignore_columns = QtWidgets.QLineEdit()
        self.le_ignore_columns.setText("1,")
        self.le_ignore_columns.setValidator(QtGui.QRegularExpressionValidator(QtCore.QRegularExpression("[0-9,]+")))
        self.le_ignore_columns.textChanged.connect(self.fill_table)

        controls = QtWidgets.QFormLayout()
        controls.addRow("Import From Row:", self.spinbox_first_line)
        controls.addRow("Ignore Columns:", self.le_ignore_columns)

        self.fill_table()

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(self.table)
        self.setLayout(layout)

    def fill_table(self) -> None:
        ignores = [int(i or 0) - 1 for i in self.le_ignore_columns.text().split(",")]
        for row, line in enumerate(self.file_header):
            for col, text in enumerate(line.split(",")):
                item = QtWidgets.QTableWidgetItem(text)
                if row < self.spinbox_first_line.value() - 1 or col in ignores:
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEnabled)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()

    def delimited_translated_columns(
        self, file: str, columns: int = 1
    ) -> Generator[str, None, None]:
        """Translates inputs with ';' to have ',' as delimiter and '.' as decimal.
        Ensures at least `columns` columns in data by prepending ','."""
        map = str.maketrans({";": ",", ",": "."})
        with open(file, "r") as fp:
            for line in fp:
                if ";" in line:
                    line = line.translate(map)
                count = line.count(",")
                if count < columns:
                    yield "," * (columns - count - 1) + line
                else:
                    yield line


if __name__ == "__main__":
    app = QtWidgets.QApplication()
    id = ImportDialog("/home/tom/Downloads/AuAg.csv")
    # id = ImportDialog(
    #     "/home/tom/MEGA/Uni/Experimental/ICPMS/Mn Single Cell/20211216_algae/20211216_algae_c.b/algae.d/algae.csv"
    # )
    id.show()
    app.exec_()
