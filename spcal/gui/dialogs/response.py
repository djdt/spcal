from pathlib import Path
from typing import Dict

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.dialogs._import import ImportDialog, NuImportDialog
from spcal.gui.graphs.response import ResponseView
from spcal.gui.objects import DoublePrecisionDelegate
from spcal.io.nu import is_nu_directory
from spcal.io.text import is_text_file
from spcal.siunits import mass_concentration_units


class ResponseDialog(QtWidgets.QDialog):
    responsesSelected = QtCore.Signal(dict)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent)
        self.setWindowTitle("Ionic Response Calculator")

        self.data = np.array([])
        self.responses: Dict[str, float] = {}

        self.button_open_file = QtWidgets.QPushButton("Open File")
        self.button_open_file.pressed.connect(self.dialogLoadFile)

        self.graph = ResponseView()
        self.graph.region.sigRegionChangeFinished.connect(self.updateResponses)

        self.table = QtWidgets.QTableWidget()
        self.table.setRowCount(2)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setVerticalHeaderLabels(["Response (counts)", "Concentration"])
        self.table.setItemDelegate(DoublePrecisionDelegate(bottom=1e-99, decimals=4))

        self.table.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents
        )
        self.table.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        self.table.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self.table.itemChanged.connect(self.completeChanged)

        self.combo_unit = QtWidgets.QComboBox()
        self.combo_unit.addItems(list(mass_concentration_units.keys()))
        self.combo_unit.setCurrentText("μg/L")

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
        )
        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout_units = QtWidgets.QHBoxLayout()
        layout_units.addStretch(1)
        layout_units.addWidget(
            QtWidgets.QLabel("Concentration units:"),
            0,
            QtCore.Qt.AlignmentFlag.AlignRight,
        )
        layout_units.addWidget(self.combo_unit, 0, QtCore.Qt.AlignmentFlag.AlignRight)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.button_open_file, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.graph, 1)
        layout.addWidget(self.table, 0)
        layout.addLayout(layout_units, 0)
        layout.addWidget(self.button_box, 0)
        self.setLayout(layout)

    def isComplete(self) -> bool:
        return any(
            self.table.item(1, i).data(QtCore.Qt.ItemDataRole.DisplayRole) > 0.0
            for i in range(self.table.columnCount())
        )

    def completeChanged(self) -> None:
        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(
            self.isComplete()
        )

    def accept(self) -> None:
        responses = {}
        factor = mass_concentration_units[self.combo_unit.currentText()]
        for i in range(self.table.columnCount()):
            name = self.table.horizontalHeaderItem(i).text()
            response = self.table.item(0, i).data(QtCore.Qt.ItemDataRole.DisplayRole)
            conc = self.table.item(1, i).data(QtCore.Qt.ItemDataRole.DisplayRole)
            if conc > 0.0:
                responses[name] = response / (conc * factor)
        if len(responses) > 0:
            self.responsesSelected.emit(responses)
        super().accept()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if event.mimeData().hasUrls():
            # Todo, nu import check
            for url in event.mimeData().urls():
                path = Path(url.toLocalFile())
                if (
                    (path.is_dir() and is_nu_directory(path))
                    or path.suffix.lower() == ".info"
                    or is_text_file(path)
                ):
                    self.dialogLoadFile(path)
                    break
            event.acceptProposedAction()
        elif event.mimeData().hasHtml():
            pass
        else:
            super().dropEvent(event)

    def dialogLoadFile(
        self, path: str | Path | None = None
    ) -> ImportDialog | NuImportDialog | None:
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Open",
                "",
                (
                    "NP Data Files (*.csv *.info);;CSV Documents(*.csv *.txt *.text);;"
                    "Nu Instruments(*.info);;All files(*)"
                ),
            )
            if path == "":
                return None

        path = Path(path)

        if path.suffix == ".info":  # Cast down for nu
            path = path.parent

        if path.is_dir():
            if is_nu_directory(path):
                dlg = NuImportDialog(path, self)
            else:
                raise FileNotFoundError("dialogLoadFile: invalid directory.")
        else:
            dlg = ImportDialog(path, self)
        dlg.dataImported.connect(self.loadData)
        dlg.open()
        return dlg

    def loadData(self, data: np.ndarray, options: dict) -> None:
        self.data = data
        tic = np.sum([data[name] for name in data.dtype.names], axis=0)

        self.table.setColumnCount(len(data.dtype.names))
        self.table.setHorizontalHeaderLabels(data.dtype.names)

        self.table.blockSignals(True)
        for i in range(self.table.columnCount()):
            item = QtWidgets.QTableWidgetItem()
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(0, i, item)
            item = QtWidgets.QTableWidgetItem()
            item.setData(QtCore.Qt.ItemDataRole.DisplayRole, np.nan)
            self.table.setItem(1, i, item)
        self.table.blockSignals(True)

        self.graph.clear()
        self.graph.drawData(np.arange(tic.size), tic)
        self.graph.drawMean(0.0)
        self.graph.updateMean()

        self.updateResponses()

    def updateResponses(self) -> None:
        if self.data.dtype.names is None:
            return
        responses = [
            np.mean(self.data[name][self.graph.region_start : self.graph.region_end])
            for name in self.data.dtype.names
        ]
        self.table.blockSignals(True)
        for i, response in enumerate(responses):
            self.table.item(0, i).setData(QtCore.Qt.ItemDataRole.DisplayRole, response)
        self.table.blockSignals(False)


if __name__ == "__main__":
    app = QtWidgets.QApplication()

    w = ResponseDialog()
    npz = np.load("/home/tom/Downloads/test_data.npz")
    names = npz.files
    data = np.empty(npz[names[0]].size, dtype=[(n, float) for n in names])
    for n in names:
        data[n] = npz[n]
    w.loadData(data, {})
    w.show()
    app.exec()
