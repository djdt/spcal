from pathlib import Path
from typing import Dict

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.dialogs._import import ImportDialog, NuImportDialog
from spcal.gui.graphs.views import ResponseView
from spcal.gui.objects import DoublePrecisionDelegate
from spcal.io.nu import is_nu_directory
from spcal.io.text import is_text_file
from spcal.siunits import mass_concentration_units


class SetResponseDialog(QtWidgets.QDialog):
    responsesSelected = QtCore.Signal(dict)

    def __init__(
        self, responses: Dict[str, float], parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent=parent)
        self.setWindowTitle("Response Concentrations")

        self.table = QtWidgets.QTableWidget()

        self.table.setColumnCount(len(responses))
        self.table.setRowCount(2)

        self.table.setHorizontalHeaderLabels(list(responses.keys()))
        self.table.setVerticalHeaderLabels(["Response (counts)", "Concentration"])
        self.table.setItemDelegate(DoublePrecisionDelegate(4))

        for i, response in enumerate(responses.values()):
            item = QtWidgets.QTableWidgetItem()
            item.setData(QtCore.Qt.ItemDataRole.DisplayRole, response)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(0, i, item)
            item = QtWidgets.QTableWidgetItem()
            item.setData(QtCore.Qt.ItemDataRole.DisplayRole, 0.0)
            self.table.setItem(1, i, item)

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
        layout.addWidget(self.table, 1)
        layout.addLayout(layout_units, 0)
        layout.addWidget(self.button_box, 0)
        self.setLayout(layout)

    def completeChanged(self) -> None:
        ready = any(
            float(self.table.item(1, i).text()) != 0.0
            for i in range(self.table.columnCount())
        )
        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(ready)

    def accept(self) -> None:
        responses = {}
        factor = mass_concentration_units[self.combo_unit.currentText()]
        for i in range(self.table.columnCount()):
            name = self.table.horizontalHeaderItem(i).text()
            response = self.table.item(0, i).data(QtCore.Qt.ItemDataRole.DisplayRole)
            conc = self.table.item(1, i).data(QtCore.Qt.ItemDataRole.DisplayRole)
            if conc != 0.0:
                responses[name] = response / (conc * factor)
        if len(responses) > 0:
            self.responsesSelected.emit(responses)


class ResponseWidget(QtWidgets.QWidget):
    responsesChanged = QtCore.Signal(dict)

    def __init__(
        self, file: str | Path | None = None, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent=parent)

        self.data = np.array([])

        self.button_open_file = QtWidgets.QPushButton("Open File")
        self.button_open_file.pressed.connect(self.dialogLoadFile)

        # self.button_reset = QtWidgets.QPushButton("Reset")
        # self.button_reset.setIcon(QtGui.QIcon.fromTheme("edit-reset"))
        # self.button_reset.pressed.connect(self.resetInputs)

        self.graph = ResponseView()

        self.combo_method = QtWidgets.QComboBox()
        self.combo_method.addItems(["Signal Mean", "Signal Median"])
        self.combo_method.currentTextChanged.connect(self.setMeanMode)

        self.button_set_responses = QtWidgets.QPushButton("Set Concentrations")
        self.button_set_responses.pressed.connect(self.calculateResponses)
        self.button_set_responses.setIcon(QtGui.QIcon.fromTheme("dialog-ok-apply"))

        layout_buttons = QtWidgets.QHBoxLayout()
        layout_buttons.addWidget(
            self.button_open_file, 0, QtCore.Qt.AlignmentFlag.AlignLeft
        )
        # layout_buttons.addWidget(
        #     self.button_reset, 0, QtCore.Qt.AlignmentFlag.AlignRight
        # )

        layout_controls = QtWidgets.QHBoxLayout()
        layout_controls.addWidget(self.combo_method, 0)
        layout_controls.addWidget(
            self.button_set_responses, 1, QtCore.Qt.AlignmentFlag.AlignRight
        )

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(layout_buttons, 0)
        layout.addWidget(self.graph, 1)
        layout.addLayout(layout_controls, 0)

        self.setLayout(layout)

    def calculateResponses(self) -> None:
        responses: Dict[str, float] = {}
        fn = (
            np.median if self.combo_method.currentText() == "Signal Median" else np.mean
        )
        if self.data.dtype.names is None:
            return
        for name in self.data.dtype.names:
            responses[name] = fn(
                self.data[name][self.graph.region_start : self.graph.region_end]
            )
        dlg = SetResponseDialog(responses, self)
        dlg.open()

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

        self.graph.clear()
        self.graph.drawData(np.arange(tic.size), tic)
        self.graph.drawMean(0.0)
        self.graph.updateMean()

    def setMeanMode(self) -> None:
        self.graph.mean_mode = (
            "median" if self.combo_method.currentText() == "Signal Median" else "mean"
        )
        self.graph.updateMean()

    def resetInputs(self) -> None:
        self.graph.clear()


if __name__ == "__main__":
    app = QtWidgets.QApplication()

    w = ResponseWidget()
    npz = np.load("/home/tom/Downloads/test_data.npz")
    names = npz.files
    data = np.empty(npz[names[0]].size, dtype=[(n, float) for n in names])
    for n in names:
        data[n] = npz[n]
    w.loadData(data, {})
    w.show()
    app.exec()
