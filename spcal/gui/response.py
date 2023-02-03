from pathlib import Path
from typing import Dict

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.dialogs._import import ImportDialog, NuImportDialog
from spcal.gui.graphs.views import ResponseView
from spcal.io.nu import is_nu_directory
from spcal.io.text import is_text_file


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

        self.button_set_responses = QtWidgets.QPushButton("Set Responses")
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
        self.responsesChanged.emit(responses)

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
    w.dialogLoadFile(
        "/home/tom/MEGA/Uni/Experimental/ICPMS/Mn Single Cell/20220126_mn_treatment/20220126_c_neb_eff.b/c.d/c.csv"
    )
    w.show()
    app.exec()
