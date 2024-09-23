import logging
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.calc import weighted_linreg
from spcal.gui.dialogs._import import _ImportDialogBase
from spcal.gui.graphs import color_schemes
from spcal.gui.graphs.calibration import CalibrationView
from spcal.gui.graphs.response import ResponseView
from spcal.gui.io import get_import_dialog_for_path, get_open_spcal_path, is_spcal_path
from spcal.gui.models import NumpyRecArrayTableModel
from spcal.siunits import mass_concentration_units

logger = logging.getLogger(__name__)


class ResponseDialog(QtWidgets.QDialog):
    responsesSelected = QtCore.Signal(dict)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent)
        self.setWindowTitle("Ionic Response Calculator")
        self.setMinimumSize(640, 480)

        self.data = np.array([])
        self.import_options: dict | None = None

        self.button_open_file = QtWidgets.QPushButton("Open File")
        self.button_open_file.pressed.connect(self.dialogLoadFile)

        self.graph = ResponseView()
        self.graph.region.sigRegionChangeFinished.connect(self.updateResponses)

        self.graph_cal = CalibrationView()
        self.graph_cal.sizeHint = lambda: QtCore.QSize(300, 300)

        data = np.array([], dtype=[("<element>", np.float64)])
        self.model = NumpyRecArrayTableModel(
            data, orientation=QtCore.Qt.Orientation.Horizontal
        )
        self.responses = np.array([], dtype=[("<element>", np.float64)])

        self.table = QtWidgets.QTableView()
        self.table.setModel(self.model)

        self.table.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents
        )
        self.table.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        self.table.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )

        self.model.dataChanged.connect(self.completeChanged)
        self.model.dataChanged.connect(self.updateCalibration)

        self.button_add_level = QtWidgets.QPushButton("Add Level")
        self.button_add_level.setIcon(QtGui.QIcon.fromTheme("list-add"))
        self.button_add_level.pressed.connect(self.dialogLoadFile)

        self.combo_unit = QtWidgets.QComboBox()
        self.combo_unit.addItems(list(mass_concentration_units.keys()))
        self.combo_unit.setCurrentText("μg/L")

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save
            | QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Reset
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
        )
        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
        self.button_box.button(QtWidgets.QDialogButtonBox.Save).setEnabled(False)

        self.button_box.clicked.connect(self.buttonClicked)
        self.button_box.rejected.connect(self.reject)

        box_concs = QtWidgets.QGroupBox("Concentrations")
        box_concs.setLayout(QtWidgets.QVBoxLayout())
        box_concs.layout().addWidget(self.table, 1)

        layout_conc_bar = QtWidgets.QHBoxLayout()
        layout_conc_bar.addStretch(1)
        layout_conc_bar.addWidget(
            self.button_add_level, 0, QtCore.Qt.AlignmentFlag.AlignRight
        )
        layout_conc_bar.addWidget(
            self.combo_unit, 0, QtCore.Qt.AlignmentFlag.AlignRight
        )
        box_concs.layout().addLayout(layout_conc_bar, 0)

        layout_graphs = QtWidgets.QHBoxLayout()
        layout_graphs.addWidget(self.graph, 3)
        layout_graphs.addWidget(self.graph_cal, 2)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(layout_graphs, 3)
        layout.addWidget(box_concs, 2)
        layout.addWidget(self.button_box, 0)
        self.setLayout(layout)

    def isComplete(self) -> bool:
        if self.model.array.dtype.names is None:
            return False
        for name in self.model.array.dtype.names:
            if np.count_nonzero(~np.isnan(self.model.array[name])) > 0:
                return True
        return False

    def completeChanged(self) -> None:
        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(
            self.isComplete()
        )
        self.button_box.button(QtWidgets.QDialogButtonBox.Save).setEnabled(
            self.isComplete()
        )

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = Path(url.toLocalFile())
                if is_spcal_path(path):
                    self.dialogLoadFile(path)
                    break
            event.acceptProposedAction()
        elif event.mimeData().hasHtml():
            pass
        else:
            super().dropEvent(event)

    def dialogLoadFile(
        self, path: str | Path | None = None
    ) -> _ImportDialogBase | None:
        if path is None:
            path = get_open_spcal_path(self)
            if path is None:
                return None
        else:
            path = Path(path)

        dlg = get_import_dialog_for_path(self, path)  # import_options managed below
        dlg.dataImported.connect(self.loadData)

        if self.import_options is None:
            dlg.open()
        else:
            try:
                dlg.setImportOptions(self.import_options)
                dlg.accept()
            except Exception:
                self.import_options = None
                logger.warning("dialogLoadFile: unable to set import options.")
                dlg.open()
        return dlg

    def loadData(self, data: np.ndarray, options: dict) -> None:
        # Check the new data is compatible with current loaded
        if self.model.array.size == 0:
            self.model.beginResetModel()
            self.model.array = np.full(1, np.nan, dtype=data.dtype)
            self.model.endResetModel()
            self.responses = self.model.array.copy()

        elif (
            data.dtype.names != self.model.array.dtype.names
        ):  # pragma: no cover, can't test msgbox
            button = QtWidgets.QMessageBox.question(
                self, "Warning", "New data does not match current, overwrite?"
            )
            if button == QtWidgets.QMessageBox.StandardButton.Yes:
                self.model.beginResetModel()
                self.model.array = np.full(1, np.nan, dtype=data.dtype)
                self.model.endResetModel()
                self.responses = self.model.array.copy()
            else:
                return
        else:
            self.model.insertColumn(self.model.columnCount())
            self.responses = np.append(
                self.responses, np.full(1, np.nan, self.responses.dtype)
            )

        if self.import_options is None:
            self.import_options = options

        old_size = self.data.size
        self.data = data
        tic = np.sum([data[name] for name in data.dtype.names], axis=0)
        xs = np.arange(tic.size)

        self.graph.clear()
        self.graph.plot.setTitle(f"TIC: {options['path'].name}")
        self.graph.drawData(xs, tic)
        if old_size != data.size:
            self.graph.region.blockSignals(True)
            self.graph.region.setRegion((xs[0], xs[-1]))
            self.graph.region.blockSignals(False)
        self.graph.updateMean()

        self.updateResponses()

    def updateResponses(self) -> None:
        if self.responses.dtype.names is None:  # pragma: no cover
            return

        for name in self.responses.dtype.names:
            self.responses[name][-1] = np.mean(
                self.data[name][self.graph.region_start : self.graph.region_end]
            )

        self.updateCalibration()

    def updateCalibration(self) -> None:
        self.graph_cal.clear()
        if self.responses.dtype.names is None:  # pragma: no cover
            return

        scheme = color_schemes[QtCore.QSettings().value("colorscheme", "IBM Carbon")]

        for i, name in enumerate(self.responses.dtype.names):
            x = self.model.array[name]
            y = self.responses[name][~np.isnan(x)]
            x = x[~np.isnan(x)]
            if x.size == 0:
                continue
            brush = QtGui.QBrush(scheme[i % len(scheme)])
            self.graph_cal.drawPoints(x, y, name=name, draw_trendline=True, brush=brush)

    def save(self) -> None:
        assert self.import_options is not None
        dir = self.import_options["path"].parent
        file, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Calibration", str(dir), "CSV Documents(*.csv);;All Files(*)"
        )
        if file == "":
            return

        assert self.responses.dtype.names is not None
        names = [  # remove any unpopulated names
            name
            for name in self.responses.dtype.names
            if np.any(~np.isnan(self.model.array[name]))
        ]
        nlevels = len(self.model.array)
        factor = mass_concentration_units[self.combo_unit.currentText()]

        def write_cal_levels(fp, name: str) -> None:
            fp.write(name + "," + ",".join(str(i) for i in range(len(xs))) + "\n")

        with open(file, "w") as fp:
            fp.write(f"#SPCal Calibration {version('spcal')}\n")
            fp.write(",Slope,Intercept,r2,Error\n")
            for name in names:
                m, b, r2, err = self.calibrationResult(name)
                fp.write(f"{name},{m},{b},{r2 or ''},{err or ''},\n")

            fp.write(
                "#Concentrations (kg/L),"
                + ",".join(str(i) for i in range(nlevels))
                + "\n"
            )
            for name in names:
                xs = self.model.array[name] * factor
                fp.write(
                    f"{name},"
                    + ",".join("" if np.isnan(x) else str(x) for x in xs)
                    + "\n"
                )
            fp.write(
                "#Responses (counts)," + ",".join(str(i) for i in range(nlevels)) + "\n"
            )
            for name in names:
                x = self.model.array[name]
                y = self.responses[name][~np.isnan(x)]
                x = x[~np.isnan(x)] * factor

                if x.size == 0:
                    continue

                fp.write(f"\n{name}," + ",".join(str(i) for i in range(len(x))) + "\n")
                fp.write("Conc. (kg/L)," + ",".join(str(xx) for xx in x) + "\n")
                fp.write("Response," + ",".join(str(xx) for xx in y) + "\n")

    def calibrationResult(
        self, name: str
    ) -> tuple[float | None, float | None, float | None, float | None]:
        factor = mass_concentration_units[self.combo_unit.currentText()]
        x = self.model.array[name]
        y = self.responses[name][~np.isnan(x)]
        x = x[~np.isnan(x)] * factor
        if x.size == 0:
            return None, None, None, None
        elif x.size == 1:  # single point, force 0
            return y[0] / x[0], 0.0, None, None
        else:
            return weighted_linreg(x, y)

    def buttonClicked(self, button: QtWidgets.QAbstractButton) -> None:
        sb = self.button_box.standardButton(button)
        if sb == QtWidgets.QDialogButtonBox.StandardButton.Ok:
            self.accept()
        elif sb == QtWidgets.QDialogButtonBox.StandardButton.Save:
            self.save()
        elif sb == QtWidgets.QDialogButtonBox.StandardButton.Reset:
            self.reset()

    def accept(self) -> None:
        assert self.responses.dtype.names is not None

        responses = {}
        for name in self.responses.dtype.names:
            m, _, _, _ = self.calibrationResult(name)
            if m is not None:
                responses[name] = m

        if len(responses) > 0:
            self.responsesSelected.emit(responses)
        super().accept()

    def reset(self) -> None:
        data = np.array([], dtype=[("<element>", np.float64)])
        self.model.beginResetModel()
        self.model.array = np.full(1, np.nan, dtype=data.dtype)
        self.model.endResetModel()
        self.responses = self.model.array.copy()

        self.import_options = None
        self.graph.clear()
        self.graph_cal.clear()
