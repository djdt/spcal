import logging
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import numpy.lib.recfunctions as rfn
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.calc import weighted_linreg
from spcal.datafile import SPCalDataFile
from spcal.gui.dialogs.io import ImportDialogBase
from spcal.gui.graphs import color_schemes
from spcal.gui.graphs.calibration import CalibrationView
from spcal.gui.graphs.particle import ParticleView
from spcal.gui.io import get_import_dialog_for_path, get_open_spcal_path, is_spcal_path
from spcal.isotope import SPCalIsotope
from spcal.siunits import mass_concentration_units

logger = logging.getLogger(__name__)


class ResponseModel(QtCore.QAbstractTableModel):
    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self.data_files: list[SPCalDataFile] = []
        self.isotopes: list[SPCalIsotope] = []
        self.concetrations: list[float] = []

    def columnCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        return len(self.isotopes) + 1

    def rowCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        return len(self.data_files)

    def flags(
        self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex
    ) -> QtCore.Qt.ItemFlag:
        flags = super().flags(index)
        if index.isValid():
            df = self.data_files[index.row()]
            if self.isotopes[index.column() - 1] not in df.isotopes:
                flags &= ~QtCore.Qt.ItemFlag.ItemIsEnabled
        if index.column() == 0:
            flags |= QtCore.Qt.ItemFlag.ItemIsEditable
        return flags

    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.EditRole,
    ) -> Any:
        if not index.isValid():
            return None
        if role in [
            QtCore.Qt.ItemDataRole.DisplayRole,
            QtCore.Qt.ItemDataRole.EditRole,
        ]:
            if index.column() == 0:
                return self.concetrations[index.row()]
            else:
                return float(
                    np.nanmean(
                        self.data_files[index.row()][self.isotopes[index.column() - 1]]
                    )
                )


class ResponseDialog(QtWidgets.QDialog):
    responsesSelected = QtCore.Signal(dict)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent)
        self.setWindowTitle("Ionic Response Calculator")
        self.setMinimumSize(640, 480)

        self.button_open_file = QtWidgets.QPushButton("Open File")
        self.button_open_file.pressed.connect(self.dialogLoadFile)

        self.graph = ParticleView()
        # self.graph.region.sigRegionChangeFinished.connect(self.updateResponses)

        self.graph_cal = CalibrationView()
        # self.graph_cal.sizeHint = lambda: QtCore.QSize(300, 300)

        self.model = ResponseModel()

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
            QtWidgets.QDialogButtonBox.StandardButton.Open
            | QtWidgets.QDialogButtonBox.StandardButton.Save
            | QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Reset
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
        )
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            False
        )
        self.button_box.button(
            QtWidgets.QDialogButtonBox.StandardButton.Save
        ).setEnabled(False)

        self.button_box.clicked.connect(self.buttonClicked)
        self.button_box.rejected.connect(self.reject)

        layout_conc_bar = QtWidgets.QHBoxLayout()
        layout_conc_bar.addStretch(1)
        layout_conc_bar.addWidget(
            self.button_add_level, 0, QtCore.Qt.AlignmentFlag.AlignRight
        )
        layout_conc_bar.addWidget(
            self.combo_unit, 0, QtCore.Qt.AlignmentFlag.AlignRight
        )

        box_concs = QtWidgets.QGroupBox("Concentrations")
        box_concs_layout = QtWidgets.QVBoxLayout()
        box_concs_layout.addWidget(self.table, 1)
        box_concs_layout.addLayout(layout_conc_bar, 0)
        box_concs.setLayout(box_concs_layout)

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

    def completeChanged(self):
        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(
            self.isComplete()
        )
        self.button_box.button(QtWidgets.QDialogButtonBox.Save).setEnabled(
            self.isComplete()
        )

    def dropEvent(self, event: QtGui.QDropEvent):
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

    def dialogLoadFile(self, path: str | Path | None = None) -> ImportDialogBase | None:
        if path is None:
            path = get_open_spcal_path(self)
            if path is None:
                return None
        else:
            path = Path(path)

        existing = None if len(self.model.data_files) == 0 else self.model.data_files[0]
        dlg = get_import_dialog_for_path(
            self, path, existing
        )  # import_options managed below
        dlg.dataImported.connect(self.loadDataFile)

        if existing is None:
            dlg.open()
        else:
            try:
                dlg.accept()
            except Exception:
                logger.warning("dialogLoadFile: unable to set import options.")
                dlg.open()

        return dlg

    def loadDataFile(self, data_file: SPCalDataFile):
        # self.model.beginInsertRows(
        #     QtCore.QModelIndex(), self.model.rowCount(), self.model.rowCount() + 1
        # )
        self.model.beginResetModel()
        self.model.data_files.append(data_file)
        self.model.concetrations.append(0.0)
        all_isotopes = set(
            iso for df in self.model.data_files for iso in df.selected_isotopes
        )
        print(all_isotopes)
        self.model.isotopes = list(all_isotopes)
        self.model.endResetModel()
        # self.model.endInsertRows()

        # Check the new data is compatible with current loaded
        # if self.model.array.size == 0:
        #     self.model.beginResetModel()
        #     self.model.array = np.full(1, np.nan, dtype=data.dtype)
        #     self.model.endResetModel()
        #     self.responses = self.model.array.copy()
        #
        # elif (
        #     data.dtype.names != self.model.array.dtype.names
        # ):  # pragma: no cover, can't test msgbox
        #     button = QtWidgets.QMessageBox.question(
        #         self, "Warning", "New data does not match current, overwrite?"
        #     )
        #     if button == QtWidgets.QMessageBox.StandardButton.Yes:
        #         self.model.beginResetModel()
        #         self.model.array = np.full(1, np.nan, dtype=data.dtype)
        #         self.model.endResetModel()
        #         self.responses = self.model.array.copy()
        #     else:
        #         return
        # else:
        #     self.model.insertColumn(self.model.columnCount())
        #     self.responses = np.append(
        #         self.responses, np.full(1, np.nan, self.responses.dtype)
        #     )
        #
        # if self.import_options is None:
        #     self.import_options = options
        #
        # old_size = self.data.size
        # self.data = data
        # tic = np.sum([data[name] for name in data.dtype.names], axis=0)
        # xs = np.arange(tic.size)
        #
        # self.graph.clear()
        # self.graph.plot.setTitle(f"TIC: {options['path'].name}")
        # self.graph.drawData(xs, tic)
        # if old_size != data.size:
        #     self.graph.region.blockSignals(True)
        #     self.graph.region.setRegion((xs[0], xs[-1]))
        #     self.graph.region.blockSignals(False)
        # self.graph.updateMean()
        #
        # self.updateResponses()

    def updateResponses(self):
        if self.responses.dtype.names is None:  # pragma: no cover
            return

        for name in self.responses.dtype.names:
            self.responses[name][-1] = np.mean(
                self.data[name][self.graph.region_start : self.graph.region_end]
            )

        self.updateCalibration()

    def updateCalibration(self):
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

    def dialogSaveToFile(self):
        assert self.import_options is not None
        dir = self.import_options["path"].parent
        file, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Calibration", str(dir), "CSV Documents(*.csv);;All Files(*)"
        )
        if file == "":
            return
        self.saveToFile(Path(file))

    def saveToFile(self, path: Path):
        assert self.responses.dtype.names is not None
        names = [  # remove any unpopulated names
            name
            for name in self.responses.dtype.names
            if np.any(~np.isnan(self.model.array[name]))
        ]
        nlevels = len(self.model.array)
        factor = mass_concentration_units[self.combo_unit.currentText()]

        def write_cal_levels(fp, name: str):
            fp.write(name + "," + ",".join(str(i) for i in range(len(xs))) + "\n")

        with path.open("w") as fp:
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
                ys = self.responses[name]
                fp.write(
                    f"{name},"
                    + ",".join("" if np.isnan(y) else str(y) for y in ys)
                    + "\n"
                )

    def dialogLoadFromFile(self):
        if self.import_options is not None:
            dir = self.import_options["path"].parent
        else:
            dir = ""
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Calibration", str(dir), "CSV Documents(*.csv);;All Files(*)"
        )
        if file == "":
            return
        self.loadFromFile(Path(file))

    def loadFromFile(self, path: Path):
        factor = mass_concentration_units[self.combo_unit.currentText()]

        concs = {}
        responses = {}

        with path.open("r") as fp:
            line = fp.readline()
            if not line.startswith("#SPCal Calibration"):
                raise ValueError("file is not valid SPCal calibration")
            while not line.startswith("#Concentrations"):
                line = fp.readline()
            line = fp.readline()
            while not line.startswith("#Responses"):
                name, *xs = line.split(",")
                concs[name] = (
                    np.array([float(x) if x != "" else np.nan for x in xs]) / factor
                )
                line = fp.readline().strip()
            line = fp.readline()
            while line:
                name, *ys = line.split(",")
                responses[name] = np.array(
                    [float(y) if y != "" else np.nan for y in ys]
                )
                line = fp.readline().strip()

        array = np.stack(list(responses.values()), axis=1)
        self.responses = rfn.unstructured_to_structured(array, names=responses.keys())
        self.model.beginResetModel()

        array = np.stack(list(concs.values()), axis=1)
        self.model.array = rfn.unstructured_to_structured(array, names=concs.keys())
        self.model.endResetModel()
        self.updateCalibration()

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

    def buttonClicked(self, button: QtWidgets.QAbstractButton):
        sb = self.button_box.standardButton(button)
        if sb == QtWidgets.QDialogButtonBox.StandardButton.Ok:
            self.accept()
        elif sb == QtWidgets.QDialogButtonBox.StandardButton.Save:
            self.dialogSaveToFile()
        elif sb == QtWidgets.QDialogButtonBox.StandardButton.Open:
            self.dialogLoadFromFile()
        elif sb == QtWidgets.QDialogButtonBox.StandardButton.Reset:
            self.reset()

    def accept(self):
        assert self.responses.dtype.names is not None

        responses = {}
        for name in self.responses.dtype.names:
            m, _, _, _ = self.calibrationResult(name)
            if m is not None:
                responses[name] = m

        if len(responses) > 0:
            self.responsesSelected.emit(responses)
        super().accept()

    def reset(self):
        data = np.array([], dtype=[("<element>", np.float64)])
        self.model.beginResetModel()
        self.model.array = np.full(1, np.nan, dtype=data.dtype)
        self.model.endResetModel()
        self.responses = self.model.array.copy()

        self.import_options = None
        self.graph.clear()
        self.graph_cal.clear()


if __name__ == "__main__":
    from spcal.datafile import SPCalNuDataFile

    app = QtWidgets.QApplication()

    dlg = ResponseDialog()
    dlg.show()

    df = SPCalNuDataFile.load(
        Path("/home/tom/Downloads/14-38-58 UPW + 80nm Au 90nm UCNP many particles")
    )
    df.selected_isotopes = [df.isotopes[10], df.isotopes[20]]
    dlg.loadDataFile(df)

    app.exec()
