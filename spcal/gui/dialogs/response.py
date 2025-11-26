import logging
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.calc import weighted_linreg
from spcal.gui.modelviews.basic import BasicTableView
from spcal.gui.modelviews.values import ValueWidgetDelegate
from spcal.datafile import SPCalDataFile
from spcal.gui.dialogs.io import ImportDialogBase
from spcal.gui.graphs import color_schemes
from spcal.gui.graphs.calibration import CalibrationView, IntensityView
from spcal.gui.io import (
    get_import_dialog_for_path,
    get_open_spcal_path,
    get_save_spcal_path,
    is_spcal_path,
)
from spcal.isotope import SPCalIsotope
from spcal.siunits import mass_concentration_units

logger = logging.getLogger(__name__)


class ConcentrationModel(QtCore.QAbstractTableModel):
    DataFileRole = QtCore.Qt.ItemDataRole.UserRole + 1
    IsotopeRole = QtCore.Qt.ItemDataRole.UserRole + 2

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self.isotopes: list[SPCalIsotope] = []
        self.concentrations: dict[SPCalDataFile, dict[SPCalIsotope, float | None]] = {}

    def columnCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        return len(self.isotopes)

    def rowCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        return len(self.concentrations)

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ):
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if orientation == QtCore.Qt.Orientation.Horizontal:
                return str(self.isotopes[section])
            else:
                return list(self.concentrations.keys())[section].path.stem

    def flags(
        self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex
    ) -> QtCore.Qt.ItemFlag:
        flags = super().flags(index)
        if index.isValid():
            data_file = list(self.concentrations.keys())[index.row()]
            isotope = self.isotopes[index.column()]
            if isotope in data_file.selected_isotopes:
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
            data_file = list(self.concentrations.keys())[index.row()]
            isotope = self.isotopes[index.column()]
            return self.concentrations[data_file].get(isotope, None)
        elif role == ConcentrationModel.DataFileRole:
            return list(self.concentrations.keys())[index.row()]
        elif role == ConcentrationModel.IsotopeRole:
            return self.isotopes[index.column()]

    def setData(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        value: Any,
        role: int = QtCore.Qt.ItemDataRole.EditRole,
    ) -> bool:
        if not index.isValid():
            return False
        if QtCore.Qt.ItemDataRole.EditRole:
            data_file = list(self.concentrations.keys())[index.row()]
            isotope = self.isotopes[index.column()]
            if value is not None:
                value = float(value)
            self.concentrations[data_file][isotope] = value
            self.dataChanged.emit(index, index)
            return True

        return False


class IntensityModel(QtCore.QAbstractTableModel):
    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self.isotopes: list[SPCalIsotope] = []
        self.intensities: dict[SPCalDataFile, dict[SPCalIsotope, float | None]] = {}

    def columnCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        return len(self.isotopes)

    def rowCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        return len(self.intensities)

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ):
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if orientation == QtCore.Qt.Orientation.Horizontal:
                return str(self.isotopes[section])
            else:
                return list(self.intensities.keys())[section].path.stem

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
            data_file = list(self.intensities.keys())[index.row()]
            isotope = self.isotopes[index.column()]
            if isotope not in self.intensities[data_file]:
                if isotope in data_file.selected_isotopes:
                    val = float(np.nanmean(data_file[isotope]))
                else:
                    val = None
                self.intensities[data_file][isotope] = val
            return self.intensities[data_file][isotope]
        elif role == ConcentrationModel.DataFileRole:
            return list(self.intensities.keys())[index.row()]
        elif role == ConcentrationModel.IsotopeRole:
            return self.isotopes[index.column()]


class ResponseDialog(QtWidgets.QDialog):
    responsesSelected = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent)
        self.setWindowTitle("Ionic Response Calculator")
        self.setMinimumSize(640, 480)

        self.intensity = IntensityView()
        # self.graph.region.sigRegionChangeFinished.connect(self.updateResponses)

        self.calibration = CalibrationView()

        self.model_concs = ConcentrationModel()
        self.model_intensity = IntensityModel()

        sf = int(QtCore.QSettings().value("SigFigs", 4))  # type: ignore

        self.table_concs = BasicTableView()
        self.table_concs.setModel(self.model_concs)

        self.table_intensity = BasicTableView()
        self.table_intensity.setModel(self.model_intensity)

        for table in [self.table_concs, self.table_intensity]:
            table.setItemDelegate(ValueWidgetDelegate(sigfigs=sf))
            table.selectionModel().currentChanged.connect(self.tableIndexChanged)

            table.setSizeAdjustPolicy(
                QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents
            )

        self.model_concs.dataChanged.connect(self.completeChanged)
        self.model_concs.dataChanged.connect(self.updateCalibration)

        self.button_add_level = QtWidgets.QPushButton("Add Level")
        self.button_add_level.setIcon(QtGui.QIcon.fromTheme("list-add"))
        self.button_add_level.pressed.connect(self.dialogLoadFile)

        self.combo_unit = QtWidgets.QComboBox()
        self.combo_unit.addItems(list(mass_concentration_units.keys()))
        self.combo_unit.setCurrentText("µg/L")

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save
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
        box_concs_layout.addWidget(self.table_concs, 1)
        box_concs_layout.addLayout(layout_conc_bar, 0)
        box_concs.setLayout(box_concs_layout)

        box_intensity = QtWidgets.QGroupBox("Intensities")
        box_intensity_layout = QtWidgets.QVBoxLayout()
        box_intensity_layout.addWidget(self.table_intensity, 1)
        box_intensity.setLayout(box_intensity_layout)

        table_layout = QtWidgets.QHBoxLayout()
        table_layout.addWidget(box_concs)
        table_layout.addWidget(box_intensity)

        layout_graphs = QtWidgets.QHBoxLayout()
        layout_graphs.addWidget(self.intensity, 3)
        layout_graphs.addWidget(self.calibration, 2)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(layout_graphs, 3)
        layout.addLayout(table_layout, 2)
        layout.addWidget(self.button_box, 0)
        self.setLayout(layout)

    def isComplete(self) -> bool:
        if self.model_concs.rowCount() == 0:
            return False

        return any(
            np.count_nonzero(~np.isnan(concs)) > 0
            for concs in self.concentrations().values()
        )

    def completeChanged(self):
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            self.isComplete()
        )
        self.button_box.button(
            QtWidgets.QDialogButtonBox.StandardButton.Save
        ).setEnabled(self.isComplete())

    def concentrations(self) -> dict[SPCalIsotope, np.ndarray]:
        concs = {}
        for col, isotope in enumerate(self.model_concs.isotopes):
            concs[isotope] = np.array(
                [
                    self.model_concs.data(self.model_concs.index(row, col))
                    for row in range(self.model_concs.rowCount())
                ],
                dtype=float,
            )
        return concs

    def intensities(self) -> dict[SPCalIsotope, np.ndarray]:
        intensity = {}
        for col, isotope in enumerate(self.model_intensity.isotopes):
            intensity[isotope] = np.array(
                [
                    self.model_intensity.data(self.model_intensity.index(row, col))
                    for row in range(self.model_intensity.rowCount())
                ],
                dtype=float,
            )
        return intensity

    def tableIndexChanged(self, index: QtCore.QModelIndex):
        self.table_concs.selectionModel().currentChanged.disconnect(
            self.tableIndexChanged
        )
        self.table_intensity.selectionModel().currentChanged.disconnect(
            self.tableIndexChanged
        )

        self.table_concs.selectionModel().setCurrentIndex(
            index, QtCore.QItemSelectionModel.SelectionFlag.SelectCurrent
        )
        self.table_intensity.selectionModel().setCurrentIndex(
            index, QtCore.QItemSelectionModel.SelectionFlag.SelectCurrent
        )

        self.table_concs.selectionModel().currentChanged.connect(self.tableIndexChanged)
        self.table_intensity.selectionModel().currentChanged.connect(
            self.tableIndexChanged
        )
        self.drawDataFile(index)

    def drawDataFile(self, index: QtCore.QModelIndex):
        self.intensity.clear()
        if not index.isValid():
            return

        data_file = index.data(ConcentrationModel.DataFileRole)
        isotope = index.data(ConcentrationModel.IsotopeRole)

        # Draw the trace
        self.intensity.drawCurve(data_file.times, data_file[isotope])

        pen = QtGui.QPen(QtCore.Qt.GlobalColor.red, 1.0)
        pen.setCosmetic(True)

        # Draw mean line
        self.intensity.drawLine(
            np.nanmean(data_file[isotope]), QtCore.Qt.Orientation.Horizontal, pen=pen
        )
        # Rescale
        self.intensity.setDataLimits(0.0, 1.0)

    def dialogLoadFile(self, path: str | Path | None = None) -> ImportDialogBase | None:
        if path is None:
            path = get_open_spcal_path(self)
            if path is None:
                return None
        else:
            path = Path(path)

        existing = (
            None
            if len(self.model_concs.concentrations) == 0
            else list(self.model_concs.concentrations.keys())[-1]
        )
        dlg = get_import_dialog_for_path(self, path, existing)
        dlg.dataImported.connect(self.addDataFile)

        dlg.open()

        return dlg

    def addDataFile(self, data_file: SPCalDataFile):
        new_isotopes = sorted(
            set(data_file.selected_isotopes).union(self.model_concs.isotopes),
            key=lambda i: i.isotope,
        )
        self.model_concs.beginResetModel()
        self.model_concs.concentrations[data_file] = {}
        self.model_concs.isotopes = new_isotopes
        self.model_concs.endResetModel()
        self.model_intensity.beginResetModel()
        self.model_intensity.intensities[data_file] = {}
        self.model_intensity.isotopes = new_isotopes
        self.model_intensity.endResetModel()

    def updateCalibration(self):
        self.calibration.clear()

        scheme = color_schemes[QtCore.QSettings().value("colorscheme", "IBM Carbon")]  # type: ignore

        concs = self.concentrations()
        intensities = self.intensities()

        for i, isotope in enumerate(concs.keys()):
            nans = np.logical_or(
                np.isnan(concs[isotope]), np.isnan(intensities[isotope])
            )
            x, y = concs[isotope][~nans], intensities[isotope][~nans]
            if x.size == 0:
                continue
            elif x.size == 1:
                x = np.concatenate(([0.0], x))
                y = np.concatenate(([0.0], y))

            brush = QtGui.QBrush(scheme[i % len(scheme)])

            scatter = self.calibration.drawScatter(
                x, y, size=6.0 * self.devicePixelRatio(), brush=brush
            )

            pen = QtGui.QPen(scheme[i % len(scheme)], 1.0 * self.devicePixelRatio())
            pen.setCosmetic(True)
            self.calibration.drawTrendline(x, y, pen=pen)

            if self.calibration.plot.legend is not None:
                self.calibration.plot.legend.addItem(str(isotope), scatter)

    def calibrationResult(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[float | None, float | None, float | None, float | None]:
        factor = mass_concentration_units[self.combo_unit.currentText()]

        nans = np.logical_or(np.isnan(x), np.isnan(y))
        x, y = x[~nans] * factor, y[~nans]
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
        elif sb == QtWidgets.QDialogButtonBox.StandardButton.Reset:
            self.reset()

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

    def accept(self):
        responses = {}
        concs = self.concentrations()
        intensities = self.intensities()
        for isotope in concs.keys():
            response, _, _, _ = self.calibrationResult(
                concs[isotope], intensities[isotope]
            )
            if response is not None:
                responses[isotope] = response

        if len(responses) > 0:
            self.responsesSelected.emit(responses)
        super().accept()

    def reset(self):
        self.model_concs.beginResetModel()
        self.model_concs.isotopes.clear()
        self.model_concs.concentrations.clear()
        self.model_concs.endResetModel()
        self.model_intensity.beginResetModel()
        self.model_intensity.isotopes.clear()
        self.model_intensity.intensities.clear()
        self.model_intensity.endResetModel()

    def dialogSaveToFile(self):
        if len(self.model_concs.concentrations) == 0:
            return
        dir = next(iter(self.model_concs.concentrations.keys())).path.parent
        path = get_save_spcal_path(self, [("CSV Documents", ".csv")], dir=dir)
        if path is None:
            return
        self.saveToFile(path)

    def saveToFile(self, path: Path):
        concs = self.concentrations()
        intensities = self.intensities()
        factor = mass_concentration_units[self.combo_unit.currentText()]

        with path.open("w") as fp:
            fp.write(f"#SPCal Calibration {version('spcal')}\n")
            fp.write("Isotope,Slope,Intercept,r2,Error\n")
            for isotope in concs.keys():
                m, b, r2, err = self.calibrationResult(
                    concs[isotope] * factor, intensities[isotope]
                )
                fp.write(f"{isotope},{m or ''},{b or ''},{r2 or ''},{err or ''},\n")

            fp.write(
                "#Concentrations (kg/L),"
                + ",".join(str(i) for i in range(len(concs)))
                + "\n"
            )
            for isotope, vals in concs.items():
                scaled = vals * factor
                fp.write(
                    f"{isotope},"
                    + ",".join("" if np.isnan(x) else str(x) for x in scaled)
                    + "\n"
                )
            fp.write(
                "#Intensities (cts),"
                + ",".join(str(i) for i in range(len(intensities)))
                + "\n"
            )
            for isotope, vals in intensities.items():
                fp.write(
                    f"{isotope},"
                    + ",".join("" if np.isnan(x) else str(x) for x in vals)
                    + "\n"
                )


if __name__ == "__main__":
    from spcal.datafile import SPCalNuDataFile

    app = QtWidgets.QApplication()

    dlg = ResponseDialog()
    dlg.show()

    df = SPCalNuDataFile.load(Path("/home/tom/Downloads/NT032/14-37-30 1 ppb att"))
    df.selected_isotopes = [df.isotopes[10], df.isotopes[20]]
    dlg.addDataFile(df)
    df = SPCalNuDataFile.load(Path("/home/tom/Downloads/NT032/14-36-31 10 ppb att"))
    df.selected_isotopes = [df.isotopes[10], df.isotopes[20], df.isotopes[30]]
    dlg.addDataFile(df)

    app.exec()
