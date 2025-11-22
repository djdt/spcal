from PySide6 import QtCore, QtWidgets

from spcal.datafile import SPCalDataFile
from spcal.gui.dialogs.selectisotope import SelectIsotopesDialog
from spcal.gui.modelviews.datafile import DataFileDelegate, DataFileModel
from spcal.processing import SPCalProcessingMethod


class SPCalDataFilesDock(QtWidgets.QDockWidget):
    dataFileAdded = QtCore.Signal(SPCalDataFile)
    currentDataFileChanged = QtCore.Signal(SPCalDataFile)
    selectedDataFilesChanged = QtCore.Signal()
    dataFileRemoved = QtCore.Signal(SPCalDataFile)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Open Data Files")

        self.screening_method: SPCalProcessingMethod | None = None

        self.model = DataFileModel()
        self.model.editIsotopesRequested.connect(self.dialogEditIsotopes)
        self.model.rowsAboutToBeRemoved.connect(self.onRowsRemoved)

        self.list = QtWidgets.QListView()
        self.list.setMouseTracking(True)
        self.list.setSelectionMode(QtWidgets.QListView.SelectionMode.ExtendedSelection)
        self.list.setModel(self.model)
        self.list.setItemDelegate(DataFileDelegate())
        self.list.selectionModel().currentChanged.connect(self.onCurrentIndexChanged)
        self.list.selectionModel().selectionChanged.connect(
            self.selectedDataFilesChanged
        )

        self.setWidget(self.list)

    def onCurrentIndexChanged(self, index: QtCore.QModelIndex):
        if index.isValid():
            self.currentDataFileChanged.emit(index.data(DataFileModel.DataFileRole))
        else:
            self.currentDataFileChanged.emit(None)

    def onRowsRemoved(self, index: QtCore.QModelIndex, first: int, last: int):
        if first <= self.list.currentIndex().row() < last:
            self.list.selectionModel().setCurrentIndex(
                self.model.index(first - 1, 0),
                QtCore.QItemSelectionModel.SelectionFlag.Current,
            )
        for row in range(first, last):
            self.dataFileRemoved.emit(self.model.data_files[row])

    def currentDataFile(self) -> SPCalDataFile | None:
        index = self.list.currentIndex()
        if index.isValid():
            return self.list.currentIndex().data(DataFileModel.DataFileRole)
        else:
            return None

    @QtCore.Slot()
    def setScreeningMethod(self, method: SPCalProcessingMethod):
        self.screening_method = method

    @QtCore.Slot()
    def addDataFile(self, data_file: SPCalDataFile):
        self.model.beginInsertRows(
            QtCore.QModelIndex(), self.model.rowCount(), self.model.rowCount() + 1
        )
        self.model.data_files.append(data_file)
        self.model.endInsertRows()
        self.dataFileAdded.emit(data_file)

        self.list.setCurrentIndex(self.model.index(self.model.rowCount() - 1, 0))
        self.list.selectionModel().select(
            self.model.index(self.model.rowCount() - 1, 0),
            QtCore.QItemSelectionModel.SelectionFlag.Select,
        )

    def selectedDataFiles(self) -> list[SPCalDataFile]:
        return [
            index.data(DataFileModel.DataFileRole)
            for index in self.list.selectedIndexes()
        ]

    def dataFiles(self) -> list[SPCalDataFile]:
        return self.model.data_files

    def dialogEditIsotopes(self, index: QtCore.QModelIndex) -> QtWidgets.QDialog:
        if self.screening_method is None:
            raise ValueError("screening method has not been set")
        dlg = SelectIsotopesDialog(
            index.data(DataFileModel.DataFileRole), self.screening_method, parent=self
        )
        dlg.isotopesSelected.connect(self.currentDataFileChanged)
        dlg.open()
        return dlg
