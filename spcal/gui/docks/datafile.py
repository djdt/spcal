from PySide6 import QtCore, QtWidgets

from spcal.datafile import SPCalDataFile
from spcal.gui.dialogs.selectisotope import SelectIsotopesDialog
from spcal.gui.modelviews.datafile import DataFileDelegate, DataFileModel


class SPCalDataFilesDock(QtWidgets.QDockWidget):
    dataFileChanged = QtCore.Signal(SPCalDataFile)
    dataFileAdded = QtCore.Signal(SPCalDataFile)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Open Data Files")

        self.list = QtWidgets.QListView()
        self.model = DataFileModel()
        self.model.editIsotopesRequested.connect(self.dialogEditIsotopes)
        self.list.setMouseTracking(True)
        self.list.setSelectionMode(QtWidgets.QListView.SelectionMode.NoSelection)
        self.list.setModel(self.model)
        self.list.setItemDelegate(DataFileDelegate())
        self.list.selectionModel().currentChanged.connect(self.onCurrentIndexChanged)
        self.setWidget(self.list)

    def onCurrentIndexChanged(self, index :QtCore.QModelIndex):
        self.dataFileChanged.emit(index.data(DataFileModel.DataFileRole))

    def currentDataFile(self) -> SPCalDataFile:
        return self.list.currentIndex().data(DataFileModel.DataFileRole)

    def addDataFile(self, data_file: SPCalDataFile):
        self.model.beginInsertRows(
            QtCore.QModelIndex(), self.model.rowCount(), self.model.rowCount() + 1
        )
        self.model.data_files.append(data_file)
        self.model.endInsertRows()
        self.list.setCurrentIndex(self.model.index(self.model.rowCount() - 1, 0))
        self.dataFileAdded.emit(data_file)

    def dataFiles(self) -> list[SPCalDataFile]:
        return self.model.data_files

    def dialogEditIsotopes(self, index: QtCore.QModelIndex) -> QtWidgets.QDialog:
        dlg = SelectIsotopesDialog(index.data(DataFileModel.DataFileRole), parent=self)
        dlg.isotopesSelected.connect(self.dataFileChanged)
        dlg.open()
        return dlg
