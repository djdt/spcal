from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.util import create_action
from spcal.datafile import SPCalDataFile
from spcal.gui.dialogs.selectisotope import SelectIsotopesDialog
from spcal.gui.modelviews import DataFileRole
from spcal.gui.modelviews.datafile import DataFileDelegate, DataFileModel
from spcal.processing.method import SPCalProcessingMethod


class DataFileInformationDialog(QtWidgets.QDialog):
    def __init__(
        self, data_file: SPCalDataFile, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent)

        self.setWindowTitle(f"{data_file.path.name} Information")
        self.resize(800, 600)

        info = data_file.information()

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(2)
        for group_key in info.keys():
            group_item = QtWidgets.QTreeWidgetItem([group_key])

            for key, val in info[group_key].items():
                item = QtWidgets.QTreeWidgetItem([key, val])
                group_item.addChild(item)

            self.tree.addTopLevelItem(group_item)

        self.tree.expandAll()
        self.tree.resizeColumnToContents(0)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tree)
        self.setLayout(layout)


class SPCalDataFilesDock(QtWidgets.QDockWidget):
    dataFileAdded = QtCore.Signal(SPCalDataFile)
    dataFilesChanged = QtCore.Signal(SPCalDataFile)
    dataFileRemoved = QtCore.Signal(SPCalDataFile)

    currentDataFileChanged = QtCore.Signal(SPCalDataFile)
    selectedDataFilesChanged = QtCore.Signal(list)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("spcal-datafiles-dock")
        self.setWindowTitle("Data Files")

        self.screening_method: SPCalProcessingMethod | None = None

        self.model = DataFileModel()
        self.model.editIsotopesRequested.connect(self.dialogEditIsotopes)
        self.model.rowsAboutToBeRemoved.connect(self.onRowsRemoved)

        self.list = QtWidgets.QListView()
        self.list.setMouseTracking(True)
        self.list.setSelectionMode(QtWidgets.QListView.SelectionMode.ExtendedSelection)
        self.list.setModel(self.model)
        self.list.setItemDelegate(DataFileDelegate())
        self.list.selectionModel().selectionChanged.connect(self.onSelectionChanged)
        self.setWidget(self.list)

    def activeDataFiles(self) -> list[SPCalDataFile]:
        files = self.selectedDataFiles()
        if len(files) == 0:
            current = self.currentDataFile()
            return [current] if current is not None else []
        return files

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        pos = self.list.viewport().mapFromGlobal(event.globalPos())
        index = self.list.indexAt(pos)
        if not index.isValid():
            return super().contextMenuEvent(event)

        menu = QtWidgets.QMenu(self)

        action_edit_isotopes = create_action(
            "view-list-icons",
            "Select Isotopes",
            "Open a dialog to select isotopes from the data file.",
            lambda: self.dialogEditIsotopes(index),
        )
        action_edit_isotopes.setParent(self)

        action_information = create_action(
            "info",
            "Information",
            "Display information about the data file.",
            lambda: self.dialogInformation(index),
        )
        action_information.setParent(self)

        action_close = create_action(
            "view-close",
            "Close Data File",
            "Close the selected data file.",
            lambda: self.model.removeRow(index.row()),
        )
        action_close.setParent(self)

        menu.addAction(action_edit_isotopes)
        menu.addAction(action_information)
        menu.addSeparator()
        menu.addAction(action_close)
        menu.popup(event.globalPos())
        event.accept()

    def dataFiles(self) -> list[SPCalDataFile]:
        return self.model.data_files

    def setDataFiles(self, data_files: list[SPCalDataFile]):
        self.model.beginResetModel()
        self.model.data_files = data_files
        self.model.endResetModel()

    def onCurrentChanged(self, index: QtCore.QModelIndex):
        self.currentDataFileChanged.emit(index.data(DataFileRole))

    def onSelectionChanged(self):
        self.selectedDataFilesChanged.emit(self.selectedDataFiles())

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

    def currentDataFile(self) -> SPCalDataFile | None:
        index = self.list.currentIndex()
        if index.isValid():
            return index.data(DataFileRole)
        else:
            return None

    def selectedDataFiles(self) -> list[SPCalDataFile]:
        return list(set(idx.data(DataFileRole) for idx in self.list.selectedIndexes()))

    def dialogEditIsotopes(self, index: QtCore.QModelIndex):
        if self.screening_method is None:
            raise ValueError("screening method has not been set")
        file = index.data(DataFileRole)
        dlg = SelectIsotopesDialog(file, self.screening_method, parent=self)
        dlg.isotopesSelected.connect(lambda: self.dataFilesChanged.emit(file))
        dlg.open()

    def dialogInformation(self, index: QtCore.QModelIndex):
        if not index.isValid():
            return
        dlg = DataFileInformationDialog(index.data(DataFileRole), parent=self)
        dlg.open()

    def onRowsRemoved(self, index: QtCore.QModelIndex, first: int, last: int):
        if index.isValid():
            raise ValueError("valid index for removed row")
        if first <= self.list.currentIndex().row() < last:
            self.list.selectionModel().setCurrentIndex(
                self.model.index(first - 1, 0),
                QtCore.QItemSelectionModel.SelectionFlag.Current,
            )
        for row in range(first, last):
            self.dataFileRemoved.emit(self.model.data_files[row])

    def clear(self):
        self.setDataFiles([])
