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

        self.table = QtWidgets.QTableWidget(len(info), 2)
        self.table.setEditTriggers(QtWidgets.QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )

        for i, (key, val) in enumerate(info.items()):
            item = QtWidgets.QTableWidgetItem(key)
            self.table.setItem(i, 0, item)
            item = QtWidgets.QTableWidgetItem(val)
            self.table.setItem(i, 1, item)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.table)
        self.setLayout(layout)

    def sizeHint(self) -> QtCore.QSize:
        height = self.table.rowHeight(0) * self.table.rowCount()
        width = self.table.columnWidth(0) + self.table.columnWidth(1)
        return QtCore.QSize(width, height)


class SPCalDataFilesDock(QtWidgets.QDockWidget):
    dataFileAdded = QtCore.Signal(SPCalDataFile)
    dataFilesChanged = QtCore.Signal(SPCalDataFile, list)
    dataFileRemoved = QtCore.Signal(SPCalDataFile)

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
        self.list.selectionModel().currentChanged.connect(
            self.onCurrentOrSelectionChanged
        )
        self.list.selectionModel().selectionChanged.connect(
            self.onCurrentOrSelectionChanged
        )
        self.setWidget(self.list)

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

    def onCurrentOrSelectionChanged(self):
        current = self.currentDataFile()
        selected = self.selectedDataFiles()
        self.dataFilesChanged.emit(current, selected)

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
            return self.list.currentIndex().data(DataFileRole)
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
        selected = [index.data(DataFileRole) for index in self.list.selectedIndexes()]
        if len(selected) == 0:
            current = self.currentDataFile()
            return [current] if current is not None else []
        return selected

    def dataFiles(self) -> list[SPCalDataFile]:
        return self.model.data_files

    def dialogEditIsotopes(self, index: QtCore.QModelIndex):
        if self.screening_method is None:
            raise ValueError("screening method has not been set")
        dlg = SelectIsotopesDialog(
            index.data(DataFileRole), self.screening_method, parent=self
        )
        dlg.isotopesSelected.connect(self.onCurrentOrSelectionChanged)
        dlg.open()

    def dialogInformation(self, index: QtCore.QModelIndex):
        if not index.isValid():
            return
        dlg = DataFileInformationDialog(index.data(DataFileRole), parent=self)
        dlg.open()

    def reset(self):
        self.model.beginResetModel()
        self.model.data_files.clear()
        self.model.endResetModel()
