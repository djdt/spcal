from PySide6 import QtCore, QtGui, QtWidgets

from spcal.datafile import SPCalDataFile, SPCalNuDataFile, SPCalTOFWERKDataFile
from spcal.gui.util import create_action
from spcal.gui.widgets import PeriodicTableSelector
from spcal.npdb import db


# class IsotopeSelectionDialog(QtWidgets.QDialog):
#     def __init__(
#         self, data_file: SPCalDataFile, parent: QtWidgets.QWidget | None = None
#     ):
#         super().__init__(parent)
#         self.setWindowTitle("Isotope Selection")
#
#         self.table = PeriodicTableSelector()
#
#         # if isinstance(data_file, SPCalNuDataFile):
#         #     enabled_isotopes = list(iso for _, iso in data_file.isotope_table.keys())
#         # elif isinstance(data_file, SPCalTOFWERKDataFile):
#         selected_isotopes = []
#         enabled_isotopes = []
#         for isotope in data_file.isotopes:
#             db["isotope"][np.logical_and]
#
#
#     def setSelectedIsotopes(self, selected: np.ndarray) -> None:
#

class DataFileListItemWidget(QtWidgets.QWidget):
    selectIsotopesRequested = QtCore.Signal(QtWidgets.QWidget)
    closeResquested = QtCore.Signal(QtWidgets.QWidget)

    def __init__(
        self, data_file: SPCalDataFile, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent)

        self.action_close = create_action(
            "window-close",
            "Close",
            "Close the data file.",
            lambda: self.closeResquested.emit(self),
        )
        self.action_isotopes = create_action(
            "edit-select",
            "Select Isotopes",
            "Open the isotope selection dialog.",
            lambda: self.selectIsotopesRequested.emit(self),
        )
        self.enable_isotopes = len(data_file.isotopes) > 1

        self.label_path = QtWidgets.QLabel(str(data_file.path))
        # self.label_path.setTextFormat
        self.label_stats = QtWidgets.QLabel(
            f"No. events: {data_file.num_events}, Isotopes: {len(data_file.selected_isotopes)}/{len(data_file.isotopes)}"
        )
        self.button_close = QtWidgets.QToolButton()
        self.button_close.setDefaultAction(self.action_close)

        layout = QtWidgets.QGridLayout()
        layout.setSpacing(0)
        layout.addWidget(self.label_path, 0, 0)
        layout.addWidget(self.label_stats, 1, 0)
        layout.addWidget(self.button_close, 0, 1, QtCore.Qt.AlignmentFlag.AlignRight)
        self.setLayout(layout)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        menu = QtWidgets.QMenu(self)

        if self.enable_isotopes:
            menu.addAction(self.action_isotopes)
        menu.addAction(self.action_close)

        menu.popup(event.globalPos())


class SPCalDataFilesDock(QtWidgets.QDockWidget):
    dataFileSelected = QtCore.Signal(SPCalDataFile)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Open Data Files")

        self.list = QtWidgets.QListWidget()
        self.list.itemSelectionChanged.connect(self.onSelectionChanged)
        self.data_files: list[SPCalDataFile] = []

        self.setWidget(self.list)

    def selectedDataFile(self) -> SPCalDataFile | None:
        if len(self.data_files) == 0:
            return None

        indicies = self.list.selectedIndexes()
        if len(indicies) == 0:
            return self.data_files[0]

        return self.data_files[indicies[0].row()]

    def closeDataFile(self, widget: QtWidgets.QWidget) -> None:
        for i in range(self.list.count()):
            item = self.list.item(i)
            if widget == self.list.itemWidget(item):
                self.list.takeItem(i)
                self.data_files.pop(i)
                break

        if len(self.list.selectedIndexes()) == 0:
            item = self.list.item(0)
            if item is not None:
                item.setSelected(True)

    def addDataFile(self, data_file: SPCalDataFile):
        self.data_files.append(data_file)

        widget = DataFileListItemWidget(data_file)
        widget.closeResquested.connect(self.closeDataFile)
        item = QtWidgets.QListWidgetItem()
        item.setSizeHint(widget.sizeHint())
        self.list.addItem(item)
        self.list.setItemWidget(item, widget)
        item.setSelected(True)

    def onSelectionChanged(self) -> None:
        self.dataFileSelected.emit(self.selectedDataFile())
