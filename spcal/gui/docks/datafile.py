from PySide6 import QtCore, QtWidgets

from spcal.datafile import SPCalDataFile
from spcal.gui.util import create_action


class DataFileListItemWidget(QtWidgets.QWidget):
    closeResquested = QtCore.Signal(QtWidgets.QWidget)

    def __init__(
        self, data_file: SPCalDataFile, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent)

        self.label_path = QtWidgets.QLabel(str(data_file.path))
        # self.label_path.setTextFormat
        self.label_stats = QtWidgets.QLabel(
            f"No. events: {data_file.num_events}, No. isotopes: {len(data_file.isotopes)}"
        )

        self.action_close = create_action(
            "window-close", "Close", "Close the data file.", self.closeButtonPressed
        )
        self.button_close = QtWidgets.QToolButton()
        self.button_close.setDefaultAction(self.action_close)

        layout = QtWidgets.QGridLayout()
        layout.setSpacing(0)
        layout.addWidget(self.label_path, 0, 0)
        layout.addWidget(self.label_stats, 1, 0)
        layout.addWidget(self.button_close, 0, 1, QtCore.Qt.AlignmentFlag.AlignRight)
        self.setLayout(layout)

    def closeButtonPressed(self) -> None:
        self.closeResquested.emit(self)


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
