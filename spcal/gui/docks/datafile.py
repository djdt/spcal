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
        indicies = self.list.selectedIndexes()
        if len(indicies) != 1:
            return None
        return self.data_files[indicies[0].row()]

    def closeDataFile(self, widget: QtWidgets.QWidget) -> None:
        for i in range(self.list.count()):
            item = self.list.item(i)
            if widget == self.list.itemWidget(item):
                self.list.takeItem(i)
                self.data_files.pop(i)
                break

    def addDataFile(self, data_file):
        self.data_files.append(data_file)

        widget = DataFileListItemWidget(data_file)
        widget.closeResquested.connect(self.closeDataFile)
        item = QtWidgets.QListWidgetItem()
        item.setSizeHint(widget.sizeHint())
        self.list.addItem(item)
        self.list.setItemWidget(item, widget)

    def onSelectionChanged(self) -> None:
        print(self.selectedDataFile())
        self.dataFileSelected.emit(self.selectedDataFile())
