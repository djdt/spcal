from PySide2 import QtCore, QtGui, QtWidgets
import numpy as np

from nanopart.io import read_nanoparticle_file

from nanopart.gui.util import NumpyArrayTableModel

from typing import List, Tuple


class DoubleSignificantFiguresDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, sigfigs: int, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.sigfigs = sigfigs

    def createEditor(
        self,
        parent: QtWidgets.QWidget,
        option: QtWidgets.QStyleOptionViewItem,
        index: int,
    ) -> QtWidgets.QWidget:  # pragma: no cover
        lineedit = QtWidgets.QLineEdit(parent)
        lineedit.setValidator(QtGui.QDoubleValidator())
        return lineedit

    def displayText(self, value: str, locale: str) -> str:
        try:
            num = float(value)
            return f"{num:#.{self.sigfigs}g}".rstrip(".").replace(".e", "e")
        except (TypeError, ValueError):
            return str(super().displayText(value, locale))


class NamedColumnModel(NumpyArrayTableModel):
    def __init__(
        self,
        column_names: List[str],
        parent: QtCore.QObject = None,
    ):
        array = np.empty((0, len(column_names)), dtype=np.float64)
        super().__init__(array, (0, 1), np.nan, parent)

        self.column_names = column_names

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: QtCore.Qt.ItemDataRole,
    ) -> str:
        if role != QtCore.Qt.DisplayRole:  # pragma: no cover
            return None

        if orientation == QtCore.Qt.Horizontal:
            return self.column_names[section]
        else:
            return str(section)


class ParticleTable(QtWidgets.QWidget):
    unitChanged = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.model = NamedColumnModel(["Response"])

        self.table = QtWidgets.QTableView()
        self.table.setModel(self.model)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)

        self.response = QtWidgets.QComboBox()
        self.response.addItems(["counts", "cps"])
        self.response.currentTextChanged.connect(self.unitChanged)

        layout_unit = QtWidgets.QHBoxLayout()
        layout_unit.addWidget(QtWidgets.QLabel("Response units:"), 1)
        layout_unit.addWidget(self.response, 0)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(layout_unit)
        layout.addWidget(self.table)
        self.setLayout(layout)

    def loadFile(self, file: str) -> dict:
        responses, parameters = read_nanoparticle_file(file, delimiter=",")

        self.response.blockSignals(True)
        self.response.setCurrentText("cps" if parameters["cps"] else "counts")
        self.response.blockSignals(False)

        self.model.beginResetModel()
        self.model.array = responses[:, None]
        self.model.endResetModel()

        return parameters

    def asCounts(
        self, dwelltime: float = None, trim: Tuple[int, int] = (None, None)
    ) -> np.ndarray:
        response = self.model.array[trim[0] : trim[1], 0]
        if self.response.currentText() == "counts":
            return response
        elif dwelltime is not None:
            return response * dwelltime
        else:
            return None


class ResultsTable(QtWidgets.QWidget):
    unitChanged = QtCore.Signal(str, str)

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.model = NamedColumnModel(["Mass (kg)", "Size (m)"])

        self.table = QtWidgets.QTableView()
        self.table.setItemDelegate(DoubleSignificantFiguresDelegate(4))
        self.table.setModel(self.model)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)

        self.response = QtWidgets.QComboBox()
        self.response.addItems(["counts", "cps"])
        # self.response.currentTextChanged.connect(self.unitChanged)

        layout_unit = QtWidgets.QHBoxLayout()
        layout_unit.addWidget(QtWidgets.QLabel("Response units:"), 1)
        layout_unit.addWidget(self.response, 0)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(layout_unit)
        layout.addWidget(self.table)
        self.setLayout(layout)
