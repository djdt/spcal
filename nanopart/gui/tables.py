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
        self.table.setItemDelegate(DoubleSignificantFiguresDelegate(4))
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

    # def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
    #     menu = QtWidgets.QMenu(self)
    #     cut_action.triggered.connect(self._cut)
    #     copy_action = QtWidgets.QAction(
    #         QtGui.QIcon.fromTheme("edit-copy"), "Copy", self
    #     )
    #     copy_action.triggered.connect(self._copy)
    #     paste_action = QtWidgets.QAction(
    #         QtGui.QIcon.fromTheme("edit-paste"), "Paste", self
    #     )
    #     paste_action.triggered.connect(self._paste)

    #     menu.addAction(cut_action)
    #     menu.addAction(copy_action)
    #     menu.addAction(paste_action)

    #     menu.popup(event.globalPos())

    # def keyPressEvent(self, event: QtCore.QEvent) -> None:  # pragma: no cover
    #     if event.key() in [QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return]:
    #         self._advance()
    #     elif event.key() in [QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete]:
    #         self._delete()
    # #     elif event.matches(QtGui.QKeySequence.Copy):
    # #         self._copy()
    # #     elif event.matches(QtGui.QKeySequence.Cut):
    # #         self._cut()
    # #     elif event.matches(QtGui.QKeySequence.Paste):
    # #         self._paste()
    #     else:
    #         super().keyPressEvent(event)

    # def _advance(self) -> None:
    #     index = self.moveCursor(
    #         QtWidgets.QAbstractItemView.MoveDown, QtCore.Qt.NoModifier
    #     )
    #     self.setCurrentIndex(index)

    # def _delete(self) -> None:
    #     for i in self.selectedIndexes():
    #         if i.flags() & QtCore.Qt.ItemIsEditable:
    #             self.model().setData(i, np.nan)

    # def _paste(self) -> None:
    #     text = QtWidgets.QApplication.clipboard().text("plain")[0]
    #     selection = self.selectedIndexes()
    #     start_row = min(selection, key=lambda i: i.row()).row()
    #     start_column = min(selection, key=lambda i: i.column()).column()

    #     for row, row_text in enumerate(text.split("\n")):
    #         for column, text in enumerate(row_text.split("\t")):
    #             if self.model().hasIndex(start_row + row, start_column + column):
    #                 index = self.model().createIndex(
    #                     start_row + row, start_column + column
    #                 )
    #                 if index.isValid() and index.flags() & QtCore.Qt.ItemIsEditable:
    #                     self.model().setData(index, text)


class ResultsTable(QtWidgets.QTableView):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        model = NamedColumnModel([""])
        self.setModel(model)

        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setVisible(False)
        self.verticalHeader().setVisible(False)

        self.setItemDelegate(DoubleSignificantFiguresDelegate(4))
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        menu = QtWidgets.QMenu(self)
        copy_action = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("edit-copy"), "Copy", self
        )
        copy_action.triggered.connect(self._copy)
        menu.addAction(copy_action)
        menu.popup(event.globalPos())

    def keyPressEvent(self, event: QtCore.QEvent) -> None:  # pragma: no cover
        if event.key() in [QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return]:
            self._advance()
        elif event.matches(QtGui.QKeySequence.Copy):
            self._copy()
        else:
            super().keyPressEvent(event)

    def _copy(self) -> None:
        selection = sorted(self.selectedIndexes(), key=lambda i: (i.row(), i.column()))
        data = (
            '<meta http-equiv="content-type" content="text/html; charset=utf-8"/>'
            "<table><tr>"
        )
        text = ""

        prev = None
        for i in selection:
            if prev is not None and prev.row() != i.row():  # New row
                data += "</tr><tr>"
                text += "\n"
            value = i.data()
            data += f"<td>{value}</td>"
            if prev is not None and prev.row() == i.row():
                text += "\t"
            text += f"{value}"
            prev = i
        data += "</tr></table>"

        mime = QtCore.QMimeData()
        mime.setHtml(data)
        mime.setText(text)
        QtWidgets.QApplication.clipboard().setMimeData(mime)
