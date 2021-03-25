from PySide2 import QtCore, QtGui, QtWidgets
import numpy as np

from nanopart.gui.util import NumpyArrayTableModel

from typing import List


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
        super().__init__(array, (0, 1), 0.0, parent)

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


class ParticleTable(QtWidgets.QTableView):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        model = NamedColumnModel(["Response"])
        self.setModel(model)

        self.setItemDelegate(DoubleSignificantFiguresDelegate(4))
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)

    # def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
    #     menu = QtWidgets.QMenu(self)
    # #     cut_action.triggered.connect(self._cut)
    # #     copy_action = QtWidgets.QAction(
    # #         QtGui.QIcon.fromTheme("edit-copy"), "Copy", self
    # #     )
    # #     copy_action.triggered.connect(self._copy)
    #     paste_action = QtWidgets.QAction(
    #         QtGui.QIcon.fromTheme("edit-paste"), "Paste", self
    #     )
    #     paste_action.triggered.connect(self._paste)

    # #     menu.addAction(cut_action)
    # #     menu.addAction(copy_action)
    #     menu.addAction(paste_action)

    #     menu.popup(event.globalPos())

    def keyPressEvent(self, event: QtCore.QEvent) -> None:  # pragma: no cover
        if event.key() in [QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return]:
            self._advance()
        elif event.key() in [QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete]:
            self._delete()
    # #     elif event.matches(QtGui.QKeySequence.Copy):
    # #         self._copy()
    # #     elif event.matches(QtGui.QKeySequence.Cut):
    # #         self._cut()
        # elif event.matches(QtGui.QKeySequence.Paste):
        #     self._paste()
        else:
            super().keyPressEvent(event)

    def _advance(self) -> None:
        index = self.moveCursor(
            QtWidgets.QAbstractItemView.MoveDown, QtCore.Qt.NoModifier
        )
        self.setCurrentIndex(index)

    def _delete(self) -> None:
        self.model().blockSignals(True)
        indicies = self.selectedIndexes()
        for i in indicies[1:]:
            self.model().setData(i, np.nan)
        self.model().blockSignals(False)
        self.model().setData(indicies[0], np.nan)
        self.clearSelection()


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
        event.accept()
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
