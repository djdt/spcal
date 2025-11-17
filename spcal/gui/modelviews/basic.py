from PySide6 import QtCore, QtGui, QtWidgets


class BasicTable(QtWidgets.QTableWidget):
    def basicTableMenu(self) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu(self)
        cut_action = QtGui.QAction(QtGui.QIcon.fromTheme("edit-cut"), "Cut", self)
        cut_action.triggered.connect(self._cut)
        copy_action = QtGui.QAction(QtGui.QIcon.fromTheme("edit-copy"), "Copy", self)
        copy_action.triggered.connect(self._copy)
        paste_action = QtGui.QAction(QtGui.QIcon.fromTheme("edit-paste"), "Paste", self)
        paste_action.triggered.connect(self._paste)

        if self.editTriggers() == QtWidgets.QTableView.EditTrigger.NoEditTriggers:
            any_editable = False
        else:
            any_editable = any(
                item.flags() & QtCore.Qt.ItemFlag.ItemIsEditable
                for item in self.selectedItems()
            )

        if any_editable:
            menu.addAction(cut_action)
        menu.addAction(copy_action)
        if any_editable:
            menu.addAction(paste_action)
        return menu

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        event.accept()
        menu = self.basicTableMenu()
        menu.popup(event.globalPos())

    def keyPressEvent(self, event: QtGui.QKeyEvent):  # pragma: no cover
        if event.key() in [QtCore.Qt.Key.Key_Enter, QtCore.Qt.Key.Key_Return]:
            self._advance()
        elif event.key() in [QtCore.Qt.Key.Key_Backspace, QtCore.Qt.Key.Key_Delete]:
            self._delete()
        elif event.matches(QtGui.QKeySequence.StandardKey.Copy):
            self._copy()
        elif event.matches(QtGui.QKeySequence.StandardKey.Cut):
            self._cut()
        elif event.matches(QtGui.QKeySequence.StandardKey.Paste):
            self._paste()
        else:
            super().keyPressEvent(event)

    def _advance(self):
        row = self.currentRow()
        if row + 1 < self.rowCount():
            self.setCurrentCell(row + 1, self.currentColumn())

    def _copy(self):
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
            value = "" if i.data() is None else i.data()
            data += f"<td>{value}</td>"
            if i.column() != 0:
                text += "\t"
            text += f"{value}"
            prev = i
        data += "</tr></table>"

        mime = QtCore.QMimeData()
        mime.setHtml(data)
        mime.setText(text)
        QtWidgets.QApplication.clipboard().setMimeData(mime)

    def _cut(self):
        self._copy()
        self._delete()

    def _delete(self):
        for i in self.selectedItems():
            if i.flags() & QtCore.Qt.ItemFlag.ItemIsEditable:
                i.setText("")

    def _paste(self):
        text = QtWidgets.QApplication.clipboard().text("plain")
        selection = self.selectedIndexes()
        start_row = min(selection, key=lambda i: i.row()).row()
        start_column = min(selection, key=lambda i: i.column()).column()

        for row, row_text in enumerate(text[0].split("\n")):
            for column, text in enumerate(row_text.split("\t")):
                item = self.item(start_row + row, start_column + column)
                if (
                    item is not None
                    and QtCore.Qt.ItemFlag.ItemIsEditable & item.flags()
                ):
                    item.setText(text)
