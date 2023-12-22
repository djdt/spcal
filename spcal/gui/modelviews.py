from PySide6 import QtCore, QtGui, QtWidgets

# class BasicTableView(QtWidgets.QTableView):
#     def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
#         event.accept()
#         menu = QtWidgets.QMenu(self)
#         cut_action = QtGui.QAction(QtGui.QIcon.fromTheme("edit-cut"), "Cut", self)
#         cut_action.triggered.connect(self._cut)
#         copy_action = QtGui.QAction(
#             QtGui.QIcon.fromTheme("edit-copy"), "Copy", self
#         )
#         copy_action.triggered.connect(self._copy)
#         paste_action = QtGui.QAction(
#             QtGui.QIcon.fromTheme("edit-paste"), "Paste", self
#         )
#         paste_action.triggered.connect(self._paste)

#         menu.addAction(cut_action)
#         menu.addAction(copy_action)
#         menu.addAction(paste_action)

#         menu.popup(event.globalPos())

#     def keyPressEvent(self, event: QtCore.QEvent) -> None:  # pragma: no cover
#         if event.key() in [QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return]:
#             self._advance()
#         elif event.key() in [QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete]:
#             self._delete()
#         elif event.matches(QtGui.QKeySequence.Copy):
#             self._copy()
#         elif event.matches(QtGui.QKeySequence.Cut):
#             self._cut()
#         elif event.matches(QtGui.QKeySequence.Paste):
#             self._paste()
#         else:
#             super().keyPressEvent(event)

#     def _advance(self) -> None:
#         index = self.moveCursor(
#             QtWidgets.QAbstractItemView.MoveDown, QtCore.Qt.NoModifier
#         )
#         self.setCurrentIndex(index)

#     def _copy(self) -> None:
#         selection = sorted(self.selectedIndexes(), key=lambda i: (i.row(), i.column()))
#         data = (
#             '<meta http-equiv="content-type" content="text/html; charset=utf-8"/>'
#             "<table><tr>"
#         )
#         text = ""

#         prev = None
#         for i in selection:
#             if prev is not None and prev.row() != i.row():  # New row
#                 data += "</tr><tr>"
#                 text += "\n"
#             value = i.data()
#             data += f"<td>{value}</td>"
#             if prev is not None and prev.row() == i.row():
#                 text += "\t"
#             text += f"{value}"
#             prev = i
#         data += "</tr></table>"

#         mime = QtCore.QMimeData()
#         mime.setHtml(data)
#         mime.setText(text)
#         QtWidgets.QApplication.clipboard().setMimeData(mime)

#     def _cut(self) -> None:
#         self._copy()
#         self._delete()

#     def _delete(self) -> None:
#         for i in self.selectedIndexes():
#             if i.flags() & QtCore.Qt.ItemIsEditable:
#                 self.model().setData(i, "")

#     def _paste(self) -> None:
#         text = QtWidgets.QApplication.clipboard().text("plain")[0]
#         selection = self.selectedIndexes()
#         start_row = min(selection, key=lambda i: i.row()).row()
#         start_column = min(selection, key=lambda i: i.column()).column()

#         for row, row_text in enumerate(text.split("\n")):
#             for column, text in enumerate(row_text.split("\t")):
#                 if self.model().hasIndex(start_row + row, start_column + column):
#                     index = self.model().createIndex(
#                         start_row + row, start_column + column
#                     )
#                     if index.isValid() and index.flags() & QtCore.Qt.ItemIsEditable:
#                         self.model().setData(index, text)


class BasicTable(QtWidgets.QTableWidget):
    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        event.accept()
        menu = QtWidgets.QMenu(self)
        cut_action = QtGui.QAction(QtGui.QIcon.fromTheme("edit-cut"), "Cut", self)
        cut_action.triggered.connect(self._cut)
        copy_action = QtGui.QAction(QtGui.QIcon.fromTheme("edit-copy"), "Copy", self)
        copy_action.triggered.connect(self._copy)
        paste_action = QtGui.QAction(QtGui.QIcon.fromTheme("edit-paste"), "Paste", self)
        paste_action.triggered.connect(self._paste)

        any_editable = any(
            item.flags() & QtCore.Qt.ItemFlag.ItemIsEditable
            for item in self.selectedItems()
        )

        if any_editable:
            menu.addAction(cut_action)
        menu.addAction(copy_action)
        if any_editable:
            menu.addAction(paste_action)

        menu.popup(event.globalPos())

    def keyPressEvent(self, event: QtCore.QEvent) -> None:  # pragma: no cover
        if event.key() in [QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return]:
            self._advance()
        elif event.key() in [QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete]:
            self._delete()
        elif event.matches(QtGui.QKeySequence.Copy):
            self._copy()
        elif event.matches(QtGui.QKeySequence.Cut):
            self._cut()
        elif event.matches(QtGui.QKeySequence.Paste):
            self._paste()
        else:
            super().keyPressEvent(event)

    def _advance(self) -> None:
        row = self.currentRow()
        if row + 1 < self.rowCount():
            self.setCurrentCell(row + 1, self.currentColumn())

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

    def _cut(self) -> None:
        self._copy()
        self._delete()

    def _delete(self) -> None:
        for i in self.selectedItems():
            if i.flags() & QtCore.Qt.ItemFlag.ItemIsEditable:
                i.setText("")

    def _paste(self) -> None:
        text = QtWidgets.QApplication.clipboard().text("plain")
        selection = self.selectedIndexes()
        start_row = min(selection, key=lambda i: i.row()).row()
        start_column = min(selection, key=lambda i: i.column()).column()

        for row, row_text in enumerate(text[0].split("\n")):
            for column, text in enumerate(row_text.split("\t")):
                item = self.item(start_row + row, start_column + column)
                if item is not None and QtCore.Qt.ItemIsEditable & item.flags():
                    item.setText(text)

    # def columnText(self, column: int) -> list[str]:
    #     return [self.item(row, column).text() for row in range(0, self.rowCount())]

    # def rowText(self, row: int) -> list[str]:
    #     return [
    #         self.item(row, column).text() for column in range(0, self.columnCount())
    #     ]

    # def setColumnText(self, column: int, text: list[str] | None = None) -> None:
    #     if text is not None:
    #         assert len(text) <= self.rowCount()
    #     for row in range(0, self.rowCount()):
    #         self.item(row, column).setText(text[row] if text is not None else "")

    # def setRowText(self, row: int, text: list[str] | None = None) -> None:
    #     if text is not None:
    #         assert len(text) <= self.columnCount()
    #     for column in range(0, self.columnCount()):
    #         self.item(row, column).setText(text[column] if text is not None else "")
