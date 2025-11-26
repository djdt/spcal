from PySide6 import QtCore, QtGui, QtWidgets


class BasicTableView(QtWidgets.QTableView):
    def basicTableMenu(self) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu(self)
        cut_action = QtGui.QAction(QtGui.QIcon.fromTheme("edit-cut"), "Cut", self)
        cut_action.setShortcut(QtGui.QKeySequence.StandardKey.Cut)
        cut_action.triggered.connect(self._cut)
        copy_action = QtGui.QAction(QtGui.QIcon.fromTheme("edit-copy"), "Copy", self)
        copy_action.triggered.connect(self._copy)
        copy_action.setShortcut(QtGui.QKeySequence.StandardKey.Copy)
        paste_action = QtGui.QAction(QtGui.QIcon.fromTheme("edit-paste"), "Paste", self)
        paste_action.triggered.connect(self._paste)
        paste_action.setShortcut(QtGui.QKeySequence.StandardKey.Paste)
        filldown_action = QtGui.QAction(
            QtGui.QIcon.fromTheme("arrow-down-double"), "Fill Down", self
        )
        filldown_action.setShortcut(QtCore.Qt.Key.Key_F9)
        filldown_action.triggered.connect(self._filldown)

        if self.editTriggers() == QtWidgets.QTableView.EditTrigger.NoEditTriggers:
            any_editable = False
        else:
            any_editable = any(
                index.flags() & QtCore.Qt.ItemFlag.ItemIsEditable
                for index in self.selectionModel().selectedIndexes()
            )

        if any_editable:
            menu.addAction(filldown_action)
            menu.addSeparator()
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
        elif event.key() == QtCore.Qt.Key.Key_F9:
            self._filldown()
        elif event.matches(QtGui.QKeySequence.StandardKey.Copy):
            self._copy()
        elif event.matches(QtGui.QKeySequence.StandardKey.Cut):
            self._cut()
        elif event.matches(QtGui.QKeySequence.StandardKey.Paste):
            self._paste()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            event.ignore()
        else:
            super().mousePressEvent(event)

    def _advance(self):
        index = self.selectionModel().currentIndex()
        if index.row() + 1 < self.model().rowCount():
            new_index = self.model().index(index.row() + 1, index.column())
            self.selectionModel().setCurrentIndex(
                new_index, QtCore.QItemSelectionModel.SelectionFlag.Current
            )

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
        for index in self.selectedIndexes():
            if index.flags() & QtCore.Qt.ItemFlag.ItemIsEditable:
                self.model().setData(index, None, QtCore.Qt.ItemDataRole.EditRole)

    def _filldown(self):
        selected = self.selectedIndexes()
        top = min(index.row() for index in selected)
        bottom = max(index.row() for index in selected)
        if top == bottom:  # single, filldown as far as possible
            bottom = self.model().rowCount() - 1

        left = min(index.column() for index in selected)
        right = max(index.column() for index in selected)
        for col in range(left, right + 1):
            value = self.model().index(top, col).data(QtCore.Qt.ItemDataRole.EditRole)
            for row in range(top + 1, bottom + 1):
                index = self.model().index(row, col)
                if index.flags() & QtCore.Qt.ItemFlag.ItemIsEditable:
                    self.model().setData(
                        index,
                        value,
                        QtCore.Qt.ItemDataRole.EditRole,
                    )

    def _paste(self):
        text = QtWidgets.QApplication.clipboard().text("plain")
        selection = self.selectedIndexes()
        start_row = min(selection, key=lambda i: i.row()).row()
        start_column = min(selection, key=lambda i: i.column()).column()

        for row, row_text in enumerate(text[0].split("\n")):
            for column, text in enumerate(row_text.split("\t")):
                index = self.model().index(start_row + row, start_column + column)
                if (
                    index.isValid()
                    and index.flags() & QtCore.Qt.ItemFlag.ItemIsEditable
                ):
                    self.model().setData(index, text, QtCore.Qt.ItemDataRole.EditRole)


# class BasicTable(QtWidgets.QTableWidget):
#     def basicTableMenu(self) -> QtWidgets.QMenu:
#         menu = QtWidgets.QMenu(self)
#         cut_action = QtGui.QAction(QtGui.QIcon.fromTheme("edit-cut"), "Cut", self)
#         cut_action.triggered.connect(self._cut)
#         copy_action = QtGui.QAction(QtGui.QIcon.fromTheme("edit-copy"), "Copy", self)
#         copy_action.triggered.connect(self._copy)
#         paste_action = QtGui.QAction(QtGui.QIcon.fromTheme("edit-paste"), "Paste", self)
#         paste_action.triggered.connect(self._paste)
#
#         if self.editTriggers() == QtWidgets.QTableView.EditTrigger.NoEditTriggers:
#             any_editable = False
#         else:
#             any_editable = any(
#                 item.flags() & QtCore.Qt.ItemFlag.ItemIsEditable
#                 for item in self.selectedItems()
#             )
#
#         if any_editable:
#             menu.addAction(cut_action)
#         menu.addAction(copy_action)
#         if any_editable:
#             menu.addAction(paste_action)
#         return menu
#
#     def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
#         event.accept()
#         menu = self.basicTableMenu()
#         menu.popup(event.globalPos())
#
#     def keyPressEvent(self, event: QtGui.QKeyEvent):  # pragma: no cover
#         if event.key() in [QtCore.Qt.Key.Key_Enter, QtCore.Qt.Key.Key_Return]:
#             self._advance()
#         elif event.key() in [QtCore.Qt.Key.Key_Backspace, QtCore.Qt.Key.Key_Delete]:
#             self._delete()
#         elif event.matches(QtGui.QKeySequence.StandardKey.Copy):
#             self._copy()
#         elif event.matches(QtGui.QKeySequence.StandardKey.Cut):
#             self._cut()
#         elif event.matches(QtGui.QKeySequence.StandardKey.Paste):
#             self._paste()
#         else:
#             super().keyPressEvent(event)
#
#     def _advance(self):
#         row = self.currentRow()
#         if row + 1 < self.rowCount():
#             self.setCurrentCell(row + 1, self.currentColumn())
#
#     def _copy(self):
#         selection = sorted(self.selectedIndexes(), key=lambda i: (i.row(), i.column()))
#         data = (
#             '<meta http-equiv="content-type" content="text/html; charset=utf-8"/>'
#             "<table><tr>"
#         )
#         text = ""
#
#         prev = None
#         for i in selection:
#             if prev is not None and prev.row() != i.row():  # New row
#                 data += "</tr><tr>"
#                 text += "\n"
#             value = "" if i.data() is None else i.data()
#             data += f"<td>{value}</td>"
#             if i.column() != 0:
#                 text += "\t"
#             text += f"{value}"
#             prev = i
#         data += "</tr></table>"
#
#         mime = QtCore.QMimeData()
#         mime.setHtml(data)
#         mime.setText(text)
#         QtWidgets.QApplication.clipboard().setMimeData(mime)
#
#     def _cut(self):
#         self._copy()
#         self._delete()
#
#     def _delete(self):
#         for i in self.selectedItems():
#             if i.flags() & QtCore.Qt.ItemFlag.ItemIsEditable:
#                 i.setText("")
#
#     def _paste(self):
#         text = QtWidgets.QApplication.clipboard().text("plain")
#         selection = self.selectedIndexes()
#         start_row = min(selection, key=lambda i: i.row()).row()
#         start_column = min(selection, key=lambda i: i.column()).column()
#
#         for row, row_text in enumerate(text[0].split("\n")):
#             for column, text in enumerate(row_text.split("\t")):
#                 item = self.item(start_row + row, start_column + column)
#                 if (
#                     item is not None
#                     and QtCore.Qt.ItemFlag.ItemIsEditable & item.flags()
#                 ):
#                     item.setText(text)
