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

    def mousePressEvent(self, event: QtGui.QMouseEvent):  # pragma, no cover
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
        selected = self.selectedIndexes()

        top = min(index.row() for index in selected)
        bottom = max(index.row() for index in selected)

        left = min(index.column() for index in selected)
        right = max(index.column() for index in selected)

        data = (
            '<meta http-equiv="content-type" content="text/html; charset=utf-8"/>'
            "<table><tr>"
        )

        text = ""
        for row in range(top, bottom + 1):
            for col in range(left, right + 1):
                index = self.model().index(row, col)
                data += "<td>"
                if index.data() is not None:
                    text += str(index.data())
                    data += str(index.data())
                data += "</td>"
                if col != right:
                    text += "\t"

            if row != bottom:
                data += "</tr><tr>"
                text += "\n"

        data += "</tr></table>"

        mime = QtCore.QMimeData()
        mime.setHtml(data)
        mime.setText(text)
        QtWidgets.QApplication.clipboard().setMimeData(mime)

    def _cut(self):  # pragma: no cover, tested in copy / delete
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

        self.model().blockSignals(True)
        for col in range(left, right + 1):
            value = self.model().index(top, col).data(QtCore.Qt.ItemDataRole.EditRole)
            for row in range(top + 1, bottom + 1):
                index = self.model().index(row, col)
                if (
                    index.isValid()
                    and index.flags() & QtCore.Qt.ItemFlag.ItemIsEditable
                ):
                    self.model().setData(
                        index,
                        value,
                        QtCore.Qt.ItemDataRole.EditRole,
                    )

        self.model().blockSignals(False)
        self.model().dataChanged.emit(
            self.model().index(top, left),
            self.model().index(bottom, right),
            QtCore.Qt.ItemDataRole.EditRole,
        )

    def _paste(self):
        clipboard_text, _ = QtWidgets.QApplication.clipboard().text("plain")

        texts = [line.split("\t") for line in clipboard_text.rstrip("\n").split("\n")]

        selected = self.selectedIndexes()

        top = min(index.row() for index in selected)
        bottom = max(index.row() for index in selected)
        left = min(index.column() for index in selected)
        right = max(index.column() for index in selected)

        # The selection is larger, expand
        if len(texts) > bottom - top:
            bottom = top + len(texts) - 1
        if any(len(line) for line in texts) > right - left:
            right = left + max(len(line) for line in texts) - 1

        self.model().blockSignals(True)
        for row in range(top, bottom + 1):
            for col in range(left, right + 1):
                text_row = (row - top) % len(texts)
                text = texts[text_row][(col - left) % len(texts[text_row])]
                index = self.model().index(row, col)
                if (
                    index.isValid()
                    and index.flags() & QtCore.Qt.ItemFlag.ItemIsEditable
                ):
                    self.model().setData(index, text, QtCore.Qt.ItemDataRole.EditRole)

        self.model().blockSignals(False)
        self.model().dataChanged.emit(
            self.model().index(top, left),
            self.model().index(bottom, right),
            QtCore.Qt.ItemDataRole.EditRole,
        )
