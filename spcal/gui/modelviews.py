from PySide6 import QtCore, QtGui, QtWidgets


class CheckableHeaderView(QtWidgets.QHeaderView):
    checkStateChanged = QtCore.Signal(int, QtCore.Qt.CheckState)

    def __init__(
        self,
        orientation: QtCore.Qt.Orientation,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(orientation, parent)

        self._checked: dict[int, QtCore.Qt.CheckState] = {}

    def checkState(self, logicalIndex: int) -> QtCore.Qt.CheckState:
        assert logicalIndex >= 0 and logicalIndex < self.count()
        return self._checked.get(logicalIndex, QtCore.Qt.CheckState.Unchecked)

    def setCheckState(self, logicalIndex: int, state: QtCore.Qt.CheckState) -> None:
        assert logicalIndex >= 0 and logicalIndex < self.count()
        if self.checkState(logicalIndex) != state:
            self._checked[logicalIndex] = state
            self.checkStateChanged.emit(logicalIndex, state)

    def sectionSizeFromContents(self, logicalIndex: int) -> QtCore.QSize:
        size = super().sectionSizeFromContents(logicalIndex)
        option = QtWidgets.QStyleOptionButton()
        cb_size = self.style().sizeFromContents(
            QtWidgets.QStyle.ContentsType.CT_CheckBox, option, size
        )
        size.setWidth(size.width() + cb_size.width())
        return size

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.cursor().shape() in [  # resizing, ignore custom code
            QtCore.Qt.CursorShape.SplitHCursor,
            QtCore.Qt.CursorShape.SplitVCursor,
        ]:
            return super().mousePressEvent(event)

        logicalIndex = self.logicalIndexAt(event.position().toPoint())
        if logicalIndex >= 0 and logicalIndex < self.count():
            state = self._checked.get(logicalIndex, QtCore.Qt.CheckState.Unchecked)

            if QtCore.Qt.KeyboardModifier.ShiftModifier & event.modifiers():
                self.setCheckState(logicalIndex, QtCore.Qt.CheckState.Checked)

                for idx in range(0, self.count()):
                    if idx == logicalIndex:
                        continue
                    self.setCheckState(idx, QtCore.Qt.CheckState.Unchecked)

            elif state == QtCore.Qt.CheckState.Checked:
                self.setCheckState(logicalIndex, QtCore.Qt.CheckState.Unchecked)
            else:
                self.setCheckState(logicalIndex, QtCore.Qt.CheckState.Checked)

            self.viewport().update()

    def paintSection(
        self, painter: QtGui.QPainter, rect: QtCore.QRect, logicalIndex: int
    ) -> None:
        painter.save()
        super().paintSection(painter, rect, logicalIndex)
        painter.restore()

        size = super().sectionSizeFromContents(logicalIndex)
        option = QtWidgets.QStyleOptionButton()
        cb_size = self.style().sizeFromContents(
            QtWidgets.QStyle.ContentsType.CT_CheckBox, option, size
        )

        option.rect = QtCore.QRect(  # type: ignore , works
            rect.left(),
            rect.center().y() - cb_size.height() // 2,
            cb_size.width(),
            cb_size.height(),
        )
        option.state = (  # type: ignore , works
            QtWidgets.QStyle.StateFlag.State_Enabled
            | QtWidgets.QStyle.StateFlag.State_Active
        )

        state = self.checkState(logicalIndex)
        if state == QtCore.Qt.CheckState.Checked:
            option.state |= QtWidgets.QStyle.StateFlag.State_On  # type: ignore , works
        elif state == QtCore.Qt.CheckState.Unchecked:
            option.state |= QtWidgets.QStyle.StateFlag.State_Off  # type: ignore , works
        else:
            option.state |= QtWidgets.QStyle.StateFlag.State_NoChange  # type: ignore , works

        self.style().drawControl(
            QtWidgets.QStyle.ControlElement.CE_CheckBox, option, painter
        )


class ComboHeaderView(QtWidgets.QHeaderView):
    """
    Params:
        selection_items: dict of section numbers to combobox items
        orientation: header type, horizontal or vertical
    """

    sectionChanged = QtCore.Signal(int)

    def __init__(
        self,
        section_items: dict[int, list[str]],
        orientation: QtCore.Qt.Orientation = QtCore.Qt.Orientation.Horizontal,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(orientation, parent)

        self.section_items = section_items

    def showComboBox(self, section: int) -> None:
        items = self.section_items.get(section, [])
        widget = QtWidgets.QComboBox(self)
        widget.addItems(items)
        widget.setCurrentText(self.model().headerData(section, self.orientation()))

        pos = self.sectionViewportPosition(section)
        size = self.sectionSizeFromContents(section)

        widget.setGeometry(QtCore.QRect(pos, 0, size.width(), size.height()))
        widget.currentTextChanged.connect(
            lambda value: self.model().setHeaderData(
                section, self.orientation(), value, QtCore.Qt.ItemDataRole.EditRole
            )
        )
        widget.currentTextChanged.connect(lambda: self.sectionChanged.emit(section))
        widget.currentTextChanged.connect(widget.deleteLater)
        widget.showPopup()

    def sectionSizeFromContents(self, logicalIndex: int) -> QtCore.QSize:
        size = super().sectionSizeFromContents(logicalIndex)
        option = QtWidgets.QStyleOptionComboBox()
        option.initFrom(self)
        return self.style().sizeFromContents(
            QtWidgets.QStyle.ContentsType.CT_ComboBox, option, size
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        logicalIndex = self.logicalIndexAt(event.position().toPoint())
        if logicalIndex in self.section_items:
            self.showComboBox(logicalIndex)
        else:
            super().mousePressEvent(event)

    def paintSection(
        self, painter: QtGui.QPainter, rect: QtCore.QRect, logicalIndex: int
    ) -> None:
        option = QtWidgets.QStyleOptionComboBox()
        option.initFrom(self)
        option.rect = rect  # type: ignore
        option.currentText = str(  # type: ignore
            self.model().headerData(logicalIndex, self.orientation())
        )
        if logicalIndex not in self.section_items:
            option.subControls = (  # type: ignore
                option.subControls & ~QtWidgets.QStyle.SubControl.SC_ComboBoxArrow  # type: ignore
            )

        if self.hasFocus():
            option.state = QtWidgets.QStyle.StateFlag.State_Selected  # type: ignore

        self.style().drawComplexControl(
            QtWidgets.QStyle.ComplexControl.CC_ComboBox, option, painter
        )
        self.style().drawControl(
            QtWidgets.QStyle.ControlElement.CE_ComboBoxLabel, option, painter
        )


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
    def basicTableMenu(self) -> QtWidgets.QMenu:
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
        return menu

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        event.accept()
        menu = self.basicTableMenu()
        menu.popup(event.globalPos())

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # pragma: no cover
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
                if (
                    item is not None
                    and QtCore.Qt.ItemFlag.ItemIsEditable & item.flags()
                ):
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
