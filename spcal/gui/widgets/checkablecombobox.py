from typing import List

from PySide6 import QtCore, QtGui, QtWidgets


class CheckableItemDelegate(QtWidgets.QStyledItemDelegate):
    def editorEvent(
        self,
        event: QtCore.QEvent,
        model: QtCore.QAbstractItemModel,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> bool:
        if event.type() == QtCore.QEvent.Type.MouseButtonRelease:
            role = QtCore.Qt.ItemDataRole.CheckStateRole
            if model.data(index, role) == QtCore.Qt.CheckState.Unchecked.value:
                check = QtCore.Qt.CheckState.Checked.value
            else:
                check = QtCore.Qt.CheckState.Unchecked.value
            return model.setData(index, check, role)

        return super().editorEvent(event, model, option, index)


class CheckableComboBox(QtWidgets.QComboBox):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setModel(QtGui.QStandardItemModel())
        self.setItemDelegate(CheckableItemDelegate(self))

    def addItem(self, text: str) -> None:
        item = QtGui.QStandardItem(text)
        item.setFlags(
            QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled
        )
        item.setData(
            QtCore.Qt.CheckState.Unchecked.value, QtCore.Qt.ItemDataRole.CheckStateRole
        )
        self.model().appendRow(item)

    def addItems(self, texts: List[str]) -> None:
        for text in texts:
            self.addItem(text)

    def checkedItems(self) -> List[str]:
        checked = []
        for row in range(self.model().rowCount()):
            item = self.model().item(row)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                checked.append(item.text())
        return checked

    def setCheckedItems(self, items: List[str]) -> None:
        for row in range(self.model().rowCount()):
            item = self.model().item(row)
            if item.text() in items:
                item.setCheckState(QtCore.Qt.CheckState.Checked)
