from typing import Dict, List

from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.util import create_action


class EnableTextDialog(QtWidgets.QDialog):
    enabledSelected = QtCore.Signal(dict)

    def __init__(
        self, items: List[QtGui.QStandardItem], parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent)
        self.setWindowTitle("Set Enabled Items")
        self.setMinimumWidth(300)

        self.texts = QtWidgets.QListWidget()
        for item in items:
            text = QtWidgets.QListWidgetItem(item.text())
            text.setCheckState(
                QtCore.Qt.CheckState.Checked
                if item.isEnabled()
                else QtCore.Qt.CheckState.Unchecked
            )
            self.texts.addItem(text)
        self.texts.itemChanged.connect(self.completeChanged)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.texts, 1)
        layout.addWidget(self.button_box, 0)
        self.setLayout(layout)

    def isComplete(self) -> bool:
        for i in range(self.texts.count()):
            if self.texts.item(i).checkState() == QtCore.Qt.CheckState.Checked:
                return True
        return False

    def completeChanged(self) -> None:
        button = self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        button.setEnabled(self.isComplete())

    def accept(self) -> None:
        enabled = {}
        for i in range(self.texts.count()):
            enabled[self.texts.item(i).text()] = (
                self.texts.item(i).checkState() == QtCore.Qt.CheckState.Checked
            )
        self.enabledSelected.emit(enabled)
        super().accept()


class EditableComboBox(QtWidgets.QComboBox):
    textsEdited = QtCore.Signal(dict)
    enabledTextsChanged = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setEditable(True)
        self.setDuplicatesEnabled(False)
        self.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.InsertAtCurrent)
        self.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)

        self.lineEdit().returnPressed.connect(self.editingFinished)
        self.currentIndexChanged.connect(self.saveCurrentName)

        self.action_enable_names = create_action(
            "font-enable",
            "Set Enabled Items",
            "Open a dialog to enable or disable items.",
            self.openEnableDialog,
        )

        self.previous_name = ""

    def saveCurrentName(self, index: int) -> None:
        self.previous_name = self.itemText(index)

    def editingFinished(self):
        new_name = self.currentText()
        if new_name != self.previous_name:
            self.textsEdited.emit({self.previous_name: new_name})
            self.previous_name = new_name

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        menu = QtWidgets.QMenu(self)
        menu.addAction(self.action_enable_names)
        menu.popup(event.globalPos())
        event.accept()

    def openEnableDialog(self) -> EnableTextDialog:
        items = [self.model().item(i) for i in range(self.count())]
        dlg = EnableTextDialog(items, self)
        dlg.enabledSelected.connect(self.setEnabled)
        dlg.open()
        return dlg

    def setEnabled(self, enabled: Dict[str, bool]) -> None:
        for text, state in enabled.items():
            i = self.findText(text)
            if i != -1:
                item = self.model().item(i)
                item.setEnabled(state)
        if not self.model().item(self.currentIndex()).isEnabled():
            for i in range(self.count()):
                if self.model().item(i).isEnabled():
                    self.setCurrentIndex(i)
                    break
        self.enabledTextsChanged.emit()
