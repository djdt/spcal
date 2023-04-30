from typing import List, Tuple

from PySide6 import QtCore, QtWidgets

from spcal.gui.util import create_action
from spcal.gui.widgets import UnitsWidget
from spcal.siunits import mass_units, signal_units, size_units


class FilterRow(QtWidgets.QWidget):
    closeRequested = QtCore.Signal(QtWidgets.QWidget)

    def __init__(self, elements: List[str], parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        self.boolean = QtWidgets.QComboBox()
        self.boolean.addItems(["And", "Or"])

        self.elements = QtWidgets.QComboBox()
        self.elements.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToContentsOnFirstShow
        )
        self.elements.addItems(elements)

        self.unit = QtWidgets.QComboBox()
        self.unit.addItems(["Intensity", "Mass", "Size"])
        self.unit.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContentsOnFirstShow)
        self.unit.currentTextChanged.connect(self.changeUnits)

        self.operation = QtWidgets.QComboBox()
        self.operation.addItems([">", "<", ">=", "<=", "=="])

        self.value = UnitsWidget(units=signal_units)
        self.value.combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)

        self.action_close = create_action(
            "list-remove", "Remove", "Remove the filter.", self.close
        )

        self.button_close = QtWidgets.QToolButton()
        self.button_close.setAutoRaise(True)
        self.button_close.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.button_close.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        self.button_close.setDefaultAction(self.action_close)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.boolean, 0)
        layout.addWidget(self.elements, 0)
        layout.addWidget(self.unit, 0)
        layout.addWidget(self.operation, 0)
        layout.addWidget(self.value, 1)
        layout.addWidget(self.button_close, 0, QtCore.Qt.AlignRight)
        self.setLayout(layout)

    def asTuple(self) -> Tuple[str, str, str, str, float | None]:
        return (
            self.boolean.currentText(),
            self.elements.currentText(),
            self.unit.currentText(),
            self.operation.currentText(),
            self.value.baseValue(),
        )

    def close(self) -> None:
        self.closeRequested.emit(self)
        super().close()

    def changeUnits(self, unit: str) -> None:
        if unit == "Intensity":
            units = signal_units
        elif unit == "Mass":
            units = mass_units
        elif unit == "Size":
            units = size_units
        else:
            raise ValueError("changeUnits: unknown unit")

        self.value.setUnits(units)


class FilterRows(QtWidgets.QScrollArea):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        self.rows: List[FilterRow] = []

        widget = QtWidgets.QWidget()
        self.setWidget(widget)
        self.setWidgetResizable(True)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.setSpacing(0)
        widget.setLayout(self.layout)

    def addRow(self, row: FilterRow) -> None:
        row.closeRequested.connect(self.removeRow)
        if len(self.rows) == 0:
            row.boolean.setEnabled(False)
        self.rows.append(row)
        self.layout.addWidget(row)

    def removeRow(self, row: FilterRow) -> None:
        self.rows.remove(row)
        self.layout.removeWidget(row)

    def asList(self) -> List[Tuple[str, str, str, str, float]]:
        filters = []
        for row in self.rows:
            filter = row.asTuple()
            if filter[-1] is not None:
                filters.append(filter)
        return filters  # type: ignore


class FilterDialog(QtWidgets.QDialog):
    filtersChanged = QtCore.Signal(list)

    def __init__(
        self,
        elements: List[str],
        filters: list,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Compositional Filters")
        self.setMinimumSize(600, 480)

        self.elements = elements
        self.rows = FilterRows()

        for filter in filters:
            self.addFilter(filter)

        self.action_add = create_action(
            "list-add", "Add Filter", "Add a new filter.", lambda: self.addFilter(None)
        )

        self.button_add = QtWidgets.QToolButton()
        self.button_add.setAutoRaise(True)
        self.button_add.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.button_add.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.button_add.setDefaultAction(self.action_add)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Close
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.button_add, 0)
        layout.addWidget(self.rows, 1)
        layout.addWidget(self.button_box, 0)
        self.setLayout(layout)

    def addFilter(self, filter: Tuple[str, str, str, str, float] | None = None) -> None:
        row = FilterRow(self.elements, parent=self)
        if filter is not None:
            boolean, element, unit, operation, value = filter
            row.boolean.setCurrentText(boolean)
            row.elements.setCurrentText(element)
            row.unit.setCurrentText(unit)
            row.operation.setCurrentText(operation)
            row.value.setBaseValue(value)
            row.value.setBestUnit()

        self.rows.addRow(row)

    def accept(self) -> None:
        self.filtersChanged.emit(self.rows.asList())
        super().accept()


class FilterItem(QtWidgets.QWidget):
    closeRequested = QtCore.Signal(QtWidgets.QWidget)

    def __init__(
        self,
        elements: List[str],
        filter: Tuple[str, str, str, float] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        self.elements = QtWidgets.QComboBox()
        self.elements.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToContentsOnFirstShow
        )
        self.elements.addItems(elements)

        self.unit = QtWidgets.QComboBox()
        self.unit.addItems(["Intensity", "Mass", "Size"])
        self.unit.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContentsOnFirstShow)
        self.unit.currentTextChanged.connect(self.changeUnits)

        self.operation = QtWidgets.QComboBox()
        self.operation.addItems([">", "<", ">=", "<=", "=="])

        self.value = UnitsWidget(units=signal_units)
        self.value.combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)

        if filter is not None:
            self.elements.setCurrentText(filter[0])
            self.unit.setCurrentText(filter[1])
            self.operation.setCurrentText(filter[2])
            self.value.setBaseValue(filter[3])
            self.value.setBestUnit()

        self.action_close = create_action(
            "list-remove", "Remove", "Remove the filter.", self.close
        )

        self.button_close = QtWidgets.QToolButton()
        self.button_close.setAutoRaise(True)
        self.button_close.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.button_close.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        self.button_close.setDefaultAction(self.action_close)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.elements, 0)
        layout.addWidget(self.unit, 0)
        layout.addWidget(self.operation, 0)
        layout.addWidget(self.value, 1)
        layout.addWidget(self.button_close, 0, QtCore.Qt.AlignRight)
        self.setLayout(layout)

    def asTuple(self) -> Tuple[str, str, str, str, float | None]:
        return (
            self.boolean.currentText(),
            self.elements.currentText(),
            self.unit.currentText(),
            self.operation.currentText(),
            self.value.baseValue(),
        )

    def close(self) -> None:
        self.closeRequested.emit(self)
        super().close()

    def changeUnits(self, unit: str) -> None:
        if unit == "Intensity":
            units = signal_units
        elif unit == "Mass":
            units = mass_units
        elif unit == "Size":
            units = size_units
        else:
            raise ValueError("changeUnits: unknown unit")

        self.value.setUnits(units)


class _FilterDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Particle Composition Filters")

        self.list = QtWidgets.QListWidget()
        self.list.setDragEnabled(True)
        self.list.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Close
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.list, 1)
        layout.addWidget(self.button_box, 0)
        self.setLayout(layout)

    def addFilter(self):
        item = QtWidgets.QListWidgetItem()
        widget = FilterItem(["a", "b", "c"])
        self.list.insertItem(self.list.count(), item)
        self.list.setItemWidget(item, widget)
        item.setSizeHint(widget.sizeHint())

    def addBooleanOr(self):
        item = QtWidgets.QListWidgetItem()
        widget = QtWidgets.QFrame()
        widget.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.list.insertItem(self.list.count(), item)
        self.list.setItemWidget(item, widget)
        item.setSizeHint(widget.sizeHint())


if __name__ == "__main__":
    app = QtWidgets.QApplication()

    dlg = _FilterDialog()
    dlg.addFilter()
    dlg.addFilter()
    dlg.addBooleanOr()
    dlg.addFilter()
    dlg.show()
    app.exec()
