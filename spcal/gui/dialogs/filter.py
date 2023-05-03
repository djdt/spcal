from typing import Dict, List

import numpy as np
from PySide6 import QtCore, QtWidgets

from spcal.gui.util import create_action
from spcal.gui.widgets import UnitsWidget
from spcal.siunits import mass_units, signal_units, size_units


class Filter(object):
    operations: Dict[str, np.ufunc] = {
        ">": np.greater,
        "<": np.less,
        ">=": np.greater_equal,
        "<=": np.less_equal,
        "==": np.equal,
    }

    def __init__(self, name: str, unit: str, operation: str, value: float):
        self.name = name
        self.unit = unit
        self.operation = operation
        self.value = value

    def __repr__(self) -> str:
        return f"Filter({self.name}::{self.unit} {self.operation!r} {self.value!r})"

    @property
    def ufunc(self) -> np.ufunc:
        return Filter.operations[self.operation]


class FilterItemWidget(QtWidgets.QWidget):
    closeRequested = QtCore.Signal(QtWidgets.QWidget)

    unit_labels = {
        "Intensity": "signal",
        "Mass": "mass",
        "Size": "size",
        # "Intracellular Conc.": "cell_concentration",
    }

    def __init__(
        self,
        names: List[str],
        filter: Filter | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        self.names = QtWidgets.QComboBox()
        self.names.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContentsOnFirstShow)
        self.names.addItems(names)

        self.unit = QtWidgets.QComboBox()
        self.unit.addItems(list(FilterItemWidget.unit_labels.keys()))
        self.unit.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContentsOnFirstShow)
        self.unit.currentTextChanged.connect(self.changeUnits)

        self.operation = QtWidgets.QComboBox()
        self.operation.addItems(list(Filter.operations.keys()))

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
        layout.addWidget(self.names, 0)
        layout.addWidget(self.unit, 0)
        layout.addWidget(self.operation, 0)
        layout.addWidget(self.value, 1)
        layout.addWidget(self.button_close, 0, QtCore.Qt.AlignRight)
        self.setLayout(layout)

        if filter is not None:
            self.setFilter(filter)

    def setFilter(self, filter: Filter) -> None:
        index = self.names.findText(filter.name)
        if index == -1:
            raise KeyError(f"names combo has no name {filter.name}")
        label = next(
            lbl
            for lbl, unit in FilterItemWidget.unit_labels.items()
            if unit == filter.unit
        )
        self.names.setCurrentIndex(index)
        self.unit.setCurrentText(label)
        self.operation.setCurrentText(filter.operation)
        self.value.setBaseValue(filter.value)
        self.value.setBestUnit()

    def asFilter(self) -> Filter:
        return Filter(
            self.names.currentText(),
            FilterItemWidget.unit_labels[self.unit.currentText()],
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


class BooleanItemWidget(QtWidgets.QWidget):
    closeRequested = QtCore.Signal(QtWidgets.QWidget)

    def __init__(self, text: str = "Or", parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        frame = QtWidgets.QFrame()
        frame.setFrameShape(QtWidgets.QFrame.Shape.HLine)

        frame2 = QtWidgets.QFrame()
        frame2.setFrameShape(QtWidgets.QFrame.Shape.HLine)

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
        layout.addWidget(frame, 1)
        layout.addWidget(QtWidgets.QLabel(text), 0, QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(frame2, 1)
        layout.addWidget(self.button_close, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        self.setLayout(layout)

    def close(self) -> None:
        self.closeRequested.emit(self)
        super().close()


class FilterDialog(QtWidgets.QDialog):
    filtersChanged = QtCore.Signal(list)

    def __init__(
        self,
        names: List[str],
        filters: List[List[Filter]],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Compositional Filters")
        self.setMinimumSize(600, 600)

        self.names = names

        self.list = QtWidgets.QListWidget()
        self.list.setDragEnabled(True)
        self.list.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)

        self.action_add = create_action(
            "list-add", "Add Filter", "Add a new filter.", lambda: self.addFilter(None)
        )
        self.action_or = create_action(
            "",
            "Or",
            "Add an or group.",
            lambda: self.addBooleanOr(),
        )

        self.button_bar = QtWidgets.QToolBar()
        self.button_bar.addAction(self.action_add)
        self.button_bar.addAction(self.action_or)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Close
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.button_bar, 0)
        layout.addWidget(self.list, 1)
        layout.addWidget(self.button_box, 0)
        self.setLayout(layout)

        # add the filters
        for i in range(len(filters)):
            for filter in filters[i]:
                if filter.name in self.names:
                    self.addFilter(filter)
            if i < len(filters) - 1:
                self.addBooleanOr()

    def addFilter(self, filter: Filter | None = None):
        widget = FilterItemWidget(self.names, filter=filter)
        self.addWidget(widget)

    def addBooleanOr(self):
        widget = BooleanItemWidget()
        self.addWidget(widget)

    def addWidget(self, widget: QtWidgets.QWidget) -> None:
        widget.closeRequested.connect(self.removeItem)
        item = QtWidgets.QListWidgetItem()
        self.list.insertItem(self.list.count(), item)
        self.list.setItemWidget(item, widget)
        item.setSizeHint(widget.sizeHint())

    def removeItem(self, widget: FilterItemWidget) -> None:
        for i in range(self.list.count()):
            item = self.list.item(i)
            if self.list.itemWidget(item) == widget:
                self.list.takeItem(i)
                break

    def accept(self) -> None:
        filters = []
        group = []
        for i in range(self.list.count()):
            widget = self.list.itemWidget(self.list.item(i))
            if isinstance(widget, FilterItemWidget):
                if widget.value.baseValue() is not None:
                    group.append(widget.asFilter())
            elif isinstance(widget, BooleanItemWidget):
                if len(group) > 0:
                    filters.append(group)
                    group = []
        if len(group) > 0:
            filters.append(group)

        self.filtersChanged.emit(filters)
        super().accept()
