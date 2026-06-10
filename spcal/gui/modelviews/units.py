from spcal.gui.util import create_action
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.modelviews import (
    BaseValueErrorRole,
    BaseValueRole,
    CurrentUnitRole,
    UnitLabelRole,
    UnitsRole,
    ValueErrorRole,
)


class UnitsModel(QtCore.QAbstractTableModel):
    """A model that can have different units for each row or column.
    To implement, subclass and reimplment `data` and `setData` to return valid values for
    'BaseValueRole' and 'BasrErrorRole' and both `rowCount` and `columnCount`.
    """

    def __init__(
        self,
        units_labels: list[str],
        current_units: list[str],
        units: list[dict[str, float]],
        units_tooltips: list[str] | None = None,
        units_orientation: QtCore.Qt.Orientation = QtCore.Qt.Orientation.Horizontal,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)

        self.units_orientation = units_orientation

        self.unit_labels = units_labels
        self.unit_tooltips = units_tooltips

        self.current_unit = current_units
        self.units = units

    def modifer(
        self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex
    ) -> float:
        return self.units[index.column()].get(self.current_unit[index.column()], 1.0)

    def unitStartEndIndices(
        self, section: int
    ) -> tuple[QtCore.QModelIndex, QtCore.QModelIndex]:
        if self.units_orientation == QtCore.Qt.Orientation.Horizontal:
            return self.index(0, section), self.index(self.rowCount() - 1, section)
        else:
            return self.index(section, 0), self.index(section, self.columnCount() - 1)

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ):
        if orientation == self.units_orientation:
            if role in [
                QtCore.Qt.ItemDataRole.DisplayRole,
                QtCore.Qt.ItemDataRole.EditRole,
            ]:
                label = self.unit_labels[section]
                unit = self.current_unit[section]
                if unit != "":
                    label = f"{label} ({unit})"
                return label
            elif (
                role == QtCore.Qt.ItemDataRole.ToolTipRole
                and self.unit_tooltips is not None
            ):
                return self.unit_tooltips[section]
            elif role == UnitLabelRole:
                return self.unit_labels[section]
            elif role == CurrentUnitRole:
                return self.current_unit[section]
            elif role == UnitsRole:
                return self.units[section]
        return super().headerData(section, orientation, role)

    def setHeaderData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        value: Any,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> bool:
        if orientation == self.units_orientation:
            if role == UnitLabelRole:
                self.unit_labels[section] = value
            elif role == CurrentUnitRole:
                self.current_unit[section] = value
                tl, br = self.unitStartEndIndices(section)
                if tl.isValid() and br.isValid():
                    self.dataChanged.emit(tl, br)
            elif role == UnitsRole:
                self.units[section] = value
                tl, br = self.unitStartEndIndices(section)
                if tl.isValid() and br.isValid():
                    self.dataChanged.emit(tl, br)
            else:
                return False
            self.headerDataChanged.emit(orientation, section, section)
            return True
        else:
            return super().setHeaderData(section, orientation, value, role)

    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.EditRole,
    ) -> Any:
        if not index.isValid():
            return

        if role == BaseValueRole:
            raise NotImplementedError
        elif role == BaseValueErrorRole:
            raise NotImplementedError
        elif role in [
            QtCore.Qt.ItemDataRole.DisplayRole,
            QtCore.Qt.ItemDataRole.EditRole,
        ]:
            base = self.data(index, BaseValueRole)
            if base is not None:
                base = float(base) / self.modifer(index)
            if role == QtCore.Qt.ItemDataRole.EditRole:
                return base
            else:
                return str(base)
        elif role == ValueErrorRole:
            base = self.data(index, BaseValueErrorRole)
            if base is not None:
                base = float(base) / self.modifer(index)
            return base
        elif role == CurrentUnitRole:
            return self.current_unit[index.column()]
        elif role == UnitsRole:
            return self.units[index.column()]
        return None

    def setData(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        value: Any,
        role: int = QtCore.Qt.ItemDataRole.EditRole,
    ) -> bool:
        if not index.isValid():
            return False

        if role == BaseValueRole:
            raise NotImplementedError
        elif role == BaseValueErrorRole:
            raise NotImplementedError
        elif role == CurrentUnitRole:
            self.current_unit[index.column()] = value
        elif role == UnitsRole:
            self.units[index.column()] = value
        elif role in [
            QtCore.Qt.ItemDataRole.EditRole,
        ]:
            if value == "":
                value = None
            if value is not None:
                value = float(value) * self.modifer(index)
            return self.setData(index, value, BaseValueRole)
        elif role == ValueErrorRole:
            if value == "":
                value = None
            if value is not None:
                value = float(value) * self.modifer(index)
            return self.setData(index, value, BaseValueErrorRole)
        else:
            return False

        self.dataChanged.emit(index, index)
        return True


class UnitsHeaderView(QtWidgets.QHeaderView):
    sectionChanged = QtCore.Signal(int)

    def showComboBox(self, section: int):
        units = self.model().headerData(section, self.orientation(), UnitsRole)

        widget = QtWidgets.QComboBox(self)
        widget.addItems(list(units.keys()))
        widget.setCurrentText(
            self.model().headerData(section, self.orientation(), CurrentUnitRole)
        )

        pos = self.sectionViewportPosition(section)
        size = self.sectionSizeFromContents(section)

        widget.setGeometry(QtCore.QRect(pos, 0, size.width(), size.height()))
        widget.currentTextChanged.connect(
            lambda value: self.model().setHeaderData(
                section, self.orientation(), value, CurrentUnitRole
            )
        )
        widget.currentIndexChanged.connect(self.sectionChanged)
        widget.currentTextChanged.connect(widget.deleteLater)
        widget.showPopup()

    def sectionSizeFromContents(self, logicalIndex: int) -> QtCore.QSize:
        size = super().sectionSizeFromContents(logicalIndex)
        option = QtWidgets.QStyleOptionComboBox()
        option.initFrom(self)
        return self.style().sizeFromContents(
            QtWidgets.QStyle.ContentsType.CT_ComboBox, option, size
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            logicalIndex = self.logicalIndexAt(event.position().toPoint())
            units = self.model().headerData(logicalIndex, self.orientation(), UnitsRole)
            if len(units) > 1:
                self.showComboBox(logicalIndex)
                event.accept()
                return
        super().mousePressEvent(event)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        def on_action_show(action: QtGui.QAction):
            self.showSection(action.data())

        logicalIndex = self.logicalIndexAt(event.pos())
        menu = QtWidgets.QMenu(parent=self)
        if logicalIndex >= 0:
            action_hide = create_action(
                "view-hidden",
                "Hide Section",
                "Hide this section from view.",
                lambda: self.hideSection(logicalIndex),
            )
            action_hide.setParent(self)
            menu.addAction(action_hide)

        if self.hiddenSectionCount() > 0:
            menu_show = menu.addMenu("Hidden Section(s)")
            action_group = QtGui.QActionGroup(self)
            action_group.triggered.connect(on_action_show)
            for i in range(self.count()):
                if self.isSectionHidden(i):
                    action = QtGui.QAction(
                        QtGui.QIcon.fromTheme("view-visible"),
                        f"Show {self.model().headerData(i, self.orientation())}",
                    )
                    action.setData(i)
                    action_group.addAction(action)
                    menu_show.addAction(action)

        menu.popup(event.globalPos())

    def paintSection(
        self, painter: QtGui.QPainter, rect: QtCore.QRect, logicalIndex: int
    ):
        option = QtWidgets.QStyleOptionComboBox()
        option.initFrom(self)
        option.rect = rect
        option.currentText = str(
            self.model().headerData(
                logicalIndex, self.orientation(), QtCore.Qt.ItemDataRole.EditRole
            )
        )
        units = self.model().headerData(logicalIndex, self.orientation(), UnitsRole)
        if len(units) < 2:
            option.subControls = (
                option.subControls & ~QtWidgets.QStyle.SubControl.SC_ComboBoxArrow
            )

        if self.hasFocus():
            option.state = QtWidgets.QStyle.StateFlag.State_Selected
        self.style().drawComplexControl(
            QtWidgets.QStyle.ComplexControl.CC_ComboBox, option, painter
        )
        self.style().drawControl(
            QtWidgets.QStyle.ControlElement.CE_ComboBoxLabel, option, painter
        )
