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
    def __init__(
        self,
        units_orientation: QtCore.Qt.Orientation = QtCore.Qt.Orientation.Horizontal,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)

        self.units_orientation = units_orientation

        self.unit_labels: list[str] = []
        self.current_unit: list[str] = []
        self.units: list[dict[str, float]] = []

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
                self.dataChanged.emit(tl, br)
            elif role == UnitsRole:
                self.units[section] = value
                tl, br = self.unitStartEndIndices(section)
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
            return base
        elif role == ValueErrorRole:
            base = self.data(index, BaseValueErrorRole)
            if base is not None:
                base = float(base) / self.modifer(index)
            return base
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
            if value is not None:
                value = float(value) * self.modifer(index)
            return self.setData(index, value, BaseValueRole)
        elif role == ValueErrorRole:
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
        logicalIndex = self.logicalIndexAt(event.position().toPoint())
        units = self.model().headerData(logicalIndex, self.orientation(), UnitsRole)
        if len(units) > 1:
            self.showComboBox(logicalIndex)
        else:
            super().mousePressEvent(event)

    def paintSection(
        self, painter: QtGui.QPainter, rect: QtCore.QRect, logicalIndex: int
    ):
        option = QtWidgets.QStyleOptionComboBox()
        option.initFrom(self)
        option.rect = rect  # type: ignore
        option.currentText = str(  # type: ignore
            self.model().headerData(
                logicalIndex, self.orientation(), QtCore.Qt.ItemDataRole.EditRole
            )
        )
        units = self.model().headerData(logicalIndex, self.orientation(), UnitsRole)
        if len(units) < 2:
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
