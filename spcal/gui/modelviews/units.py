import re
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.modelviews.basic import BasicTable, BasicTableView
from spcal.gui.modelviews.headers import ComboHeaderView
from spcal.gui.modelviews.values import ValueWidgetDelegate


class UnitsModel(QtCore.QAbstractTableModel):
    BaseValueRole = QtCore.Qt.ItemDataRole.UserRole + 10
    CurrentUnitRole = QtCore.Qt.ItemDataRole.UserRole + 11
    UnitsRole = QtCore.Qt.ItemDataRole.UserRole + 12
    UnitLabelRole = QtCore.Qt.ItemDataRole.UserRole + 13

    ErrorRole = ValueWidgetDelegate.ErrorRole
    BaseErrorRole = ValueWidgetDelegate.ErrorRole + 1

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
            return self.index(0, section), self.index(self.rowCount(), section)
        else:
            return self.index(section, 0), self.index(section, self.columnCount())

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
            elif role == UnitsModel.UnitLabelRole:
                return self.unit_labels[section]
            elif role == UnitsModel.CurrentUnitRole:
                return self.current_unit[section]
            elif role == UnitsModel.UnitsRole:
                self.dataChanged.emit(*self.unitStartEndIndices(section))
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
            if role == UnitsModel.UnitLabelRole:
                self.unit_labels[section] = value
            elif role == UnitsModel.CurrentUnitRole:
                self.current_unit[section] = value
                self.dataChanged.emit(*self.unitStartEndIndices(section))
            elif role == UnitsModel.UnitsRole:
                self.units[section] = value
                self.dataChanged.emit(*self.unitStartEndIndices(section))
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

        if role == UnitsModel.BaseValueRole:
            raise NotImplementedError
        elif role == UnitsModel.BaseErrorRole:
            raise NotImplementedError
        elif role in [
            QtCore.Qt.ItemDataRole.DisplayRole,
            QtCore.Qt.ItemDataRole.EditRole,
        ]:
            base = self.data(index, UnitsModel.BaseValueRole)
            if base is not None:
                base = float(base) * self.modifer(index)
            return base
        elif role == UnitsModel.ErrorRole:
            base = self.data(index, UnitsModel.BaseErrorRole)
            if base is not None:
                base = float(base) * self.modifer(index)
            return base

    def setData(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        value: Any,
        role: int = QtCore.Qt.ItemDataRole.EditRole,
    ) -> bool:
        if not index.isValid():
            return False

        if role == UnitsModel.BaseValueRole:
            raise NotImplementedError
        elif role == UnitsModel.BaseErrorRole:
            raise NotImplementedError
        elif role == UnitsModel.CurrentUnitRole:
            self.current_unit[index.column()] = value
        elif role == UnitsModel.UnitsRole:
            self.units[index.column()] = value
        elif role in [
            QtCore.Qt.ItemDataRole.EditRole,
        ]:
            if value is not None:
                value = float(value) / self.modifer(index)
            return self.setData(index, value, UnitsModel.BaseValueRole)
        elif role == UnitsModel.ErrorRole:
            if value is not None:
                value = float(value) / self.modifer(index)
            return self.setData(index, value, UnitsModel.BaseErrorRole)
        else:
            return False

        self.dataChanged.emit(index, index)
        return True


class UnitsHeaderView(QtWidgets.QHeaderView):
    sectionChanged = QtCore.Signal(int)

    def showComboBox(self, section: int):
        units = self.model().headerData(
            section, self.orientation(), UnitsModel.UnitsRole
        )

        widget = QtWidgets.QComboBox(self)
        widget.addItems(list(units.keys()))
        widget.setCurrentText(
            self.model().headerData(
                section, self.orientation(), UnitsModel.CurrentUnitRole
            )
        )

        pos = self.sectionViewportPosition(section)
        size = self.sectionSizeFromContents(section)

        widget.setGeometry(QtCore.QRect(pos, 0, size.width(), size.height()))
        widget.currentTextChanged.connect(
            lambda value: self.model().setHeaderData(
                section, self.orientation(), value, UnitsModel.CurrentUnitRole
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
        units = self.model().headerData(
            logicalIndex, self.orientation(), UnitsModel.UnitsRole
        )
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
        units = self.model().headerData(
            logicalIndex, self.orientation(), UnitsModel.UnitsRole
        )
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


class UnitsTableView(BasicTableView):
    def __init__(
        self,
        headers: list[
            tuple[str, dict[str, float] | None, str | None, tuple[float, float] | None]
        ],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent=parent)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        self.header_units: dict[int, dict] = {}
        self.current_units: dict[int, float] = {}

        self.header = ComboHeaderView({})
        self.header.setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.setHorizontalHeader(self.header)
        self.setHeaders(headers)
        self.header.sectionChanged.connect(self.adjustSectionValues)

    def setHeaders(
        self,
        headers: list[
            tuple[str, dict[str, float] | None, str | None, tuple[float, float] | None]
        ],
    ):
        sf = int(QtCore.QSettings().value("SigFigs", 4))  # type: ignore

        self.header_units.clear()
        self.current_units.clear()

        header_items = {}
        header_labels = []
        for i, (label, units, default, ranges) in enumerate(headers):
            if units is not None:
                header_items[i] = [f"{label} ({unit})" for unit in units.keys()]
                if default is None:
                    default = next(iter(units.keys()))
                self.header_units[i] = units
                self.current_units[i] = units[default]
                header_labels.append(f"{label} ({default})")
            else:
                header_labels.append(label)

            delegate = ValueWidgetDelegate(sf, parent=self)
            if ranges is not None:
                delegate.setMin(ranges[0])
                delegate.setMax(ranges[1])
            self.setItemDelegateForColumn(i, delegate)

        # self.setHorizontalHeaderLabels(header_labels)
        self.header.section_items = header_items

    def adjustSectionValues(self, section: int):
        if section not in self.current_units:
            return
        current = self.current_units[section]
        new = self.unitForSection(section)
        self.current_units[section] = new

        for row in range(self.model().rowCount()):
            index = self.model().index(row, section)
            if index.isValid():
                value = index.data(QtCore.Qt.ItemDataRole.EditRole)
                if value is not None:
                    value = value * current / new
                self.model().setData(index, value, role=QtCore.Qt.ItemDataRole.EditRole)

    def baseValueForIndex(self, index: QtCore.QModelIndex) -> float | None:
        if not index.isValid():
            return None
        value = index.data(QtCore.Qt.ItemDataRole.EditRole)
        if value is not None:
            value = float(index.data(QtCore.Qt.ItemDataRole.EditRole))
            if index.column() in self.current_units:
                value *= self.current_units[index.column()]
        return value

    def setBaseValueForItem(
        self, index: QtCore.QModelIndex, value: float | None, error: float | None = None
    ):
        if not index.isValid():
            return
            # item = QtWidgets.QTableWidgetItem()
            # self.setItem(row, column, item)

        if value is not None and index.column() in self.current_units:
            value = float(value) / self.current_units[index.column()]
        self.model().setData(index, value, QtCore.Qt.ItemDataRole.EditRole)
        if error is not None and index.column() in self.current_units:
            error = float(error) / self.current_units[index.column()]
        self.model().setData(index, error, ValueWidgetDelegate.ErrorRole)

    def sizeHint(self) -> QtCore.QSize:
        width = sum(
            self.horizontalHeader().sectionSize(i)
            for i in range(self.model().columnCount())
        )
        width += self.verticalHeader().width()
        height = super().sizeHint().height()
        return QtCore.QSize(width, height)

    def unitForSection(self, section: int) -> float:
        text = self.model().headerData(section, QtCore.Qt.Orientation.Horizontal)
        m = re.match("([\\w ]+) \\((.+)\\)", text)
        if m is None:
            return 1.0
        return self.header_units[section][m.group(2)]


class UnitsTable(BasicTable):
    def __init__(
        self,
        headers: list[
            tuple[str, dict[str, float] | None, str | None, tuple[float, float] | None]
        ],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(0, len(headers), parent=parent)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        self.header_units: dict[int, dict] = {}
        self.current_units: dict[int, float] = {}

        self.header = ComboHeaderView({})
        self.header.setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.setHorizontalHeader(self.header)
        self.setHeaders(headers)
        self.header.sectionChanged.connect(self.adjustSectionValues)

    def setHeaders(
        self,
        headers: list[
            tuple[str, dict[str, float] | None, str | None, tuple[float, float] | None]
        ],
    ):
        sf = int(QtCore.QSettings().value("SigFigs", 4))  # type: ignore

        self.header_units.clear()
        self.current_units.clear()

        header_items = {}
        header_labels = []
        for i, (label, units, default, ranges) in enumerate(headers):
            if units is not None:
                header_items[i] = [f"{label} ({unit})" for unit in units.keys()]
                if default is None:
                    default = next(iter(units.keys()))
                self.header_units[i] = units
                self.current_units[i] = units[default]
                header_labels.append(f"{label} ({default})")
            else:
                header_labels.append(label)

            delegate = ValueWidgetDelegate(sf, parent=self)
            if ranges is not None:
                delegate.setMin(ranges[0])
                delegate.setMax(ranges[1])
            self.setItemDelegateForColumn(i, delegate)

        self.setHorizontalHeaderLabels(header_labels)
        self.header.section_items = header_items

    def adjustSectionValues(self, section: int):
        if section not in self.current_units:
            return
        current = self.current_units[section]
        new = self.unitForSection(section)
        self.current_units[section] = new

        for row in range(self.rowCount()):
            item = self.item(row, section)
            if item is not None:
                value = item.data(QtCore.Qt.ItemDataRole.EditRole)
                if value is not None:
                    value = value * current / new
                item.setData(QtCore.Qt.ItemDataRole.EditRole, value)

    def baseValueForItem(self, row: int, column: int) -> float | None:
        item = self.item(row, column)
        if item is None:
            return None
        value = item.data(QtCore.Qt.ItemDataRole.EditRole)
        if value is not None:
            value = float(item.data(QtCore.Qt.ItemDataRole.EditRole))
            if column in self.current_units:
                value *= self.current_units[column]
        return value

    def setBaseValueForItem(
        self, row: int, column: int, value: float | None, error: float | None = None
    ):
        item = self.item(row, column)
        if item is None:
            item = QtWidgets.QTableWidgetItem()
            self.setItem(row, column, item)

        if value is not None and column in self.current_units:
            value = float(value) / self.current_units[column]
        item.setData(QtCore.Qt.ItemDataRole.EditRole, value)
        if error is not None and column in self.current_units:
            error = float(error) / self.current_units[column]
        item.setData(ValueWidgetDelegate.ErrorRole, error)

    def sizeHint(self) -> QtCore.QSize:
        width = sum(
            self.horizontalHeader().sectionSize(i) for i in range(self.columnCount())
        )
        width += self.verticalHeader().width()
        height = super().sizeHint().height()
        return QtCore.QSize(width, height)

    def unitForSection(self, section: int) -> float:
        text = self.model().headerData(section, QtCore.Qt.Orientation.Horizontal)
        m = re.match("([\\w ]+) \\((.+)\\)", text)
        if m is None:
            return 1.0
        return self.header_units[section][m.group(2)]
