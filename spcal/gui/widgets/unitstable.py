import re

from PySide6 import QtCore, QtWidgets

from spcal.gui.modelviews import BasicTable, ComboHeaderView
from spcal.gui.widgets.values import ValueWidgetDelegate


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

        sf = int(QtCore.QSettings().value("SigFigs", 4))  # type: ignore

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

        self.header = ComboHeaderView(header_items)
        self.header.setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.header.sectionChanged.connect(self.adjustSectionValues)

        self.setHorizontalHeader(self.header)
        self.setHorizontalHeaderLabels(header_labels)

    def adjustSectionValues(self, section: int) -> None:
        if section not in self.current_units:
            return
        current = self.current_units[section]
        new = self.unitForSection(section)
        self.current_units[section] = new

        for row in range(self.rowCount()):
            item = self.item(row, section)
            if item is not None:
                value = (
                    float(item.data(QtCore.Qt.ItemDataRole.EditRole)) * current / new
                )
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

    def setBaseValueForItem(self, row: int, column: int, value: float | None, error: float | None = None):
        item = self.item(row, column)
        if item is None:
            item = QtWidgets.QTableWidgetItem()
            self.setItem(row, column, item)
        if value is not None and column in self.current_units:
            value /= self.current_units[column]
        item.setData(QtCore.Qt.ItemDataRole.EditRole, value)
        if error is not None:
            if column in self.current_units:
                error /= self.current_units[column]
            item.setData(QtCore.Qt.ItemDataRole.UserRole, error)

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
