from PySide6 import QtCore, QtWidgets

import numpy as np

from spcal.gui.modelviews.values import ValueWidgetDelegate
from spcal.isotope import SPCalIsotope, SPCalIsotopeBase

from spcal.gui.modelviews import IsotopeRole


class ManualLimitDialog(QtWidgets.QDialog):
    manualLimitsChanged = QtCore.Signal(object)

    def __init__(
        self,
        manual_limits: dict[SPCalIsotopeBase, float],
        isotopes: list[SPCalIsotopeBase],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("SPCal Manual Limits")
        self.setWhatsThis(
            "Here you can set manual limits for specific elements in the table below, or a default manual limit that is applied to all elements without a specific limit."
        )

        self.manual_limits = manual_limits

        self.table = QtWidgets.QTableWidget(len(isotopes), 2)
        self.table.setHorizontalHeaderLabels(["Isotope", "Limit"])
        self.table.setItemDelegateForColumn(1, ValueWidgetDelegate())

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Close
        )

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        for i, isotope in enumerate(
            sorted(
                isotopes,
                key=lambda i: i.isotope if isinstance(i, SPCalIsotope) else 9999,
            )
        ):
            item = QtWidgets.QTableWidgetItem()
            item.setText(str(isotope))
            item.setData(IsotopeRole, isotope)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 0, item)

            item = QtWidgets.QTableWidgetItem()
            if isotope in manual_limits:
                item.setData(QtCore.Qt.ItemDataRole.EditRole, manual_limits[isotope])
            self.table.setItem(i, 1, item)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.table, 1)
        layout.addWidget(self.button_box, 0)
        self.setLayout(layout)

    def manualLimits(self) -> dict[SPCalIsotopeBase, float]:
        manual_limits = {}
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 1)
            assert item is not None
            value = item.data(QtCore.Qt.ItemDataRole.EditRole)
            if value is not None:
                item = self.table.item(row, 0)
                assert item is not None
                manual_limits[item.data(IsotopeRole)] = value
        return manual_limits

    def accept(self):
        manual_limits = self.manualLimits()
        if manual_limits.keys() != self.manual_limits.keys():
            self.manualLimitsChanged.emit(manual_limits)
        else:
            for isotope in manual_limits:
                if not np.isclose(manual_limits[isotope], self.manual_limits[isotope]):
                    self.manualLimitsChanged.emit(manual_limits)
                    break
        super().accept()
