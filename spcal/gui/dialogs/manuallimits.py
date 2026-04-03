from PySide6 import QtCore, QtWidgets

import numpy as np

from spcal.gui.modelviews.values import ValueWidgetDelegate
from spcal.isotope import SPCalIsotopeBase

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
            "Here you can set manual limits for specific isotopes in the table below. "
            "Isotopes without a specific limit will use the default manual limit."
        )

        self.manual_limits = manual_limits

        self.table = QtWidgets.QTableWidget(len(isotopes), 1)
        self.table.setItemDelegate(ValueWidgetDelegate())
        self.table.setHorizontalHeaderLabels(["Limit"])
        self.table.setVerticalHeaderLabels([str(iso) for iso in isotopes])
        self.table.horizontalHeader().setStretchLastSection(True)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Close
        )

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        for i, isotope in enumerate(sorted(isotopes)):
            item = QtWidgets.QTableWidgetItem()
            item.setData(IsotopeRole, isotope)
            if isotope in manual_limits:
                item.setData(QtCore.Qt.ItemDataRole.EditRole, manual_limits[isotope])
            self.table.setItem(i, 0, item)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.table, 1)
        layout.addWidget(self.button_box, 0)
        self.setLayout(layout)

    def manualLimits(self) -> dict[SPCalIsotopeBase, float]:
        manual_limits = {}
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            assert item is not None
            value = item.data(QtCore.Qt.ItemDataRole.EditRole)
            if value is not None:
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
