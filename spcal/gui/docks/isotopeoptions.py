import logging
from typing import Any

import numpy as np
from PySide6 import QtCore, QtWidgets

from spcal.gui.modelviews import BasicTable
from spcal.gui.widgets.units import UnitsWidget
from spcal.gui.widgets.values import ValueWidget
from spcal.siunits import density_units, response_units

logger = logging.getLogger(__name__)


# class SPCalLimitOptionsDock(QtWidgets.QDockWidget):
#     optionsChanged = QtCore.Signal()
#
#     def __init__(self, parent: QtWidgets.QWidget | None = None):
#         super().__init__(parent)
#         self.setWindowTitle("Isotope Options")
#
#         sf = int(QtCore.QSettings().value("SigFigs", 4))  # type: ignore
#
#         self.action_density = create_action(
#             "folder-database",
#             "Lookup Density",
#             "Search for compound densities.",
#             self.dialogParticleDatabase,
#         )
#         self.action_mass_fraction = create_action(
#             "folder-calculate",
#             "Calculate Mass Fraction",
#             "Calculate the mass fraction and MW for a given formula.",
#             self.dialogMassFractionCalculator,
#         )
#         self.action_ionic_response = create_action(
#             "document-open",
#             "Ionic Response Tool",
#             "Read ionic responses from a file and apply to sample and reference.",
#             lambda: self.request.emit("ionic response"),
#         )
#
#         self.density = UnitsWidget(
#             {"g/cm³": 1e-3 * 1e6, "kg/m³": 1.0},
#             default_unit="g/cm³",
#             format=sf,
#         )
#         self.density.lineedit.addAction(
#             self.action_density, QtWidgets.QLineEdit.ActionPosition.TrailingPosition
#         )
#         self.response = UnitsWidget(
#             {
#                 "counts/(pg/L)": 1e15,
#                 "counts/(ng/L)": 1e12,
#                 "counts/(μg/L)": 1e9,
#                 "counts/(mg/L)": 1e6,
#             },
#             default_unit="counts/(μg/L)",
#             format=sf,
#         )
#         self.response.lineedit.addAction(
#             self.action_ionic_response,
#             QtWidgets.QLineEdit.ActionPosition.TrailingPosition,
#         )
#
#         self.massfraction = ValueWidget(
#             1.0,
#             validator=QtGui.QDoubleValidator(0.0, 1.0, 16),
#             format=sf,
#         )
#         self.massfraction.addAction(
#             self.action_mass_fraction,
#             QtWidgets.QLineEdit.ActionPosition.TrailingPosition,
#         )
#
#         self.density.setToolTip("Sample particle density.")
#         self.response.setToolTip(
#             "ICP-MS response for an ionic standard of this element."
#         )
#         self.massfraction.setToolTip(
#             "Ratio of the mass of the analyte over the mass of the particle."
#         )
#
#         self.density.baseValueChanged.connect(self.optionsChanged)
#         self.response.baseValueChanged.connect(self.optionsChanged)
#         self.massfraction.valueChanged.connect(self.optionsChanged)
#
#         self.inputs = QtWidgets.QGroupBox("Inputs")
#         input_layout = QtWidgets.QFormLayout()
#         input_layout.addRow("Density:", self.density)
#         input_layout.addRow("Ionic response:", self.response)
#         input_layout.addRow("Mass fraction:", self.massfraction)
#         self.inputs.setLayout(input_layout)
#
#         self.count = ValueWidget(0, format=("f", 0))
#         self.count.setReadOnly(True)
#         self.background_count = ValueWidget(format=sf)
#         self.background_count.setReadOnly(True)
#         self.lod_count = ValueWidget(format=sf)
#         self.lod_count.setReadOnly(True)
#         self.lod_label = OverLabel(self.lod_count, "")
#
#         self.outputs = QtWidgets.QGroupBox("Outputs")
#         output_layout = QtWidgets.QFormLayout()
#         output_layout.addRow("Particle count:", self.count)
#         output_layout.addRow("Background count:", self.background_count)
#         output_layout.addRow("Detection threshold:", self.lod_label)
#         self.outputs.setLayout(output_layout)
#
#         layout = QtWidgets.QHBoxLayout()
#         layout.addWidget(self.inputs)
#         layout.addWidget(self.outputs)
#
#         self.setLayout(layout)


class IsotopeOptionsModel(QtCore.QAbstractTableModel):
    HEADERS = ["Density", "Response", "Mass Fraction"]

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent=parent)

        self.base_values = np.array([[]])
        self.isotopes = []
        self.column_units = [density_units, response_units, None]
        self.units = ["g/cm³", "L/μg", None]

    def columnCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        if parent.isValid():
            return 0
        return self.base_values.shape[1]

    def rowCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        if parent.isValid():
            return 0
        return self.base_values.shape[0]

    # Data
    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ):
        if not index.isValid():
            return None

        row, column = index.row(), index.column()

        if role == QtCore.Qt.ItemDataRole.UserRole:
            value = self.base_values[row, column]
            if value is np.nan:
                value = None
        elif role == QtCore.Qt.ItemDataRole.DisplayRole:
            unit = self.units[column]
            value = self.base_values[row, column]
            if unit is not None:
                value /= self.column_units[column][unit]
            value = str(value) if not np.isnan(value) else ""
        elif role == QtCore.Qt.ItemDataRole.UserRole - 1:
            value = self.units[column]
        else:  # pragma: no cover
            return None
        return value

    def setData(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        value: Any,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> bool:
        if not index.isValid():
            return False

        row, column = index.row(), index.column()
        if role == QtCore.Qt.ItemDataRole.UserRole:
            self.base_values[row, column] = float(value)
            self.dataChanged.emit(index, index, [role])
            return True
        elif role == QtCore.Qt.ItemDataRole.UserRole - 1:
            self.units[column] = value
            self.dataChanged.emit(index, index, [role])
            return True
        return False

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> str:
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if orientation == QtCore.Qt.Orientation.Horizontal:
                unit = self.units[section]
                header = self.HEADERS[section]
                if unit != "":
                    header += f" ({unit})"
                return header
            else:
                return self.isotopes[section]
        return super().headerData(section, orientation, role)

    def flags(
        self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex
    ) -> QtCore.Qt.ItemFlag:
        return super().flags(index) | QtCore.Qt.ItemFlag.ItemIsEditable


class ValueDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(
        self,
        units: dict[str, float],
        default_unit: str | None = None,
        format: int = 6,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self.default_unit = default_unit
        self.units = units
        self.format = format

    def setDefaultUnit(self, unit: str) -> None:
        self.default_unit = unit

    def createEditor(
        self,
        parent: QtWidgets.QWidget,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> QtWidgets.QWidget:
        editor = ValueWidget(parent=parent)
        editor.setFrame(False)
        return editor

    def setEditorData(
        self,
        editor: QtWidgets.QWidget,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> None:
        value = index.data(QtCore.Qt.ItemDataRole.UserRole)
        assert isinstance(editor, ValueWidget)
        editor.setValue(value)

    def setModelData(
        self,
        editor: QtWidgets.QWidget,
        model: QtCore.QAbstractItemModel,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> None:
        assert isinstance(editor, UnitsWidget)
        model.setData(index, editor.value(), QtCore.Qt.ItemDataRole.UserRole)


class UnitsDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(
        self,
        units: dict[str, float],
        default_unit: str | None = None,
        format: int = 6,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self.units = units
        self.format = format

    def createEditor(
        self,
        parent: QtWidgets.QWidget,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> QtWidgets.QWidget:
        editor = UnitsWidget(self.units, format=self.format, parent=parent)
        return editor

    # def closeAndCommitEditor(self) -> None:
    #     sender = self.sender()
    #     assert isinstance(sender, UnitsWidget)
    #     self.commitData.emit(sender)

    def setEditorData(
        self,
        editor: QtWidgets.QWidget,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> None:
        value = index.data(QtCore.Qt.ItemDataRole.UserRole)
        unit = index.data(QtCore.Qt.ItemDataRole.UserRole - 1)
        assert isinstance(editor, UnitsWidget)
        editor.setUnit(unit)
        editor.setBaseValue(value)
        editor.lineedit.grabKeyboard()

    def setModelData(
        self,
        editor: QtWidgets.QWidget,
        model: QtCore.QAbstractItemModel,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> None:
        assert isinstance(editor, UnitsWidget)
        model.setItemData(
            index,
            {
                QtCore.Qt.ItemDataRole.UserRole: editor.baseValue(),
                QtCore.Qt.ItemDataRole.UserRole - 1: editor.unit(),
            },
        )

class ComboDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, options: list, parent: QtWidgets.QWidget|None=None):
        super().__init__(parent=parent)
        self.options = options

    def createEditor(
        self,
        parent: QtWidgets.QWidget,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> QtWidgets.QWidget:
        editor = QtWidgets.QComboBox(parent=parent)
        editor.addItems(self.options)
        return editor


    def setEditorData(
        self,
        editor: QtWidgets.QWidget,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> None:
        assert isinstance(editor, QtWidgets.QComboBox)
        editor.setCurrentText(index.data(QtCore.Qt.ItemDataRole.EditRole))

    def setModelData(
        self,
        editor: QtWidgets.QWidget,
        model: QtCore.QAbstractItemModel,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> None:
        assert isinstance(editor, QtWidgets.QComboBox)
        model.setData(index, editor.currentText(), QtCore.Qt.ItemDataRole.EditRole)

if __name__ == "__main__":
    app = QtWidgets.QApplication()

    dlg = QtWidgets.QWidget()

    # model = IsotopeOptionsModel()
    # table = QtWidgets.QTableView()
    # table.setModel(model)
    # sf = 4
    # model.base_values = np.full((3, 3), np.nan)
    # model.isotopes = ["Ti42", "Fe56", "Au197"]
    #
    # # table = BasicTable(4, 3)
    # # table.setHorizontalHeader(header)
    # # table.setHorizontalHeaderLabels(["Density", "Response", "Mass Fraction"])
    # # table.setVerticalHeaderLabels(["", "Ti42", "Fe56", "Au197"])
    # # model.insertRows(0, 3)
    #
    # density_delegate = UnitsDelegate(density_units, "g/cm³")
    # table.setItemDelegateForColumn(0, density_delegate)
    table = BasicTable(4, 3)
    table.setItemDelegateForRow(0, ComboDelegate(list(density_units.keys())))
    # # table.setItemDelegateForColumn(
    # #     1,
    # #     UnitsDelegate(
    # #         "counts/(μg/L)",
    # #     ),
    # # )
    # table.setItemDelegateForColumn(2, UnitsDelegate({"": 1.0}, ""))

    layout = QtWidgets.QVBoxLayout()
    layout.addWidget(table)
    dlg.setLayout(layout)

    dlg.show()
    app.exec()
