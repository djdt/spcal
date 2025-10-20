import logging
import re

from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.modelviews import BasicTable
from spcal.processing import SPCalIsotopeOptions
from spcal.siunits import (
    density_units,
    response_units,
)

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


# class UnitsTableModel(QtCore.QAbstractTableModel):
#     UNITS = {
#         "Density": density_units,
#         "Response": response_units,
#         "Mass Fraction": None,
#         "Diameter": size_units,
#         "Concentration": mass_concentration_units,
#     }
#
#     def __init__(self, columns: list[str], parent: QtCore.QObject | None = None):
#         super().__init__(parent=parent)
#
#         self.options: dict[str, SPCalIsotopeOptions] = {}
#         self.columns = columns
#
#     def columnCount(
#         self,
#         parent: QtCore.QModelIndex
#         | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
#     ) -> int:
#         if parent.isValid():
#             return 0
#         return len(self.columns)
#
#     def rowCount(
#         self,
#         parent: QtCore.QModelIndex
#         | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
#     ) -> int:
#         if parent.isValid():
#             return 0
#         return len(self.options)

# # Data
# def data(
# # Data
# def data(
#     self,
#     index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
#     role: int = QtCore.Qt.ItemDataRole.DisplayRole,
# ):
#     if not index.isValid():
#         return None
#
#     row, column = index.row(), index.column()
#
#     if role == QtCore.Qt.ItemDataRole.UserRole:
#         value = self.base_values[row, column]
#         if value is np.nan:
#             value = None
#     elif role == QtCore.Qt.ItemDataRole.DisplayRole:
#         unit = self.units[column]
#         value = self.base_values[row, column]
#         if unit is not None:
#             value /= self.column_units[column][unit]
#         value = str(value) if not np.isnan(value) else ""
#     else:  # pragma: no cover
#         return None
#     return value
#
# def setData(
#     self,
#     index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
#     value: Any,
#     role: int = QtCore.Qt.ItemDataRole.DisplayRole,
# ) -> bool:
#     if not index.isValid():
#         return False
#
#     row, column = index.row(), index.column()
#     if role == QtCore.Qt.ItemDataRole.UserRole:
#         self.base_values[row, column] = float(value)
#         self.dataChanged.emit(index, index, [role])
#         return True
#     elif role == QtCore.Qt.ItemDataRole.UserRole - 1:
#         self.units[column] = value
#         self.dataChanged.emit(index, index, [role])
#         return True
#     return False

# def headerData(
#     self,
#     section: int,
#     orientation: QtCore.Qt.Orientation,
#     role: int = QtCore.Qt.ItemDataRole.DisplayRole,
# ) -> str:
#     if role == QtCore.Qt.ItemDataRole.DisplayRole:
#         if orientation == QtCore.Qt.Orientation.Horizontal:
#             unit = self.units[section]
#             header = self.HEADERS[section]
#             if unit != "":
#                 header += f" ({unit})"
#             return header
#         else:
#             return self.isotopes[section]
#     return super().headerData(section, orientation, role)
#
# def flags(
#     self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex
# ) -> QtCore.Qt.ItemFlag:
#     return super().flags(index) | QtCore.Qt.ItemFlag.ItemIsEditable


class ComboHeaderView(QtWidgets.QHeaderView):
    """
    Params:
        selection_items: dict of section numbers to combobox items
        orientation: header type, horizontal or vertical
    """

    sectionChanged = QtCore.Signal(int)

    def __init__(
        self,
        section_items: dict[int, list[str]],
        orientation: QtCore.Qt.Orientation = QtCore.Qt.Orientation.Horizontal,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(orientation, parent)

        self.section_items = section_items

    def showComboBox(self, section: int) -> None:
        items = self.section_items.get(section, [])
        widget = QtWidgets.QComboBox(self)
        widget.addItems(items)
        widget.setCurrentText(self.model().headerData(section, self.orientation()))

        pos = self.sectionViewportPosition(section)
        size = self.sectionSizeFromContents(section)

        widget.setGeometry(QtCore.QRect(pos, 0, size.width(), size.height()))
        widget.currentTextChanged.connect(
            lambda value: self.model().setHeaderData(
                section, self.orientation(), value, QtCore.Qt.ItemDataRole.EditRole
            )
        )
        widget.currentTextChanged.connect(lambda: self.sectionChanged.emit(section))
        widget.currentTextChanged.connect(widget.deleteLater)
        widget.showPopup()

    def sectionSizeFromContents(self, logicalIndex: int) -> QtCore.QSize:
        size = super().sectionSizeFromContents(logicalIndex)
        option = QtWidgets.QStyleOptionComboBox()
        option.initFrom(self)
        return self.style().sizeFromContents(
            QtWidgets.QStyle.ContentsType.CT_ComboBox, option, size
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        logicalIndex = self.logicalIndexAt(event.position().toPoint())
        if logicalIndex in self.section_items:
            self.showComboBox(logicalIndex)
        else:
            super().mousePressEvent(event)

    def paintSection(
        self, painter: QtGui.QPainter, rect: QtCore.QRect, logicalIndex: int
    ) -> None:
        option = QtWidgets.QStyleOptionComboBox()
        option.initFrom(self)
        option.rect = rect  # type: ignore
        option.currentText = str(  # type: ignore
            self.model().headerData(logicalIndex, self.orientation())
        )
        if logicalIndex not in self.section_items:
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


# class IsotopeOptionsModel(QtCore.QAbstractItemModel):
#     def setData(index :QtCore.QModelIndex|QtCore.QPersistentModelIndex, value:Any, role: int = QtCore.Qt.ItemDataRole.EditRole) -> bool:
#         if not index.isValid():
#             return False
#
#         if role == QtCore.Qt.ItemDataRole.EditRole:
#             self.


class IsotopeOptionTable(BasicTable):
    HEADER_UNITS = {
        "Density": density_units,
        "Response": response_units,
        "Mass Fraction": None,
        # "Diameter": size_units,
        # "Concentration": mass_concentration_units,
    }

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(0, 3, parent=parent)

        header_items = {}
        for i, (key, units) in enumerate(self.HEADER_UNITS.items()):
            if units is None:
                continue
            header_items[i] = [f"{key} ({unit})" for unit in units.keys()]

        self.header = ComboHeaderView(header_items)
        self.current_units = {0: density_units["g/cm³"], 1: response_units["L/µg"]}

        self.header.sectionChanged.connect(self.adjustSectionValues)
        self.header.setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.setHorizontalHeader(self.header)
        self.setHorizontalHeaderLabels(
            ["Density (g/cm³)", "Response (L/µg)", "Mass Fraction"]
        )

    def unitForSection(self, section: int) -> float:
        text = self.model().headerData(section, QtCore.Qt.Orientation.Horizontal)
        m = re.match("([\\w ]+) \\((.+)\\)", text)
        if m is None:
            return 1.0
        return self.HEADER_UNITS[m.group(1)][m.group(2)]

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
        try:
            value = float(item.data(QtCore.Qt.ItemDataRole.EditRole))
            if column in self.current_units:
                value *= self.current_units[column]
            return value
        except ValueError:
            return None

    def asIsotopeOptions(self) -> dict[str, SPCalIsotopeOptions]:
        options = {}
        for row in range(self.rowCount()):
            label = self.verticalHeaderItem(row).text()
            options[label] = SPCalIsotopeOptions(
                self.baseValueForItem(row, 0),
                self.baseValueForItem(row, 1),
                self.baseValueForItem(row, 2),
            )
        return options


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
    table = IsotopeOptionTable()
    table.setRowCount(3)
    table.setVerticalHeaderLabels(["Ti42", "Fe56", "Au197"])
    # table.setHorizontalHeader(header)
    # table.horizontalHeader().sectionPressed.disconnect()
    # table.setHorizontalHeaderLabels(["Density (g/cm³)", "Response", "Mass Fraction"])
    # # table.setVerticalHeaderLabels(["", "Ti42", "Fe56", "Au197"])
    # # model.insertRows(0, 3)
    #
    # density_delegate = UnitsDelegate(density_units, "g/cm³")
    # table.setItemDelegateForColumn(0, density_delegate)
    # table = BasicTable(4, 3)
    # table.setItemDelegateForRow(0, ComboDelegate(list(density_units.keys())))
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
