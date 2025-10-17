import logging
import numpy as np

from PySide6 import QtCore, QtWidgets

from spcal.gui.modelviews import BasicTable
from spcal.gui.widgets.units import UnitsWidget
from spcal.siunits import density_units

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

from spcal.gui.models import NumpyRecArrayTableModel


class IsotopeOptionsModel(NumpyRecArrayTableModel):
    def __init__(self, isotopes: list[str], parent: QtCore.QObject | None = None):
        array = np.zeros(
            0, dtype=[(i, float) for i in isotopes]
        )
        super().__init__(array, QtCore.Qt.Orientation.Horizontal, parent=parent)


class UnitDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(
        self,
        units: dict[str, float],
        default_unit: str | None = None,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self.default_unit = default_unit
        self.units = units

    def createEditor(
        self,
        parent: QtWidgets.QWidget,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> QtWidgets.QWidget:
        editor = UnitsWidget(self.units, self.default_unit, parent=parent)
        editor.lineedit.setFrame(False)
        return editor

    def setEditorData(
        self,
        editor: QtWidgets.QWidget,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> None:
        value = index.data(QtCore.Qt.ItemDataRole.EditRole)
        assert isinstance(editor, UnitsWidget)
        editor.setBaseValue(value)

    def setModelData(
        self,
        editor: QtWidgets.QWidget,
        model: QtCore.QAbstractItemModel,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> None:
        assert isinstance(editor, UnitsWidget)
        model.setData(index, editor.baseValue(), QtCore.Qt.ItemDataRole.EditRole)

    def updateEditorGeometry(
        self,
        editor: QtWidgets.QWidget,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> None:
        editor.setGeometry(option.rect)  # type: ignore , works

    def sizeHint(
        self,
        option: QtWidgets.QStyleOptionViewItem
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> QtCore.QSize:
        return


if __name__ == "__main__":
    app = QtWidgets.QApplication()

    dlg = QtWidgets.QWidget()

    model = IsotopeOptionsModel(["Ti42", "Fe56", "Au197"])

    table = QtWidgets.QTableWidget()
    table.setModel(model)
    # table.setHorizontalHeaderLabels(["Density", "Response", "Mass Fraction"])
    # table.setVerticalHeaderLabels(["Ti42", "Fe56", "Au197"])

    table.setItemDelegateForColumn(0, UnitDelegate(density_units, "g/cm³"))
    table.setItemDelegateForColumn(
        1,
        UnitDelegate(
            {
                "counts/(pg/L)": 1e15,
                "counts/(ng/L)": 1e12,
                "counts/(μg/L)": 1e9,
                "counts/(mg/L)": 1e6,
            },
            "counts/(μg/L)",
        ),
    )
    table.setItemDelegateForColumn(2, UnitDelegate({"": 1.0}, ""))

    layout = QtWidgets.QVBoxLayout()
    layout.addWidget(table)
    dlg.setLayout(layout)

    dlg.show()
    app.exec()
