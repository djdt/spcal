from typing import Dict, Generator, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.models import NumpyRecArrayTableModel, SearchColumnsProxyModel
from spcal.gui.widgets import ValidColorLineEdit
from spcal.npdb import db


class FormulaValidator(QtGui.QValidator):
    def __init__(
        self, regex: QtCore.QRegularExpression, parent: QtCore.QObject | None = None
    ):
        super().__init__(parent)
        self.regex = regex

    def validate(self, input: str, _: int) -> QtGui.QValidator.State:
        iter = self.regex.globalMatch(input)
        if len(input) == 0:
            return QtGui.QValidator.Acceptable
        if not str.isalnum(input.replace(".", "")):
            return QtGui.QValidator.Invalid
        if not iter.hasNext():  # no match
            return QtGui.QValidator.Intermediate
        while iter.hasNext():
            match = iter.next()
            if match.captured(1) not in db["elements"]["Symbol"]:
                return QtGui.QValidator.Intermediate
        return QtGui.QValidator.Acceptable


class MassFractionCalculatorDialog(QtWidgets.QDialog):
    ratiosChanged = QtCore.Signal()
    ratiosSelected = QtCore.Signal(dict)
    molarMassSelected = QtCore.Signal(float)

    def __init__(self, formula: str = "", parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Mass Fraction Calculator")
        self.resize(300, 120)

        self.regex = QtCore.QRegularExpression("([A-Z][a-z]?)([0-9\\.]*)")
        self.ratios: Dict[str, float] = {}
        self.mw = 0.0

        self.lineedit_formula = ValidColorLineEdit(formula)
        self.lineedit_formula.setPlaceholderText("Molecular Formula")
        self.lineedit_formula.setValidator(FormulaValidator(self.regex))
        self.lineedit_formula.textChanged.connect(self.recalculate)

        self.label_mw = QtWidgets.QLabel("MW = 0 g/mol")

        self.textedit_ratios = QtWidgets.QTextEdit()
        self.textedit_ratios.setReadOnly(True)
        self.textedit_ratios.setFont(QtGui.QFont("Courier"))

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )

        self.ratiosChanged.connect(self.updateLabels)
        self.ratiosChanged.connect(self.completeChanged)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.lineedit_formula, 0)
        layout.addWidget(self.label_mw, 0)
        layout.addWidget(self.textedit_ratios, 1)
        layout.addWidget(self.button_box, 0)

        self.setLayout(layout)
        self.completeChanged()

    def accept(self) -> None:
        self.ratiosSelected.emit(self.ratios)
        self.molarMassSelected.emit(self.mw)
        super().accept()

    def completeChanged(self) -> None:
        complete = self.isComplete()
        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(complete)

    def isComplete(self) -> bool:
        return len(self.ratios) > 0

    def recalculate(self) -> None:
        """Calculates the mass fraction of each valid element in the formula."""
        self.ratios = {}
        elements = db["elements"]
        for element, number in self.searchFormula():
            idx = np.flatnonzero(elements["Symbol"] == element)
            if idx.size > 0:
                ratio = elements["MW"][idx[0]] * float(number or 1.0)
                self.ratios[element] = self.ratios.get(element, 0.0) + ratio
        self.mw = sum(self.ratios.values())
        for element in self.ratios:
            self.ratios[element] = self.ratios[element] / self.mw
        self.ratiosChanged.emit()

    def searchFormula(self) -> Generator[Tuple[str, float], None, None]:
        iter = self.regex.globalMatch(self.lineedit_formula.text())
        while iter.hasNext():
            match = iter.next()
            yield match.captured(1), float(match.captured(2) or 1.0)

    def updateLabels(self) -> None:
        self.textedit_ratios.setPlainText("")
        if len(self.ratios) == 0:
            return
        text = "<html>"
        for i, (element, ratio) in enumerate(self.ratios.items()):
            if i == 0:
                text += "<b>"
            text += f"{element:<2}&nbsp;{ratio:.4f}&nbsp;&nbsp;"
            if i == 0:
                text += "</b>"
            if i % 3 == 2:
                text += "<br>"
        text += "</html>"
        self.textedit_ratios.setText(text)

        self.label_mw.setText(f"MW = {self.mw:.4g} g/mol")


class ParticleDatabaseDialog(QtWidgets.QDialog):
    densitySelected = QtCore.Signal(float)
    formulaSelected = QtCore.Signal(str)

    def __init__(self, formula: str = "", parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Density Database")
        self.resize(800, 600)

        self.lineedit_search = QtWidgets.QLineEdit(formula)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.model = NumpyRecArrayTableModel(
            np.concatenate((db["inorganic"], db["polymer"])),
            column_formats={"Density": "{:.4g}"},
        )
        self.proxy = SearchColumnsProxyModel([0, 1])
        self.proxy.setSourceModel(self.model)

        self.table = QtWidgets.QTableView()
        self.table.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContentsOnFirstShow
        )
        self.table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setModel(self.proxy)
        self.table.setColumnHidden(4, True)

        self.lineedit_search.textChanged.connect(self.searchDatabase)
        self.lineedit_search.textChanged.connect(self.table.clearSelection)
        self.table.pressed.connect(self.completeChanged)
        self.table.doubleClicked.connect(self.accept)
        self.proxy.rowsInserted.connect(self.completeChanged)
        self.proxy.rowsRemoved.connect(self.completeChanged)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Search"), 0)
        layout.addWidget(self.lineedit_search, 0)
        layout.addWidget(self.table)
        layout.addWidget(self.button_box, 0)

        self.setLayout(layout)
        self.completeChanged()

    def searchDatabase(self, string: str) -> None:
        self.proxy.setSearchString(string)

    def isComplete(self) -> bool:
        return len(self.table.selectedIndexes()) > 0

    def completeChanged(self) -> None:
        complete = self.isComplete()
        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(complete)

    def accept(self) -> None:
        idx = self.table.selectedIndexes()
        self.densitySelected.emit(float(self.proxy.data(idx[3])))
        self.fromulaSelected.emit(float(self.proxy.data(idx[0])))
        super().accept()
