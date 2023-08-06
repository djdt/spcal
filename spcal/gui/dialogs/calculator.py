from typing import Dict, List

import numpy as np
import numpy.lib.recfunctions as rfn
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.inputs import ReferenceWidget, SampleWidget
from spcal.gui.widgets import CollapsableWidget
from spcal.pratt import (
    BinaryFunction,
    Parser,
    ParserException,
    Reducer,
    ReducerException,
    UnaryFunction,
)


class CalculatorFormula(QtWidgets.QTextEdit):
    """Input for the calculator.

    Parsers input using a `:class:pewpew.lib.pratt.Parser` and
    colors input red when invalid. Implements completion when `completer` is set.
    """

    def __init__(
        self,
        text: str,
        variables: List[str],
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(text, parent)
        self.base_color = self.palette().color(QtGui.QPalette.Base)

        self.completer: QtWidgets.QCompleter = None

        self.textChanged.connect(self.calculate)

        self.parser = Parser(variables)
        self.expr = ""

    def calculate(self) -> None:
        try:
            self.expr = self.parser.parse(self.toPlainText())
        except ParserException:
            self.expr = ""
        self.revalidate()

    def hasAcceptableInput(self) -> bool:
        return self.expr != ""

    def revalidate(self) -> None:
        self.setValid(self.hasAcceptableInput())

    def setValid(self, valid: bool) -> None:
        palette = self.palette()
        if valid:
            color = self.base_color
        else:
            color = QtGui.QColor.fromRgb(255, 172, 172)
        palette.setColor(QtGui.QPalette.Base, color)
        self.setPalette(palette)

    def setCompleter(self, completer: QtWidgets.QCompleter) -> None:
        """Set the completer used."""
        if self.completer is not None:
            self.completer.disconnect(self)

        self.completer = completer
        self.completer.setWidget(self)
        self.completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self.completer.activated.connect(self.insertCompletion)

    def insertCompletion(self, completion: str) -> None:
        tc = self.textCursor()
        for _ in range(len(self.completer.completionPrefix())):
            tc.deletePreviousChar()
        tc.insertText(completion)
        self.setTextCursor(tc)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if self.completer is not None and self.completer.popup().isVisible():
            if event.key() in [  # Ignore keys when popup is present
                QtCore.Qt.Key_Enter,
                QtCore.Qt.Key_Return,
                QtCore.Qt.Key_Escape,
                QtCore.Qt.Key_Tab,
                QtCore.Qt.Key_Down,
                QtCore.Qt.Key_Up,
            ]:
                event.ignore()
                return

        super().keyPressEvent(event)

        eow = "~!@#$%^&*()+{}|:\"<>?,./;'[]\\-="
        tc = self.textCursor()
        tc.select(QtGui.QTextCursor.WordUnderCursor)
        prefix = tc.selectedText()
        if prefix != self.completer.completionPrefix():
            self.completer.setCompletionPrefix(prefix)
            self.completer.popup().setCurrentIndex(
                self.completer.completionModel().index(0, 0)
            )

        if (
            len(prefix) < 2
            or event.text() == ""
            or event.text()[-1] in eow
            or prefix == self.completer.currentCompletion()
        ):
            self.completer.popup().hide()
        else:
            rect = self.cursorRect()
            rect.setWidth(
                self.completer.popup().sizeHintForColumn(0)
                + self.completer.popup().verticalScrollBar().sizeHint().width()
            )
            self.completer.complete(rect)


class CalculatorExprList(QtWidgets.QListWidget):
    exprRemoved = QtCore.Signal(str)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.matches(QtGui.QKeySequence.StandardKey.Backspace) or event.matches(
            QtGui.QKeySequence.StandardKey.Delete
        ):
            for index in self.selectedIndexes():
                item = self.takeItem(index.row())
                self.exprRemoved.emit(item.text())
                del item

        super().keyPressEvent(event)


class CalculatorDialog(QtWidgets.QDialog):
    """Calculator for element data operations."""

    functions = {
        "abs": (
            (UnaryFunction("abs"), "(<x>)", "The absolute value of <x>."),
            (np.abs, 1),
        ),
        "mean": (
            (UnaryFunction("mean"), "(<x>)", "Returns the mean of <x>."),
            (np.nanmean, 1),
        ),
        "median": (
            (
                UnaryFunction("median"),
                "(<x>)",
                "Returns the median of <x>.",
            ),
            (np.nanmedian, 1),
        ),
        "nantonum": (
            (UnaryFunction("nantonum"), "(<x>)", "Sets nan values to 0."),
            (np.nan_to_num, 1),
        ),
        "percentile": (
            (
                BinaryFunction("percentile"),
                "(<x>, <percent>)",
                "Returns the <percent> percentile of <x>.",
            ),
            (np.nanpercentile, 2),
        ),
    }

    current_expressions: Dict[str, str] = {}

    def __init__(
        self,
        sample: SampleWidget,
        reference: ReferenceWidget,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Calculator")

        self.sample = sample
        self.reference = reference

        self.output = QtWidgets.QLineEdit("Result")
        self.output.setEnabled(False)

        self.combo_element = QtWidgets.QComboBox()
        self.combo_element.activated.connect(self.insertVariable)

        functions = [k + v[0][1] for k, v in self.functions.items()]
        tooltips = [v[0][2] for v in self.functions.values()]
        self.combo_function = QtWidgets.QComboBox()
        self.combo_function.addItem("Functions")
        self.combo_function.addItems(functions)
        for i in range(0, len(tooltips)):
            self.combo_function.setItemData(i + 1, tooltips[i], QtCore.Qt.ToolTipRole)
        self.combo_function.activated.connect(self.insertFunction)

        self.reducer = Reducer({})
        self.formula = CalculatorFormula("", variables=[])
        self.formula.textChanged.connect(self.completeChanged)
        self.formula.textChanged.connect(self.refresh)

        self.reducer.operations.update({k: v[1] for k, v in self.functions.items()})
        self.formula.parser.nulls.update(
            {k: v[0][0] for k, v in self.functions.items()}
        )

        layout_combos = QtWidgets.QHBoxLayout()
        layout_combos.addWidget(self.combo_element)
        layout_combos.addWidget(self.combo_function)

        layout_controls = QtWidgets.QFormLayout()
        layout_controls.addRow("Insert:", layout_combos)
        layout_controls.addRow("Formula:", self.formula)
        layout_controls.addRow("Result:", self.output)

        self.expressions = CalculatorExprList()
        self.expressions.exprRemoved.connect(self.removeExpr)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        # CollapsableWidget needs layout to be defined
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        expr_layout = QtWidgets.QVBoxLayout()
        expr_layout.addWidget(self.expressions, 1)
        collapse = CollapsableWidget("Current Expressions", parent=self)
        collapse.area.setLayout(expr_layout)

        layout.addLayout(layout_controls, 1)
        layout.addWidget(collapse, 0)
        layout.addWidget(self.button_box, 0)

        self.initialise()  # refreshes

    def isComplete(self) -> bool:
        if not self.formula.hasAcceptableInput():
            return False
        if len(self.formula.expr.split(" ")) < 2:
            return False
        if self.formula.expr in CalculatorDialog.current_expressions.values():
            return False
        return True

    def completeChanged(self) -> None:
        complete = self.isComplete()
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            complete
        )

    @staticmethod
    def updateNames(names: Dict[str, str]) -> None:
        for old, new in names.items():
            if old in CalculatorDialog.current_expressions:
                CalculatorDialog.current_expressions[
                    new
                ] = CalculatorDialog.current_expressions.pop(old)

    def removeExpr(self, expr: str) -> None:
        name = next(
            k for k, v in CalculatorDialog.current_expressions.items() if v == expr
        )
        CalculatorDialog.current_expressions.pop(name)
        if name in self.sample.names:
            data = rfn.drop_fields(self.sample.responses, name)
            self.sample.loadData(data, self.sample.import_options)
        if name in self.reference.names:
            data = rfn.drop_fields(self.reference.responses, name)
            self.reference.loadData(data, self.reference.import_options)

    def accept(self) -> None:
        new_name = (
            "{"
            + self.formula.toPlainText().translate(str.maketrans("", "", " \n\t"))
            + "}"
        )

        CalculatorDialog.current_expressions[new_name] = self.formula.expr
        self.sample.loadData(self.sample.responses, self.sample.import_options)

        try:  # attempt to set response of new variable
            self.reducer.variables = {
                name: self.sample.io[name].response.baseValue()
                for name in self.sample.names
            }
            data = self.reducer.reduce(self.formula.expr)
            self.sample.io[new_name].response.setBaseValue(data)
        except ReducerException:
            pass

        if len(self.reference.names) > 0:
            self.reference.loadData(
                self.reference.responses, self.reference.import_options
            )
            try:  # attempt to set response of new variable
                self.reducer.variables = {
                    name: self.reference.io[name].response.baseValue()
                    for name in self.reference.names
                }
                data = self.reducer.reduce(self.formula.expr)
                self.reference.io[new_name].response.setBaseValue(data)
            except ReducerException:
                pass

        self.expressions.addItem(self.formula.expr)
        self.completeChanged()
        # super().accept()

    def initialise(self) -> None:
        self.combo_element.clear()
        self.combo_element.addItem("Elements")
        self.combo_element.addItems(self.sample.names)

        self.formula.parser.variables = list(self.sample.names)
        self.formula.setCompleter(
            QtWidgets.QCompleter(
                list(self.formula.parser.variables)
                + [k + "(" for k in self.functions.keys()]
            )
        )
        self.formula.valid = True
        self.formula.setText(self.sample.names[0])  # refreshes

        self.expressions.clear()
        for name, expr in self.current_expressions.items():
            self.expressions.addItem(expr)

    def insertFunction(self, index: int) -> None:
        if index == 0:
            return
        function = self.combo_function.itemText(index)
        function = function[: function.find("(") + 1]
        self.formula.insertPlainText(function)
        self.combo_function.setCurrentIndex(0)
        self.formula.setFocus()

    def insertVariable(self, index: int) -> None:
        if index == 0:
            return
        self.formula.insertPlainText(self.combo_element.itemText(index))
        self.combo_element.setCurrentIndex(0)
        self.formula.setFocus()

    def refresh(self) -> None:
        self.reducer.variables = {
            name: self.sample.responses[name] for name in self.sample.names
        }
        try:
            result = self.reducer.reduce(self.formula.expr)
            if np.isscalar(result):
                self.output.setText(f"{result:.6g}")  # type: ignore
            elif isinstance(result, np.ndarray):
                self.output.setText(f"Events: {result.size}, mean: {result.mean():.6g}")
        except (ReducerException, ValueError) as e:
            self.output.setText(str(e))

    @staticmethod
    def reduceForData(data: np.ndarray) -> np.ndarray:
        reducer = Reducer(variables={name: data[name] for name in data.dtype.names})

        for name, expr in CalculatorDialog.current_expressions.items():
            if name in data.dtype.names:
                continue  # already calculated
            try:
                new_data = reducer.reduce(expr)
                data = rfn.append_fields(data, name, new_data, usemask=False)
            except ReducerException:
                pass

        return data
