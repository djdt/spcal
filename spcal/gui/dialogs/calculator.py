import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from spcal.isotope import SPCalIsotope, SPCalIsotopeBase, SPCalIsotopeExpression
from spcal.gui.modelviews.isotope import IsotopeComboBox

from spcal.pratt import (
    BinaryFunction,
    Parser,
    ParserException,
    UnaryFunction,
)


class CalculatorFormula(QtWidgets.QTextEdit):
    """Input for the calculator.

    Parsers input using a `:class:pewpew.lib.pratt.Parser` and
    colors input red when invalid. Implements completion when `completer` is set.
    """

    expressionSelected = QtCore.Signal(str)

    def __init__(
        self,
        text: str,
        variables: list[str],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(text, parent)
        self.base_color = self.palette().color(QtGui.QPalette.ColorRole.Base)

        self.completer: QtWidgets.QCompleter | None = None

        self.textChanged.connect(self.calculate)

        self.parser = Parser(variables)
        self.expr = ""

    def sizeHint(self) -> QtCore.QSize:
        size = super().sizeHint()
        return QtCore.QSize(size.width(), self.fontMetrics().height() * 5)

    def calculate(self):
        try:
            self.expr = self.parser.parse(self.toPlainText())
        except ParserException:
            self.expr = ""
        self.revalidate()

    def hasAcceptableInput(self) -> bool:
        return self.expr != "" and any(x in self.expr for x in self.parser.variables)

    def revalidate(self):
        self.setValid(self.hasAcceptableInput())

    def setValid(self, valid: bool):
        palette = self.palette()
        if valid:
            color = self.base_color
        else:
            color = QtGui.QColor.fromRgb(255, 172, 172)
        palette.setColor(QtGui.QPalette.ColorRole.Base, color)
        self.setPalette(palette)

    def setCompleter(self, completer: QtWidgets.QCompleter):
        """Set the completer used."""
        if self.completer is not None:
            self.completer.disconnect(self)

        self.completer = completer
        self.completer.setWidget(self)
        self.completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        self.completer.activated.connect(self.insertCompletion)

    def insertCompletion(self, completion: str):
        if self.completer is None:
            return
        tc = self.textCursor()
        for _ in range(len(self.completer.completionPrefix())):
            tc.deletePreviousChar()
        tc.insertText(completion)
        self.setTextCursor(tc)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if self.completer is None:
            raise ValueError("no completor, did you not set it?")

        if self.completer.popup().isVisible():
            if event.key() in [  # Ignore keys when popup is present
                QtCore.Qt.Key.Key_Enter,
                QtCore.Qt.Key.Key_Return,
                QtCore.Qt.Key.Key_Escape,
                QtCore.Qt.Key.Key_Tab,
                QtCore.Qt.Key.Key_Down,
                QtCore.Qt.Key.Key_Up,
            ]:
                event.ignore()
                return

        if event.key() in [QtCore.Qt.Key.Key_Enter, QtCore.Qt.Key.Key_Return]:
            if self.hasAcceptableInput():
                self.sumbitExpression()
                event.accept()
            else:
                event.ignore()
            return
        else:
            super().keyPressEvent(event)

        eow = "~!@#$%^&*()+{}|:\"<>?,./;'[]\\-="
        tc = self.textCursor()
        tc.select(QtGui.QTextCursor.SelectionType.WordUnderCursor)
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

    def sumbitExpression(self):
        self.expressionSelected.emit(self.expr)
        self.clear()


class CalculatorExprList(QtWidgets.QListWidget):
    expressionRemoved = QtCore.Signal(SPCalIsotopeExpression)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.matches(QtGui.QKeySequence.StandardKey.Backspace) or event.matches(
            QtGui.QKeySequence.StandardKey.Delete
        ):
            for index in self.selectedIndexes():
                item = self.takeItem(index.row())
                self.expressionRemoved.emit(item.data(QtCore.Qt.ItemDataRole.UserRole))
                del item

        super().keyPressEvent(event)

    def addExpression(self, expr: SPCalIsotopeExpression):
        item = QtWidgets.QListWidgetItem()
        item.setData(QtCore.Qt.ItemDataRole.UserRole, expr)
        item.setText(" ".join(str(x) for x in expr.tokens))
        self.addItem(item)

    def expressions(self) -> list[SPCalIsotopeExpression]:
        exprs = []
        for i in range(self.count()):
            expr = self.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
            exprs.append(expr)
        return exprs


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

    expressionsChanged = QtCore.Signal(list)

    def __init__(
        self,
        isotopes: list[SPCalIsotopeBase],
        expressions: list[SPCalIsotopeExpression],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Calculator")

        self.isotope_table: dict[str, SPCalIsotope] = {
            str(isotope): isotope
            for isotope in isotopes
            if isinstance(isotope, SPCalIsotope)
        }
        self.old_expressions = expressions

        self.combo_isotope = IsotopeComboBox()
        self.combo_isotope.addItem("Isotopes")
        self.combo_isotope.addIsotopes(isotopes)
        self.combo_isotope.activated.connect(self.insertIsotope)

        functions = [k + v[0][1] for k, v in self.functions.items()]
        tooltips = [v[0][2] for v in self.functions.values()]
        self.combo_function = QtWidgets.QComboBox()
        self.combo_function.addItem("Functions")
        self.combo_function.addItems(functions)
        for i in range(0, len(tooltips)):
            self.combo_function.setItemData(
                i + 1, tooltips[i], QtCore.Qt.ItemDataRole.ToolTipRole
            )
        self.combo_function.activated.connect(self.insertFunction)

        self.formula = CalculatorFormula("", variables=list(self.isotope_table.keys()))
        self.formula.parser.nulls.update(
            {k: v[0][0] for k, v in self.functions.items()}
        )
        self.formula.setCompleter(
            QtWidgets.QCompleter(
                list(self.isotope_table.keys())
                + [k + "(" for k in self.functions.keys()]
            )
        )
        self.button_add = QtWidgets.QToolButton()
        self.button_add.setIcon(QtGui.QIcon.fromTheme("list-add"))
        self.button_add.setText("Add Expression")
        self.button_add.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.button_add.pressed.connect(self.formula.sumbitExpression)

        self.formula.expressionSelected.connect(self.reformAndAddExpression)
        self.formula.textChanged.connect(self.completeChanged)

        self.expressions = CalculatorExprList()
        for expr in expressions:
            self.expressions.addExpression(expr)

        layout_combos = QtWidgets.QHBoxLayout()
        layout_combos.addWidget(self.combo_isotope)
        layout_combos.addWidget(self.combo_function)

        layout_formula = QtWidgets.QVBoxLayout()
        layout_formula.addWidget(self.formula, 1)
        layout_formula.addWidget(self.button_add, 0, QtCore.Qt.AlignmentFlag.AlignRight)

        layout_controls = QtWidgets.QFormLayout()
        layout_controls.addRow("Insert:", layout_combos)
        layout_controls.addRow("Formula:", layout_formula)
        layout_controls.addRow("Current:", self.expressions)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(layout_controls, 1)
        layout.addWidget(self.button_box, 0)
        self.setLayout(layout)

    def reformAndAddExpression(self, expr_string: str):
        tokens = expr_string.split(" ")
        expr = SPCalIsotopeExpression(
            name=f"({expr_string})",
            tokens=tuple([self.isotope_table.get(token, token) for token in tokens]),
        )
        self.expressions.addExpression(expr)

    # def isComplete(self) -> bool:
    #     return self.formula.hasAcceptableInput()

    def completeChanged(self):
        formula_ready = self.formula.hasAcceptableInput()
        self.button_add.setEnabled(formula_ready)

    def insertIsotope(self, index: int):
        if index == 0:
            return
        isotope = self.combo_isotope.isotope(index)
        self.combo_isotope.setCurrentIndex(0)
        self.formula.insertPlainText(str(isotope))
        self.formula.setFocus()

    def accept(self):
        if self.formula.hasAcceptableInput():
            self.formula.sumbitExpression()

        expressions = self.expressions.expressions()
        if set(expressions) != set(self.old_expressions):
            self.expressionsChanged.emit(expressions)
        super().accept()

    def insertFunction(self, index: int):
        if index == 0:
            return
        function = self.combo_function.itemText(index)
        function = function[: function.find("(") + 1]
        self.formula.insertPlainText(function)
        self.combo_function.setCurrentIndex(0)
        self.formula.setFocus()
