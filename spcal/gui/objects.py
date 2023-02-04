from typing import Tuple

from PySide6 import QtCore, QtGui, QtWidgets


class DoublePrecisionDelegate(QtWidgets.QStyledItemDelegate):
    """Delegate to display items with a certain number of decimals.

    Item inputs are also validated using a QDoubleValidator.
    """

    def __init__(self, decimals: int, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.decimals = decimals

    def createEditor(
        self,
        parent: QtWidgets.QWidget,
        option: QtWidgets.QStyleOptionViewItem,
        index: int,
    ) -> QtWidgets.QWidget:  # pragma: no cover
        lineedit = QtWidgets.QLineEdit(parent)
        lineedit.setValidator(QtGui.QDoubleValidator())
        return lineedit

    def displayText(self, value: str, locale: str) -> str:
        """Attempts to display text as a float with 'decimal' places."""
        try:
            num = float(value)
            return f"{num:.{self.decimals}f}"
        except (TypeError, ValueError):
            return str(super().displayText(value, locale))


class DragDropRedirectFilter(QtCore.QObject):
    """Redirects drag/drop events to parent."""

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.DragEnter:
            self.parent().dragEnterEvent(event)
            return True
        elif event.type() == QtCore.QEvent.DragLeave:
            self.parent().dragLeaveEvent(event)
            return True
        elif event.type() == QtCore.QEvent.DragMove:
            self.parent().dragMoveEvent(event)
            return True
        elif event.type() == QtCore.QEvent.Drop:
            self.parent().dropEvent(event)
            return True
        return bool(super().eventFilter(obj, event))


class KeepMenuOpenFilter(QtCore.QObject):
    """Keeps menu open when action is triggered."""

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() in [
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QEvent.Type.MouseButtonDblClick,
        ] and isinstance(obj, QtWidgets.QMenu):
            if obj.activeAction() and not obj.activeAction().menu():
                obj.activeAction().trigger()
                return True
        elif event.type() in [QtCore.QEvent.Type.MouseButtonRelease]:
            return True
        return super().eventFilter(obj, event)


class DoubleOrPercentValidator(QtGui.QDoubleValidator):
    """QDoubleValidator that accepts inputs as a percent.

    Inputs that end with '%' are treated as a percentage input.

    Args:
        bottom: decimal lower bound
        top: decimal upper bound
        decimals: number of decimals allowed
        percent_bottom: percent lower bound
        percent_top: percent upper bound
        parent: parent object
    """

    def __init__(
        self,
        bottom: float,
        top: float,
        decimals: int = 4,
        percent_bottom: float = 0.0,
        percent_top: float = 100.0,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(bottom, top, decimals, parent)
        self._bottom = bottom
        self._top = top
        self.percent_bottom = percent_bottom
        self.percent_top = percent_top

    def validate(self, input: str, pos: int) -> Tuple[QtGui.QValidator.State, str, int]:
        # Treat as percent
        if "%" in input:
            if not input.endswith("%") or input.count("%") > 1:
                return (QtGui.QValidator.Invalid, input, pos)
            self.setRange(self.percent_bottom, self.percent_top, self.decimals())
            return (super().validate(input.rstrip("%"), pos)[0], input, pos)
        # Treat as double
        self.setRange(self._bottom, self._top, self.decimals())
        return super().validate(input, pos)
