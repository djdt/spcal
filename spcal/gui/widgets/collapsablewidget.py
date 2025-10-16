from PySide6 import QtCore, QtWidgets


class CollapsableWidget(QtWidgets.QWidget):
    """A widget that can be hidden.

    Clicking on the widget will show and resize it.

    Args:
        title: hide/show button text
        parent: parent widget
    """

    def __init__(self, title: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        self.button = QtWidgets.QToolButton()
        self.button.setArrowType(QtCore.Qt.RightArrow)
        self.button.setAutoRaise(True)
        self.button.setCheckable(True)
        self.button.setChecked(False)
        self.button.setText(title)
        self.button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.button.toggled.connect(self.setCollapsed)

        self.widget: QtWidgets.QWidget | None = None

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.button, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def setWidget(self, widget: QtWidgets.QWidget) -> None:
        if self.widget is not None:
            self.layout().removeWidget(widget)

        self.widget = widget
        self.layout().addWidget(widget, 1)
        self.widget.setVisible(not self.isCollapsed())

    def isCollapsed(self) -> bool:
        return self.button.arrowType() != QtCore.Qt.DownArrow

    def setCollapsed(self, collapsed: bool) -> None:  # pragma: no cover, trivial
        self.button.setArrowType(
            QtCore.Qt.DownArrow if collapsed else QtCore.Qt.RightArrow
        )
        if self.widget is not None:
            self.widget.setVisible(collapsed)
