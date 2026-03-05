from PySide6 import QtCore, QtWidgets


class CollapsableWidget(QtWidgets.QWidget):
    collapsed = QtCore.Signal(bool)
    """A widget that can be hidden.

    Clicking on the widget will show and resize it.

    Args:
        title: hide/show button text
        parent: parent widget
    """

    def __init__(self, title: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        self.button = QtWidgets.QToolButton()
        self.button.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.button.setAutoRaise(True)
        self.button.setCheckable(True)
        self.button.setChecked(False)
        self.button.setText(title)
        self.button.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.button.toggled.connect(self.setCollapsed)

        # self.widget: QtWidgets.QWidget | None = None
        self.area = QtWidgets.QScrollArea()
        self.area.hide()
        self.area.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        self.area.setFrameStyle(QtWidgets.QFrame.Shape.NoFrame)
        self.area.setWidgetResizable(True)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.button, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.area)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def setWidget(self, widget: QtWidgets.QWidget):
        self.area.setWidget(widget)

    def isCollapsed(self) -> bool:
        return self.button.arrowType() != QtCore.Qt.ArrowType.DownArrow

    def setCollapsed(self, collapsed: bool):  # pragma: no cover, trivial
        self.button.setArrowType(
            QtCore.Qt.ArrowType.DownArrow
            if collapsed
            else QtCore.Qt.ArrowType.RightArrow
        )
        self.area.setVisible(collapsed)
        self.collapsed.emit(collapsed)
