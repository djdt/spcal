from PySide6 import QtCore, QtWidgets


class CollapsableWidget(QtWidgets.QWidget):
    """A widget that can be hidden.

    Clicking on the widget will show and resize it.

    Args:
        title: hide/show button text
        parent: parent widget
    """

    def __init__(self, title: str, parent: QtWidgets.QWidget):
        super().__init__(parent)

        self.button = QtWidgets.QToolButton()
        self.button.setArrowType(QtCore.Qt.RightArrow)
        self.button.setAutoRaise(True)
        self.button.setCheckable(True)
        self.button.setChecked(False)
        self.button.setText(title)
        self.button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)

        self.line = QtWidgets.QFrame()
        self.line.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum
        )
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)

        self.area = QtWidgets.QWidget()

        self.button.toggled.connect(self.collapse)

        layout_line = QtWidgets.QHBoxLayout()
        layout_line.addWidget(self.button, 0, QtCore.Qt.AlignLeft)
        layout_line.addWidget(self.line, 1)
        layout_line.setAlignment(QtCore.Qt.AlignTop)

        layout = QtWidgets.QVBoxLayout()
        # layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(layout_line, 0)
        layout.addWidget(self.area, 1)
        self.setLayout(layout)

        self.area.hide()

        self.parent().layout().setSizeConstraint(QtWidgets.QLayout.SetFixedSize)

    def collapse(self, down: bool) -> None:  # pragma: no cover, trivial
        self.button.setArrowType(QtCore.Qt.DownArrow if down else QtCore.Qt.RightArrow)
        self.area.setVisible(down)
