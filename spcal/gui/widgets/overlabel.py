from PySide6 import QtCore, QtWidgets


class OverLabel(QtWidgets.QWidget):
    def __init__(
        self,
        widget: QtWidgets.QWidget,
        label: str,
        alignment: QtCore.Qt.AlignmentFlag = QtCore.Qt.AlignmentFlag.AlignRight
        | QtCore.Qt.AlignmentFlag.AlignVCenter,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        self.label = QtWidgets.QLabel(label)
        self.label.setAlignment(alignment)
        self.label.setIndent(10)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(widget, 0, 0, 1, 1)
        layout.addWidget(self.label, 0, 0, 1, 1, alignment)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def setText(self, label: str) -> None:
        self.label.setText(label)
