from spcal.gui.util import create_action
from PySide6 import QtCore, QtGui, QtWidgets


def color_button(color: QtGui.QColor) -> QtWidgets.QToolButton:
    button = QtWidgets.QToolButton()
    button.setText("Color")
    button.setIcon(QtGui.QIcon.fromTheme("color-picker"))
    button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)

    effect = QtWidgets.QGraphicsColorizeEffect(button)
    effect.setColor(color)
    button.setGraphicsEffect(effect)

    def set_color():
        effect = button.graphicsEffect()
        if not isinstance(effect, QtWidgets.QGraphicsColorizeEffect):
            raise ValueError("effect has no color")
        color = QtWidgets.QColorDialog.getColor(effect.color())
        if color.isValid():
            effect.setColor(color)

    button.pressed.connect(set_color)

    return button


class ColorDialog(QtWidgets.QDialog):
    colorsSelected = QtCore.Signal(list)

    def __init__(
        self, colors: list[QtGui.QColor], parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent)

        gbox = QtWidgets.QGroupBox("Custom colors")
        self.grid = QtWidgets.QGridLayout()
        gbox.setLayout(self.grid)

        self.action_add = create_action(
            "list-add", "Add Color", "Add a new custom color.", self.addColor
        )
        self.action_remove = create_action(
            "list-remove",
            "Remove Color",
            "Removethe last custom color.",
            self.removeColor,
        )

        self.button_add = QtWidgets.QToolButton()
        self.button_add.setDefaultAction(self.action_add)
        self.button_remove = QtWidgets.QToolButton()
        self.button_remove.setDefaultAction(self.action_remove)

        layout_buttons = QtWidgets.QHBoxLayout()
        layout_buttons.addStretch(1)
        layout_buttons.addWidget(self.button_add, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        layout_buttons.addWidget(
            self.button_remove, 0, QtCore.Qt.AlignmentFlag.AlignRight
        )

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(gbox, 1)
        layout.addLayout(layout_buttons)
        layout.addWidget(self.button_box, 0)

        self.setColors(colors)

        self.setLayout(layout)

    def colors(self) -> list[QtGui.QColor]:
        colors = []
        for i in range(self.grid.count()):
            item = self.grid.itemAt(i)
            assert isinstance(item, QtWidgets.QWidgetItem)
            widget = item.widget()
            assert isinstance(widget, QtWidgets.QToolButton)
            effect = widget.graphicsEffect()
            assert isinstance(effect, QtWidgets.QGraphicsColorizeEffect)
            colors.append(effect.color())
        return colors

    def addColor(self):
        button = color_button(QtGui.QColor(0, 0, 0))
        i = self.grid.count()
        self.grid.addWidget(button, i // 8, i % 8)

    def removeColor(self):
        if self.grid.count() > 0:
            item = self.grid.itemAt(self.grid.count() - 1)
            if not isinstance(item, QtWidgets.QWidgetItem):
                return
            widget = item.widget()
            if widget is not None:
                widget.hide()
            self.grid.removeItem(item)
            del item

    def setColors(self, colors: list[QtGui.QColor]):
        for i, color in enumerate(colors):
            row = i // 8
            col = i % 8

            button = color_button(color)

            self.grid.addWidget(button, row, col)

    def accept(self):
        self.colorsSelected.emit(self.colors())
        super().accept()
