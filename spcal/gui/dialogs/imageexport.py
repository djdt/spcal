from PySide6 import QtCore, QtGui, QtWidgets
import numpy as np


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


class ImageExportDialog(QtWidgets.QDialog):
    imageOptionsSelected = QtCore.Signal(
        QtCore.QSize, int, QtGui.QColor, QtGui.QFont, QtGui.QColor, QtGui.QColor
    )

    def __init__(
        self,
        size: QtCore.QSize | None = None,
        dpi: int = 96,
        color: QtGui.QColor | QtCore.Qt.GlobalColor | None = None,
        font: QtGui.QFont | None = None,
        font_color: QtGui.QColor | QtCore.Qt.GlobalColor | None = None,
        background_color: QtGui.QColor | QtCore.Qt.GlobalColor | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Image Export")

        if size is None:
            size = QtCore.QSize(800, 600)

        if font is None:
            font = QtGui.QFont()

        if color is None:
            color = QtGui.QColor(QtCore.Qt.GlobalColor.red)
        elif isinstance(color, QtCore.Qt.GlobalColor):
            color = QtGui.QColor(color)

        if font_color is None:
            font_color = QtGui.QColor(QtCore.Qt.GlobalColor.black)
        elif isinstance(font_color, QtCore.Qt.GlobalColor):
            font_color = QtGui.QColor(font_color)

        if background_color is None:
            background_color = QtGui.QColor(QtCore.Qt.GlobalColor.white)
        elif isinstance(background_color, QtCore.Qt.GlobalColor):
            background_color = QtGui.QColor(background_color)

        self.size_x = QtWidgets.QDoubleSpinBox()
        self.size_x.setRange(100, 10000)
        self.size_x.setValue(size.width())

        self.size_y = QtWidgets.QDoubleSpinBox()
        self.size_y.setRange(100, 10000)
        self.size_y.setValue(size.height())

        self.size_unit = QtWidgets.QComboBox()
        self.size_unit.addItems(["pixel", "inch", "mm"])
        self.size_unit.currentTextChanged.connect(self.sizeUnitChanged)
        self._previous_unit = "pixel"

        size_layout = QtWidgets.QHBoxLayout()
        size_layout.addWidget(self.size_x, 1)
        size_layout.addWidget(QtWidgets.QLabel("×"), 0)
        size_layout.addWidget(self.size_y, 1)
        size_layout.addWidget(self.size_unit, 0)

        self.dpi = QtWidgets.QSpinBox()
        self.dpi.setRange(96, 1200)
        self.dpi.setValue(dpi)

        self.button_color = color_button(color)

        self.button_font = QtWidgets.QPushButton(f"{font.family()}: {font.pointSize()}")
        self.button_font.setIcon(QtGui.QIcon.fromTheme("gtk-select-font"))
        self.button_font.pressed.connect(self.dialogFont)

        self.selected_font = font

        self.button_font_color = color_button(font_color)

        self.button_background_color = color_button(background_color)

        self.check_transparent = QtWidgets.QCheckBox("Transparent")
        self.check_transparent.setChecked(background_color.alpha() == 0)

        background_layout = QtWidgets.QHBoxLayout()
        background_layout.addWidget(self.button_background_color)
        background_layout.addWidget(self.check_transparent)

        gbox = QtWidgets.QGroupBox("Image options")
        gbox_layout = QtWidgets.QFormLayout()
        gbox_layout.addRow("Size", size_layout)
        gbox_layout.addRow("DPI", self.dpi)
        gbox_layout.addRow("Color", self.button_color)
        gbox_layout.addRow("Background", background_layout)
        gbox.setLayout(gbox_layout)

        gbox_font = QtWidgets.QGroupBox("Font options")
        gbox_font_layout = QtWidgets.QFormLayout()
        gbox_font_layout.addRow("Font", self.button_font)
        gbox_font_layout.addRow("Color", self.button_font_color)
        gbox_font.setLayout(gbox_font_layout)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Close,
        )

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QtWidgets.QGridLayout()

        layout.addWidget(gbox, 0, 0, 1, 1)
        layout.addWidget(gbox_font, 1, 0, 1, 2)
        layout.addWidget(self.button_box, 2, 0, 1, 2)
        self.setLayout(layout)

    def convertSizeUnit(self, unit_from: str, unit_to: str, value: float) -> float:
        if unit_from == unit_to:
            return value

        if unit_from == "inch":
            value = value * self.dpi.value()
        elif unit_from == "mm":
            value = value * self.dpi.value() / 25.4

        if unit_to == "inch":
            value = value / self.dpi.value()
        elif unit_to == "mm":
            value = value / self.dpi.value() * 25.4

        return value

    def sizeUnitChanged(self, unit: str):
        size_x = self.convertSizeUnit(self._previous_unit, unit, self.size_x.value())
        size_y = self.convertSizeUnit(self._previous_unit, unit, self.size_y.value())

        for sb in [self.size_x, self.size_y]:
            if unit == "inch":
                sb.setRange(2, 100)
                sb.setDecimals(2)
            elif unit == "mm":
                sb.setRange(50, 2000)
                sb.setDecimals(0)
            else:  # pixel
                sb.setRange(100, 10000)
                sb.setDecimals(0)

        self.size_x.setValue(size_x)
        self.size_y.setValue(size_y)
        self._previous_unit = unit

    def dialogFont(self):
        ok, font = QtWidgets.QFontDialog.getFont(
            self.button_font.font(), self, "Graph Font"
        )
        if ok:
            self.button_font.setText(f"{font.family()}: {font.pointSize()}")
            self.selected_font = font

    def imageSize(self) -> QtCore.QSize:
        x = self.convertSizeUnit(
            self.size_unit.currentText(), "pixel", self.size_x.value()
        )
        y = self.convertSizeUnit(
            self.size_unit.currentText(), "pixel", self.size_y.value()
        )
        return QtCore.QSize(int(x), int(y))

    def color(self) -> QtGui.QColor:
        effect = self.button_color.graphicsEffect()
        assert isinstance(effect, QtWidgets.QGraphicsColorizeEffect)
        return effect.color()

    def backgroundColor(self) -> QtGui.QColor:
        effect = self.button_background_color.graphicsEffect()
        assert isinstance(effect, QtWidgets.QGraphicsColorizeEffect)
        color = effect.color()
        if self.check_transparent.isChecked():
            color.setAlpha(0)
        return color

    def fontColor(self) -> QtGui.QColor:
        effect = self.button_font_color.graphicsEffect()
        assert isinstance(effect, QtWidgets.QGraphicsColorizeEffect)
        return effect.color()

    def imageOptions(
        self,
    ) -> tuple[
        QtCore.QSize, int, QtGui.QColor, QtGui.QFont, QtGui.QColor, QtGui.QColor
    ]:
        return (
            self.imageSize(),
            self.dpi.value(),
            self.color(),
            self.selected_font,
            self.fontColor(),
            self.backgroundColor(),
        )

    def accept(self):
        self.imageOptionsSelected.emit(*self.imageOptions())
        super().accept()
