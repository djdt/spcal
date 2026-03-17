from PySide6 import QtCore, QtGui, QtWidgets


def color_icon(color: QtGui.QColor) -> QtGui.QIcon:
    pixmap = QtGui.QPixmap(32, 32)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.black, 2.0))
    painter.setBrush(QtGui.QBrush(color))
    painter.drawEllipse(
        pixmap.rect().center(), pixmap.width() // 2 - 2, pixmap.height() // 2 - 2
    )
    return QtGui.QIcon(pixmap)


class ImageExportDialog(QtWidgets.QDialog):
    exportSettingsSelected = QtCore.Signal(QtCore.QSize, int, object)  # object for dict

    def __init__(
        self,
        size: QtCore.QSize | None = None,
        font: QtGui.QFont | None = None,
        dpi: int = 96,
        color: QtGui.QColor | None = None,
        font_color: QtGui.QColor | None = None,
        options: dict[str, bool] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Image Export")

        if size is None:
            size = QtCore.QSize(800, 600)

        if font is None:
            font = QtGui.QFont()

        if color is None:
            color = QtGui.QColor(QtCore.Qt.GlobalColor.black)

        if font_color is None:
            font_color = QtGui.QColor(QtCore.Qt.GlobalColor.white)

        self.spinbox_size_x = QtWidgets.QSpinBox()
        self.spinbox_size_x.setRange(100, 10000)
        self.spinbox_size_x.setValue(size.width())

        self.spinbox_size_y = QtWidgets.QSpinBox()
        self.spinbox_size_y.setRange(100, 10000)
        self.spinbox_size_y.setValue(size.height())

        size_layout = QtWidgets.QHBoxLayout()
        size_layout.addWidget(self.spinbox_size_x, 1)
        size_layout.addWidget(QtWidgets.QLabel("×"), 0)
        size_layout.addWidget(self.spinbox_size_y, 1)

        self.spinbox_dpi = QtWidgets.QSpinBox()
        self.spinbox_dpi.setRange(96, 1200)
        self.spinbox_dpi.setValue(dpi)

        self.button_font = QtWidgets.QFontComboBox()
        self.button_font.setCurrentFont(font)

        self.button_color = QtWidgets.QPushButton("Select Color...")
        self.button_color.setIcon(color_icon(color))
        self.button_color.pressed.connect(self.selectColor)

        self.button_font_color = QtWidgets.QPushButton("Select Color...")
        self.button_font_color.setIcon(color_icon(font_color))

        gbox = QtWidgets.QGroupBox("Image options")
        gbox_layout = QtWidgets.QFormLayout()
        gbox_layout.addRow("Size", size_layout)
        gbox_layout.addRow("Color", self.button_color)
        gbox.setLayout(gbox_layout)

        gbox_font = QtWidgets.QGroupBox("Font options")
        gbox_font_layout = QtWidgets.QFormLayout()
        gbox_font_layout.addRow("Font", self.button_font)
        gbox_font_layout.addRow("Color", self.button_font_color)
        gbox_font_layout.addRow("DPI", self.spinbox_dpi)
        gbox_font.setLayout(gbox_font_layout)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Close,
        )

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QtWidgets.QGridLayout()

        layout.addWidget(gbox, 0, 0, 1, 1)
        layout.addWidget(gbox_font, 0, 1, 1, 1)
        layout.addWidget(self.button_box, 1, 0, 1, 2)
        self.setLayout(layout)

    def selectColor(self, color: QtGui.QColor) -> QtGui.QColor | None:
        color = QtWidgets.QColorDialog.getColor(color, parent=self)
        if not color.isValid():
            return None
        return color

    def accept(self):
        super().accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication()
    dlg = ImageExportDialog()
    dlg.show()
    app.exec()
