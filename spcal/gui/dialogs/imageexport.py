from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.graphs.base import SinglePlotGraphicsView


class ImageExportDialog(QtWidgets.QDialog):
    def __init__(
        self, graph: SinglePlotGraphicsView, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent)
        self.graph = graph
        size = graph.viewport().rect()

        self.spinbox_size_x = QtWidgets.QSpinBox()
        self.spinbox_size_x.setRange(100, 10000)
        self.spinbox_size_x.setValue(size.width())

        self.spinbox_size_y = QtWidgets.QSpinBox()
        self.spinbox_size_y.setRange(100, 10000)
        self.spinbox_size_y.setValue(size.height())

        self.spinbox_dpi = QtWidgets.QSpinBox()
        self.spinbox_dpi.setRange(96, 1200)
        self.spinbox_dpi.setValue(96)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Close,
        )

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout_size = QtWidgets.QHBoxLayout()
        layout_size.addWidget(self.spinbox_size_x, 1)
        layout_size.addWidget(QtWidgets.QLabel("x"), 0)
        layout_size.addWidget(self.spinbox_size_y, 1)

        layout_form = QtWidgets.QFormLayout()
        layout_form.addRow("Size:", layout_size)
        layout_form.addRow("DPI:", self.spinbox_dpi)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(layout_form)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def accept(self) -> None:
        self.prepareForRender()
        self.timer = QtCore.QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.render)
        self.timer.start(100)

    def prepareForRender(self) -> None:
        self.original_size = self.graph.size()
        self.original_font = QtGui.QFont(self.graph.font)
        self.image = QtGui.QImage(
            self.spinbox_size_x.value(),
            self.spinbox_size_y.value(),
            QtGui.QImage.Format.Format_ARGB32,
        )
        self.image.fill(QtGui.QColor(0, 0, 0, 0))

        resized_font = QtGui.QFont(self.original_font)
        resized_font.setPointSizeF(
            self.original_font.pointSizeF() / 96.0 * self.spinbox_dpi.value()
        )
        self.graph.resize(self.image.size())
        self.graph.setFont(resized_font)

    def postRender(self) -> None:
        self.graph.resize(self.original_size)
        self.graph.setFont(self.original_font)

    def render(self):
        painter = QtGui.QPainter(self.image)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)

        self.graph.scene().prepareForPaint()
        self.graph.scene().render(
            painter,
            QtCore.QRectF(self.image.rect()),
            self.graph.viewRect(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        )
        painter.end()
        self.image.save("/home/tom/Downloads/out.png")

        self.post_timer = QtCore.QTimer()
        self.post_timer.setSingleShot(True)
        self.post_timer.timeout.connect(self.postRender)
        self.post_timer.start(100)

        super().accept()
