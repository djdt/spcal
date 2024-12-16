import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.graphs.base import SinglePlotGraphicsView


class ImageExportDialog(QtWidgets.QDialog):
    def __init__(
        self, graph: pyqtgraph.GraphicsView, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent)
        self.graph = graph
        self.scene = graph.scene()

        size = self.graph.viewport().rect()

        self.spinbox_size_x = QtWidgets.QSpinBox()
        self.spinbox_size_x.setRange(100, 10000)
        self.spinbox_size_x.setValue(size.width())

        self.spinbox_size_y = QtWidgets.QSpinBox()
        self.spinbox_size_y.setRange(100, 10000)
        self.spinbox_size_y.setValue(size.height())

        self.spinbox_dpi = QtWidgets.QSpinBox()
        # 96, 10000)
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
        self.render()
        super().accept()

    def scaleFonts(self, points: float, dpi: float) -> None:
        size = points / 96.0 * dpi
        for item in self.scene.items():
            if isinstance(item, pyqtgraph.AxisItem):
                item.setStyle(
                    tickLength=int(size),
                    tickTextHeight=int(size),
                    tickFont=QtGui.QFont("sans", size)
                )
            elif isinstance(item, QtWidgets.QGraphicsTextItem):
                item.setFont(QtGui.QFont("sans", size))
            elif isinstance(item, pyqtgraph.LabelItem):
                item.setText(item.text, family="sans", size=f"{size}pt")
                item.item.setFont(QtGui.QFont("sans", size))
                item.item.setPlainText(item.text)

    def render(self):
        original_size = self.graph.size()
        image = QtGui.QImage(
            self.spinbox_size_x.value(),
            self.spinbox_size_y.value(),
            QtGui.QImage.Format.Format_ARGB32,
        )
        image.fill(QtGui.QColor(0, 0, 0, 0))
        painter = QtGui.QPainter(image)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)

        self.graph.resize(image.size())
        self.scaleFonts(10.0, 300.0)

        # Set font prior, in program is required
        # We can then scale everything to the approriate dpi (assume 96 standard)

        # print(image.dotsPerMeterY() * 0.0254)
        # image.setDotsPerMeterY(200.0 / 0.0254)
        # image.setDotsPerMeterX(200.0 / 0.0254)
        #


        self.scene.render(
            painter,
            QtCore.QRectF(image.rect()),
            self.graph.viewRect(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        )
        painter.end()
        image.save("/home/tom/Downloads/out.png")
        self.graph.resize(original_size)
