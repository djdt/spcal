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

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Close,
        )

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.spinbox_size_x)
        layout.addWidget(self.spinbox_size_y)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def accept(self) -> None:
        self.render()
        super().accept()

    def render(self):
        pixmap = QtGui.QImage(
            self.spinbox_size_x.value(),
            self.spinbox_size_y.value(),
            QtGui.QImage.Format.Format_ARGB32,
        )
        pixmap.fill(QtGui.QColor(0, 0, 0, 0))
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        source = self.graph.viewRect()

        self.scene.render(
            painter,
            QtCore.QRectF(pixmap.rect()),
            self.graph.viewRect(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio
        )
        painter.end()
        pixmap.save("/home/tom/Downloads/out.png")
        # painter.drawSc
        # for item in self.graph.items():
        #     if isinstance(item, pyqtgraph.PlotCurveItem):
        #         pen: QtGui.QPen = item.opts["pen"]
        #         pen.setWidthF(2.0)
