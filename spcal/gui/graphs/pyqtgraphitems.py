from typing import List

import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets


class PieChart(QtWidgets.QGraphicsObject, pyqtgraph.GraphicsItem):
    hovered = QtCore.Signal(int)

    def __init__(
        self,
        radius: float,
        values: List[float],
        brushes: List[QtGui.QBrush],
        pen: QtGui.QPen | None = None,
        labels: List[str] | None = None,
        # label_format: str = "{:.4g}",
        parent: QtWidgets.QGraphicsItem | None = None,
    ):
        """Pie is centered on item.pos()."""
        super().__init__(parent=parent)
        self.setAcceptHoverEvents(True)

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)

        self.pen = pen
        self.brushes = brushes

        self.radius = radius
        self.values = values
        self._paths: List[QtGui.QPainterPath] = []

        self.hovered_idx: int = -1

        self.labels: List[pyqtgraph.TextItem] = []
        # if labels is not None:
        #     assert len(labels) == len(values)
        #     angle = 0.0
        #     for label in labels:

        # self.label_format = label_format

    def name(self) -> str:
        return "pie"

    def setHoveredIdx(self, idx: int) -> None:
        if self.hovered_idx != idx:
            self.hovered_idx = idx
            self.update()

    def hoverEvent(self, event):
        hovered_idx = -1
        if not event.exit:
            for i, path in enumerate(self._paths):
                if path.contains(event.pos()):
                    hovered_idx = i
                    break

        if hovered_idx != self.hovered_idx:
            self.hovered.emit(hovered_idx)
            self.update()
        self.hovered_idx = hovered_idx

    @property
    def paths(self) -> List[QtGui.QPainterPath]:
        if len(self._paths) == 0:
            rect = self.boundingRect()
            total = np.sum(self.values)
            angle = 0.0
            for value in self.values:
                span = 360.0 * value / total
                path = QtGui.QPainterPath(QtCore.QPointF(0, 0))
                path.arcTo(rect, angle, span)
                self._paths.append(path)
                angle += span

        return self._paths

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ) -> None:
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(self.pen)
        for i, (path, brush) in enumerate(zip(self.paths, self.brushes)):
            if i == self.hovered_idx:
                brush = QtGui.QBrush(brush)
                brush.setColor(brush.color().lighter())
            painter.setBrush(brush)
            painter.drawPath(path)
        painter.restore()

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(
            -self.radius, -self.radius, self.radius * 2, self.radius * 2
        )

    def shape(self) -> QtGui.QPainterPath:
        path = QtGui.QPainterPath()
        path.addEllipse(self.boundingRect())
        return path

    def dataBounds(self, ax, frac, orthoRange=None):
        if self.pen.isCosmetic():
            pw = 0.0
        else:
            pw = self.pen.width() * 0.7072
        br = self.boundingRect()
        if ax == 0:
            return [br.left() - pw, br.right() + pw]
        else:
            return [br.top() - pw, br.bottom() + pw]

    def pixelPadding(self):
        if self.pen.isCosmetic():
            return max(1, self.pen.width()) * 0.7072
        else:
            return 0.0


if __name__ == "__main__":

    from spcal.gui.graphs import color_schemes

    app = QtWidgets.QApplication()

    view = pyqtgraph.PlotWidget(background="white")
    view.setAspectLocked(1.0)
    view.addLegend()
    item = pyqtgraph.BarGraphItem(
        x=[1, 2, 3], y=[4, 5, 6], width=1.0, y1=0.0, name="eggs"
    )
    view.addItem(item)

    item = PieChart(100.0, [1.0, 2.0, 3.0, 4.0], color_schemes["IBM Carbon"])
    item.setPos(0.0, 0.0)
    view.addItem(item)

    item2 = PieChart(100.0, [23.0, 12.0, 0.0, 32.1], color_schemes["IBM Carbon"])
    item2.setPos(300.0, 0.0)
    view.addItem(item2)

    item.hovered.connect(item2.setHoveredIdx)
    item2.hovered.connect(item.setHoveredIdx)

    view.show()

    app.exec()
