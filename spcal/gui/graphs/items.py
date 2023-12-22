
import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

# class MarkerItem(QtWidgets.QGraphicsPathItem):
#     def __init__(
#         self,
#         x: float,
#         y: float,
#         text: str = "",
#         height: float = 6.0,
#         pen: QtGui.QPen | None = None,
#         brush: QtGui.QBrush | None = None,
#         parent: QtWidgets.QGraphicsItem | None = None,
#     ):
#         if pen is None:
#             pen = QtGui.QPen(QtCore.Qt.black, 1.0)
#             pen.setCosmetic(True)

#         if brush is None:
#             brush = QtGui.QBrush(QtCore.Qt.black)

#         super().__init__(parent)
#         self.setFlag(self.GraphicsItemFlag.ItemIgnoresTransformations, True)
#         self.setPos(x, y)

#         width = height / np.sqrt(3.0)
#         path = QtGui.QPainterPath()
#         path.addPolygon(
#             QtGui.QPolygonF(
#                 [
#                     QtCore.QPointF(0, 0),
#                     QtCore.QPointF(-width, -height),
#                     QtCore.QPointF(width, -height),
#                 ]
#             )
#         )
#         self.setPath(path)
#         self.setPen(pen)
#         self.setBrush(brush)

#         self.text = QtWidgets.QGraphicsSimpleTextItem(text, self)
#         # self.text.setPen(pen)
#         # self.text.setBrush(brush)

#         trect = QtGui.QFontMetrics(self.text.font()).boundingRect(self.text.text())
#         self.text.setPos(-trect.width() / 2.0, -(height + trect.height()))

#     def paint(
#         self,
#         painter: QtGui.QPainter,
#         option: QtWidgets.QStyleOptionGraphicsItem,
#         widget: QtWidgets.QWidget | None = None,
#     ) -> None:
#         painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
#         super().paint(painter, option, widget)


class HoverableChartItem(pyqtgraph.GraphicsObject):
    hovered = QtCore.Signal(int)

    def __init__(self, parent: QtWidgets.QGraphicsItem | None = None):
        """Item with multiple hoverable sections.
        Sections are defined by self.path and by default are highlighted on
        hover."""
        super().__init__(parent)
        self.setAcceptHoverEvents(True)

        self.hovered_idx = -1

    @property
    def paths(self) -> list[QtGui.QPainterPath]:
        raise NotImplementedError

    def setHoveredIdx(self, idx: int) -> None:
        if self.hovered_idx != idx:
            self.hovered_idx = idx
            self.update()

    def hoverEvent(self, event):
        hovered_idx = -1
        if not event.exit:
            for i, path in enumerate(self.paths):
                if path.contains(event.pos()):
                    hovered_idx = i
                    break

        if hovered_idx != self.hovered_idx:
            self.hovered.emit(hovered_idx)
            self.update()
        self.hovered_idx = hovered_idx

    def boundingRect(self) -> QtCore.QRectF:
        raise NotImplementedError

    def shape(self) -> QtGui.QPainterPath:
        raise NotImplementedError

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

    def dataBounds(self, ax, frac, orthoRange=None):
        """Pad by pen width"""
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
        """Pad by pen width"""
        if self.pen.isCosmetic():
            return max(1, self.pen.width()) * 0.7072
        else:
            return 0.0


class BarChart(HoverableChartItem):
    def __init__(
        self,
        height: float,
        width: float,
        values: list[float],
        brushes: list[QtGui.QBrush],
        pen: QtGui.QPen | None = None,
        labels: list[str] | None = None,
        # label_format: str = "{:.4g}",
        parent: QtWidgets.QGraphicsItem | None = None,
    ):
        """Pie is centered on item.pos()."""
        super().__init__(parent)

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)

        self.pen = pen
        self.brushes = brushes

        self.height = height
        self.width = width
        self.values = values
        self._paths: list[QtGui.QPainterPath] = []

        self.labels: list[pyqtgraph.TextItem] = []
        # if labels is not None:
        #     assert len(labels) == len(values)
        #     angle = 0.0
        #     for label in labels:

        # self.label_format = label_format

    @property
    def paths(self) -> list[QtGui.QPainterPath]:
        if len(self._paths) == 0:
            rect = self.boundingRect()
            total = np.sum(self.values)
            top = rect.height()
            for value in self.values:
                path = QtGui.QPainterPath()
                height = value / total * rect.height()
                path.addRect(0, top - height, rect.width(), height)
                self._paths.append(path)
                top -= height

        return self._paths

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(0, 0, self.width, self.height)

    def shape(self) -> QtGui.QPainterPath:
        path = QtGui.QPainterPath()
        path.addRect(self.boundingRect())
        return path


class PieChart(HoverableChartItem):
    def __init__(
        self,
        radius: float,
        values: list[float],
        brushes: list[QtGui.QBrush],
        pen: QtGui.QPen | None = None,
        labels: list[str] | None = None,
        # label_format: str = "{:.4g}",
        parent: QtWidgets.QGraphicsItem | None = None,
    ):
        """Pie is centered on item.pos()."""
        super().__init__(parent)

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)

        self.pen = pen
        self.brushes = brushes

        self.radius = radius
        self.values = values
        self._paths: list[QtGui.QPainterPath] = []

        self.labels: list[pyqtgraph.TextItem] = []
        # if labels is not None:
        #     assert len(labels) == len(values)
        #     angle = 0.0
        #     for label in labels:

        # self.label_format = label_format

    @property
    def paths(self) -> list[QtGui.QPainterPath]:
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

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(
            -self.radius, -self.radius, self.radius * 2, self.radius * 2
        )

    def shape(self) -> QtGui.QPainterPath:
        path = QtGui.QPainterPath()
        path.addEllipse(self.boundingRect())
        return path


class StaticRectItemSample(pyqtgraph.GraphicsWidget):
    def __init__(self, brush: QtGui.QBrush):
        super().__init__()
        self.brush = brush

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(0, 0, 20, 20)

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ) -> None:
        painter.save()
        painter.setBrush(self.brush)
        painter.drawRect(QtCore.QRectF(2, 2, 18, 18))
        painter.restore()


if __name__ == "__main__":
    from spcal.gui.graphs import color_schemes

    app = QtWidgets.QApplication()

    view = pyqtgraph.PlotWidget(background="white")
    view.setAspectLocked(1.0)
    legend = view.addLegend(sampleType=StaticRectItemSample)
    # item = pyqtgraph.BarGraphItem(
    #     x=[1, 2, 3], y=[4, 5, 6], width=1.0, y1=0.0, name="eggs"
    # )
    # view.addItem(item)

    item = PieChart(100.0, [1.0, 2.0, 3.0, 4.0], color_schemes["IBM Carbon"])
    item.setPos(0.0, 0.0)
    view.addItem(item)

    item2 = PieChart(100.0, [23.0, 12.0, 0.0, 32.1], color_schemes["IBM Carbon"])
    item2.setPos(300.0, 0.0)
    view.addItem(item2)

    for name, color in zip("abcd", color_schemes["IBM Carbon"]):
        legend.addItem(StaticRectItemSample(QtGui.QBrush(color)), name)

    item.hovered.connect(item2.setHoveredIdx)
    item2.hovered.connect(item.setHoveredIdx)

    view.show()

    app.exec()
