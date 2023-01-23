from typing import List

from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.graphs import color_schemes


class MarkerItem(QtWidgets.QGraphicsPathItem):
    def __init__(
        self,
        x: float,
        y: float,
        text: str = "",
        height: float = 6.0,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
        parent: QtWidgets.QGraphicsItem | None = None,
    ):
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)

        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.black)

        super().__init__(parent)
        self.setFlag(self.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setPos(x, y)

        width = height / np.sqrt(3.0)
        path = QtGui.QPainterPath()
        path.addPolygon(
            QtGui.QPolygonF(
                [
                    QtCore.QPointF(0, 0),
                    QtCore.QPointF(-width, -height),
                    QtCore.QPointF(width, -height),
                ]
            )
        )
        self.setPath(path)
        self.setPen(pen)
        self.setBrush(brush)

        self.text = QtWidgets.QGraphicsSimpleTextItem(text, self)
        # self.text.setPen(pen)
        # self.text.setBrush(brush)

        trect = QtGui.QFontMetrics(self.text.font()).boundingRect(self.text.text())
        self.text.setPos(-trect.width() / 2.0, -(height + trect.height()))

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ) -> None:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        super().paint(painter, option, widget)


class PieSlice(QtWidgets.QGraphicsEllipseItem):
    def __init__(
        self,
        radius: float,
        angle: int,
        span: int,
        hover_brush: QtGui.QBrush | None = None,
        parent: QtWidgets.QGraphicsItem | None = None,
    ):
        super().__init__(-radius, -radius, radius * 2, radius * 2, parent=parent)
        self.setStartAngle(angle)
        self.setSpanAngle(span)

        self.hover_brush = hover_brush
        self._brush = self.brush()

    def setBrush(self, brush: QtGui.QBrush) -> None:
        self._brush = brush
        super().setBrush(brush)

    def setHoverBrush(self, brush: QtGui.QBrush) -> None:
        self.hover_brush = brush

    def hoverEnterEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent) -> None:
        if self.hover_brush is not None:
            super().setBrush(self.hover_brush)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent) -> None:
        if self.hover_brush is not None:
            super().setBrush(self._brush)
        super().hoverLeaveEvent(event)

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ) -> None:
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        super().paint(painter, option, widget)
        painter.restore()


class PieChart(QtWidgets.QGraphicsItem):
    def __init__(
        self,
        radius: float,
        fractions: List[float],
        colors: List[QtGui.QColor],
        pen: QtGui.QPen | None = None,
        parent: QtWidgets.QGraphicsItem | None = None,
    ):
        """Pie is centered on item.pos()."""

        assert sum(fractions) == 1.0
        super().__init__(parent=parent)
        self.setFlag(QtWidgets.QGraphicsItem.ItemHasNoContents)

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 0.0)
            pen.setCosmetic(True)
        self.pen = pen

        self.slices: List[PieSlice] = []
        self.radius = radius
        self.buildPie(fractions, colors)

    def buildPie(
        self, fractions: List[float], colors: List[QtGui.QColor]
    ) -> List[PieSlice]:
        self.slices.clear()

        angle = 0
        for frac, color in zip(fractions, colors):
            span = int(360 * 16 * frac)

            item = PieSlice(self.radius, angle, span, parent=self)
            item.setPen(self.pen)
            item.setBrush(QtGui.QBrush(color))
            item.setHoverBrush(QtGui.QBrush(color.lighter()))
            item.setAcceptHoverEvents(True)
            angle += span
            self.slices.append(item)
        return self.slices

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(
            -self.radius, -self.radius, self.radius * 2, self.radius * 2
        )


app = QtWidgets.QApplication()

scene = QtWidgets.QGraphicsScene(-100, -100, 200, 200)
view = QtWidgets.QGraphicsView(scene)
view.setMouseTracking(True)

item = PieChart(100.0, [0.1, 0.7, 0.2], color_schemes["IBM Carbon"])
item.setPos(0.0, 100.0)
scene.addItem(item)

view.show()

app.exec()
