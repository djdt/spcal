from PySide6 import QtCore, QtGui, QtWidgets
import numpy as np

from spcal.gui.graphs import color_schemes

from typing import List


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

        # self.setAcceptHoverEvents(True)

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
        for item in self.slices:
            self.scene().removeItem(item)
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

    def boundingRect(self):
        return QtCore.QRectF(
            -self.radius, -self.radius, self.radius * 2, self.radius * 2
        )

    def paint(self):
        pass


app = QtWidgets.QApplication()

scene = QtWidgets.QGraphicsScene(-100, -100, 200, 200)
view = QtWidgets.QGraphicsView(scene)
view.setMouseTracking(True)

item = PieChart(100.0, [0.1, 0.7, 0.2], color_schemes["IBM Carbon"])
item.setPos(0.0, 0.0)
scene.addItem(item)

view.show()

app.exec()
