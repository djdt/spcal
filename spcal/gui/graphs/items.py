import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets


class HoverableChartItem(pyqtgraph.GraphicsObject):
    hovered = QtCore.Signal(int)

    def __init__(
        self,
        paths: list[QtGui.QPainterPath],
        values: list[float] | np.ndarray,
        label_positions: list[QtCore.QPointF],
        labels: list[str] | None = None,
        font: QtGui.QFont | None = None,
        pen: QtGui.QPen | None = None,
        brushes: list[QtGui.QBrush] | None = None,
        parent: QtWidgets.QGraphicsItem | None = None,
    ):
        """Item with multiple hoverable sections.
        Sections are defined by self.path and by default are highlighted on
        hover."""
        super().__init__(parent)
        self.setAcceptHoverEvents(True)

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0)
            pen.setCosmetic(True)
        if brushes is None:
            brushes = [QtGui.QBrush(QtCore.Qt.GlobalColor.red)]

        self.paths = paths
        self.values = values
        self.label_positions = label_positions
        self.labels = labels

        self.font = font

        self.pen = pen
        self.brushes = brushes

        self.hovered_idx = -1

    def setHoveredIdx(self, idx: int):
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
            self.hovered.emit(hovered_idx)  # type: ignore
            self.update()
        self.hovered_idx = hovered_idx

    def boundingRect(self) -> QtCore.QRectF:  # type: ignore
        raise NotImplementedError

    def shape(self) -> QtGui.QPainterPath:  # type: ignore
        raise NotImplementedError

    def paint(  # type: ignore
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ):
        painter.save()
        if self.font is not None:
            painter.setFont(self.font)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(self.pen)
        pwidth = self.pen.widthF()
        for i, (path, value, brush) in enumerate(
            zip(self.paths, self.values, self.brushes)
        ):
            if value == 0.0 or np.isnan(value):
                continue
            if i == self.hovered_idx:
                brush = QtGui.QBrush(brush)
                brush.setColor(brush.color().lighter())
            painter.setBrush(brush)
            painter.drawPath(path)
            if i == self.hovered_idx and self.labels is not None:
                painter.save()
                # save position then reset the painter transform to remove scaling
                pos = painter.transform().map(self.label_positions[i])
                painter.setTransform(self.transform())  # type: ignore , pyqtgraph weirdness

                # now draw unscaled text
                rect = painter.fontMetrics().boundingRect(self.labels[i]).toRectF()
                rect.moveCenter(pos)
                rect.adjust(-pwidth, -pwidth, pwidth, pwidth)
                painter.drawText(rect, self.labels[i])

                painter.restore()
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
        values: list[float] | np.ndarray,
        brushes: list[QtGui.QBrush],
        labels: list[str] | None = None,
        font: QtGui.QFont | None = None,
        pen: QtGui.QPen | None = None,
        parent: QtWidgets.QGraphicsItem | None = None,
    ):
        """Pie is centered on item.pos()."""
        self.height = height
        self.width = width

        rect = self.boundingRect()
        total = np.sum(values)
        bottom = 0
        paths = []
        label_pos = []
        for value in values:
            path = QtGui.QPainterPath()
            height = value / total * rect.height()
            path.addRect(0, bottom, rect.width(), height)
            paths.append(path)
            label_pos.append(path.boundingRect().center())
            bottom += height

        super().__init__(paths, values, label_pos, labels, font, pen, brushes, parent)

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
        values: list[float] | np.ndarray,
        brushes: list[QtGui.QBrush],
        labels: list[str] | None = None,
        font: QtGui.QFont | None = None,
        pen: QtGui.QPen | None = None,
        parent: QtWidgets.QGraphicsItem | None = None,
    ):
        """Pie is centered on item.pos()."""
        self.radius = radius

        rect = self.boundingRect()
        total = np.sum(values)
        angle = 0.0

        paths = []
        label_pos = []
        for value in values:
            span = 360.0 * value / total
            path = QtGui.QPainterPath(QtCore.QPointF(0, 0))
            path.arcTo(rect, angle, span)
            paths.append(path)
            label_pos.append(
                QtCore.QPointF(
                    self.radius * 0.5 * np.cos(np.deg2rad(-angle - 0.5 * span)),
                    self.radius * 0.5 * np.sin(np.deg2rad(-angle - 0.5 * span)),
                )
            )
            angle += span
        super().__init__(paths, values, label_pos, labels, font, pen, brushes, parent)

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(
            -self.radius, -self.radius, self.radius * 2.0, self.radius * 2.0
        )

    def shape(self) -> QtGui.QPainterPath:
        path = QtGui.QPainterPath()
        path.addEllipse(self.boundingRect())
        return path


class StaticRectItemSample(pyqtgraph.GraphicsWidget):
    def __init__(self, brush: QtGui.QBrush):
        super().__init__()
        self.brush = brush

    def boundingRect(self) -> QtCore.QRectF:  # type: ignore
        return QtCore.QRectF(0, 0, 20, 20)

    def paint(  # type: ignore
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ):
        painter.save()
        painter.setBrush(self.brush)
        painter.drawRect(QtCore.QRectF(2, 2, 18, 18))
        painter.restore()
