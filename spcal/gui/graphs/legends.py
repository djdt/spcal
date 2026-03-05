import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets


class FontScaledItemSample(pyqtgraph.ItemSample):
    def __init__(
        self,
        item: QtWidgets.QGraphicsItem | None,
        font_metrics: QtGui.QFontMetrics,
        brush: QtGui.QBrush | None = None,
    ):
        super().__init__(item)
        self.setFixedHeight(font_metrics.ascent())
        self.setFixedWidth(font_metrics.ascent())
        self.pad = font_metrics.lineWidth()
        self.brush = brush

    def boundingRect(self):
        return QtCore.QRectF(  # vertically center
            0, 0, self.width(), self.height()
        )

    def paint(  # type: ignore
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ):
        painter.save()
        painter.setBrush(self.brush)
        painter.drawRect(
            self.boundingRect().adjusted(self.pad, self.pad, -self.pad, -self.pad)
        )
        painter.restore()


class HistogramItemSample(FontScaledItemSample):
    """Legend item for a histogram and its fit."""

    def __init__(
        self,
        font_metrics: QtGui.QFontMetrics,
        histograms: list[pyqtgraph.PlotCurveItem],
        fit: pyqtgraph.PlotCurveItem | None = None,
        limits: list[pyqtgraph.InfiniteLine] | None = None,
    ):
        super().__init__(histograms[0], font_metrics)
        self.other_items = histograms[1:]
        self.item_fit = fit
        self.item_limits = []
        if limits is not None:
            self.item_limits.extend(limits)

        self.setFixedWidth(self.height() * 3)

    def setLimitsVisible(self, visible: bool):
        for limit in self.item_limits:
            limit.setVisible(visible)
        self.update()

    def setSibilingsVisble(
        self,
        hist: bool | None = None,
        limits: bool | None = None,
        fit: bool | None = None,
    ):
        for item in self.parentItem().childItems():
            if isinstance(item, HistogramItemSample) and item != self:
                if hist is not None:
                    for other in item.other_items:
                        other.setVisible(hist)
                    item.item.setVisible(hist)
                if limits is not None:
                    item.setLimitsVisible(limits)
                if fit is not None and item.item_fit is not None:
                    item.item_fit.setVisible(fit)
                item.update()

    def mouseClickEvent(
        self, event: QtGui.QMouseEvent
    ):  # Dumb pyqtgraph class, use pos()
        """Use the mouseClick event to toggle the visibility of the plotItem"""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if event.pos().x() < self.boundingRect().left() + self.height():
                if len(self.item_limits) > 0:
                    if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                        self.setLimitsVisible(True)
                        self.setSibilingsVisble(hist=False)
                    else:
                        visible = any(limit.isVisible() for limit in self.item_limits)
                        self.setLimitsVisible(not visible)
            elif event.pos().x() < self.boundingRect().right() - self.height():
                if self.item_fit is not None:
                    if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                        self.item_fit.setVisible(True)
                        self.setSibilingsVisble(fit=False)
                    else:
                        self.item_fit.setVisible(not self.item_fit.isVisible())
            else:
                if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                    self.item.setVisible(True)
                    for item in self.other_items:
                        item.setVisible(True)
                    self.setSibilingsVisble(hist=False)
                else:
                    visible = not self.item.isVisible()
                    self.item.setVisible(visible)
                    for item in self.other_items:
                        item.setVisible(visible)

            event.accept()
            self.update()

    def drawHiddenIcon(self, painter: QtGui.QPainter, rect: QtCore.QRectF):
        icon = pyqtgraph.icons.invisibleEye.qicon  # type: ignore
        painter.drawPixmap(
            rect.topLeft().toPoint(), icon.pixmap(rect.width(), rect.height())
        )

    def paint(  # type: ignore
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ):
        # painter.drawRect(self.boundingRect())
        painter.setRenderHint(painter.RenderHint.Antialiasing)

        rect = self.boundingRect()
        rect.setWidth(self.width() / 3)

        if len(self.item_limits) > 0:
            if not all(limit.isVisible() for limit in self.item_limits):
                self.drawHiddenIcon(
                    painter, rect.adjusted(self.pad, self.pad, -self.pad, -self.pad)
                )
            else:
                painter.setPen(self.item_limits[0].pen)
                painter.drawLine(
                    QtCore.QPointF(rect.center().x(), self.pad),
                    QtCore.QPointF(rect.center().x(), rect.top() - self.pad),
                )

        rect.moveLeft(rect.left() + self.height())
        if self.item_fit is not None:
            opts = self.item_fit.opts
            if not self.item_fit.isVisible():
                self.drawHiddenIcon(
                    painter, rect.adjusted(self.pad, self.pad, -self.pad, -self.pad)
                )
            else:
                path = self.pathForFit(
                    rect.left() + self.pad,
                    self.pad,
                    rect.left() + self.height() - self.pad * 2.0,
                    self.height() - self.pad * 2.0,
                )
                painter.strokePath(path, pyqtgraph.mkPen(opts["pen"]))

        rect.moveLeft(rect.left() + self.height())
        if not self.item.isVisible():
            self.drawHiddenIcon(
                painter, rect.adjusted(self.pad, self.pad, -self.pad, -self.pad)
            )
        else:
            opts = self.item.opts
            painter.setBrush(pyqtgraph.mkBrush(opts["brush"]))
            painter.setPen(pyqtgraph.mkPen(opts["pen"]))
            painter.drawPath(
                self.pathForHist(
                    rect.adjusted(self.pad, self.pad, -self.pad, -self.pad)
                )
            )

    def pathForFit(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> QtGui.QPainterPath:
        xm = x0 + (x1 - x0) / 2.0
        path = QtGui.QPainterPath()
        path.moveTo(x0, y1)
        path.cubicTo(x0 + (xm - x0) / 2.0, y1, xm - (xm - x0) / 3.0, y0, xm, y0)
        path.cubicTo(xm + (xm - x0) / 3.0, y0, x1 - (xm - x0) / 2.0, y1, x1, y1)
        return path

    def pathForHist(self, rect: QtCore.QRectF) -> QtGui.QPainterPath:
        dx = rect.width() / 3.0
        dy = rect.height() / 3.0
        path = QtGui.QPainterPath()
        for i, h in enumerate([2.0, 1.0, 3.0]):
            r = QtCore.QRectF(rect.left() + dx * i, rect.bottom() - dy * h, dx, dy * h)
            path.addRect(r)
        return path

    def addLimit(self, limit: pyqtgraph.InfiniteLine):
        self.item_limits.append(limit)
        self.prepareGeometryChange()

    def setFit(self, fit: pyqtgraph.PlotCurveItem | None):
        self.item_fit = fit
        self.prepareGeometryChange()


class ParticleItemSample(FontScaledItemSample):
    """Legend item for signals and limits in a particle view."""

    def __init__(
        self,
        font_metrics: QtGui.QFontMetrics,
        signals: pyqtgraph.PlotCurveItem,
        detections: pyqtgraph.ScatterPlotItem,
        lines: list[pyqtgraph.PlotCurveItem] | None = None,
    ):
        super().__init__(signals, font_metrics)
        self.detections = detections
        self.lines = lines

        self.setFixedWidth(self.height() * 2)

    def setSibilingsVisble(
        self,
        signal: bool | None = None,
        detections: bool | None = None,
        lines: bool | None = None,
    ):
        for item in self.parentItem().childItems():
            if isinstance(item, ParticleItemSample) and item != self:
                if signal is not None:
                    item.item.setVisible(signal)
                if detections is not None:
                    item.detections.setVisible(detections)
                if lines is not None and item.lines is not None:
                    for line in item.lines:
                        line.setVisible(lines)
                item.update()

    def mouseClickEvent(
        self, event: QtWidgets.QGraphicsSceneMouseEvent
    ):  # Dumb pyqtgraph class, use pos()
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if event.pos().x() < self.boundingRect().center().x():  # lines
                if self.lines is not None:
                    if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                        for item in self.lines:
                            item.setVisible(True)
                        self.setSibilingsVisble(lines=False)
                    else:
                        visible = any(item.isVisible() for item in self.lines)
                        for item in self.lines:
                            item.setVisible(not visible)
            else:
                if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                    self.item.setVisible(True)
                    self.detections.setVisible(True)
                    self.setSibilingsVisble(signal=False, detections=False)
                else:
                    visible = self.item.isVisible()
                    self.item.setVisible(not visible)
                    self.detections.setVisible(not visible)

            event.accept()
            self.update()

    def drawHiddenIcon(self, painter: QtGui.QPainter, rect: QtCore.QRectF):
        icon = pyqtgraph.icons.invisibleEye.qicon  # type: ignore
        painter.drawPixmap(
            rect.topLeft().toPoint(), icon.pixmap(rect.width(), rect.height())
        )

    def paint(  # type: ignore
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ):
        # painter.drawRect(self.boundingRect())
        painter.setRenderHint(painter.RenderHint.Antialiasing)

        rect = self.boundingRect()
        rect.setRight(rect.left() + self.height())

        if self.lines is not None and len(self.lines) > 0:
            if all(line.isVisible() for line in self.lines):
                painter.save()
                painter.setPen(self.lines[0].opts["pen"])
                painter.drawLine(
                    QtCore.QPointF(rect.left() + self.pad * 2, rect.center().y()),
                    QtCore.QPointF(rect.right() - self.pad * 2, rect.center().y()),
                )
                painter.restore()
            else:
                self.drawHiddenIcon(
                    painter, rect.adjusted(self.pad, self.pad, -self.pad, -self.pad)
                )

        rect.moveLeft(rect.left() + self.height())
        if self.item.isVisible():
            painter.setRenderHint(painter.RenderHint.Antialiasing, False)
            painter.setBrush(self.detections.opts["brush"])
            painter.drawRect(rect.adjusted(self.pad, self.pad, -self.pad, -self.pad))
        else:
            self.drawHiddenIcon(
                painter, rect.adjusted(self.pad, self.pad, -self.pad, -self.pad)
            )
