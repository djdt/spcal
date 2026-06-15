import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets


class FontScaledItemSample(pyqtgraph.ItemSample):
    def __init__(
        self,
        font: QtGui.QFont,
        item: QtWidgets.QGraphicsItem,
        ratio: tuple[int, int] = (1, 1),
        brush: QtGui.QBrush | None = None,
    ):
        super().__init__(item)

        self.font_metrics = QtGui.QFontMetrics(font)
        self.ratio = ratio

        if brush is None:
            brush = QtGui.QBrush()
        self.brush = brush

        self.setFixedWidth(self.font_metrics.ascent() * ratio[0])
        self.setFixedHeight(self.font_metrics.ascent() * ratio[1])
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed
        )

    def pad(self) -> float:
        return self.font_metrics.lineWidth()

    def boundingRect(self):
        return QtCore.QRectF(0, 0, self.width(), self.height())

    def updateFontMetrics(self, font: QtGui.QFont):
        self.font_metrics = QtGui.QFontMetrics(font)
        self.setFixedWidth(self.font_metrics.ascent() * self.ratio[0])
        self.setFixedHeight(self.font_metrics.ascent() * self.ratio[1])
        self.prepareGeometryChange()

    def mouseClickEvent(self, event):
        pass

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
        pad = self.pad()
        pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 0)

        painter.save()
        painter.setPen(pen)
        painter.setBrush(self.brush)

        painter.drawRect(self.boundingRect().adjusted(pad, pad, -pad, -pad))
        painter.restore()


class HistogramItemSample(FontScaledItemSample):
    """Legend item for a histogram and its fit.
    Scales to match font size"""

    def __init__(
        self,
        font: QtGui.QFont,
        histograms: list[pyqtgraph.PlotCurveItem],
        limits: list[pyqtgraph.InfiniteLine] | None = None,
    ):
        super().__init__(font, histograms[0], ratio=(2, 1))
        self.other_items = histograms[1:]
        self.item_limits = []
        if limits is not None:
            self.item_limits.extend(limits)

    def setLimitsVisible(self, visible: bool):
        for limit in self.item_limits:
            limit.setVisible(visible)
        self.update()

    def setSibilingsVisble(
        self,
        hist: bool | None = None,
        limits: bool | None = None,
    ):
        for item in self.parentItem().childItems():
            if isinstance(item, HistogramItemSample) and item != self:
                if hist is not None:
                    for other in item.other_items:
                        other.setVisible(hist)
                    item.item.setVisible(hist)
                if limits is not None:
                    item.setLimitsVisible(limits)
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

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ):
        painter.setRenderHint(painter.RenderHint.Antialiasing)

        rect = self.boundingRect()
        rect.setWidth(self.width() / 2)

        pad = self.pad()

        if len(self.item_limits) > 0:
            if not all(limit.isVisible() for limit in self.item_limits):
                self.drawHiddenIcon(painter, rect.adjusted(pad, pad, -pad, -pad))
            else:
                painter.setPen(self.item_limits[0].pen)
                painter.drawLine(
                    QtCore.QPointF(rect.center().x(), pad),
                    QtCore.QPointF(rect.center().x(), rect.top() - pad),
                )

        rect.moveLeft(rect.left() + self.width() / 2)
        if not self.item.isVisible():
            self.drawHiddenIcon(painter, rect.adjusted(pad, pad, -pad, -pad))
        else:
            opts = self.item.opts
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 0.0)
            painter.setRenderHint(painter.RenderHint.Antialiasing, False)
            painter.setBrush(pyqtgraph.mkBrush(opts["brush"]))
            painter.setPen(pen)
            painter.drawRect(rect.adjusted(pad, pad, -pad, -pad))

    def addLimit(self, limit: pyqtgraph.InfiniteLine):
        self.item_limits.append(limit)
        self.prepareGeometryChange()


class ParticleItemSample(FontScaledItemSample):
    """Legend item for signals and limits in a particle view."""

    def __init__(
        self,
        font: QtGui.QFont,
        signals: pyqtgraph.PlotCurveItem,
        detections: pyqtgraph.ScatterPlotItem,
        lines: list[pyqtgraph.PlotCurveItem] | None = None,
    ):
        super().__init__(font, signals, ratio=(2, 1))
        self.detections = detections
        self.lines = lines

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

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ):
        painter.setRenderHint(painter.RenderHint.Antialiasing)

        rect = self.boundingRect()
        rect.setRight(rect.left() + self.width() / 2)

        pad = self.pad()

        if self.lines is not None and len(self.lines) > 0:
            if all(line.isVisible() for line in self.lines):
                painter.save()
                painter.setPen(self.lines[0].opts["pen"])
                painter.drawLine(
                    QtCore.QPointF(rect.left() + pad * 2, rect.center().y()),
                    QtCore.QPointF(rect.right() - pad * 2, rect.center().y()),
                )
                painter.restore()
            else:
                self.drawHiddenIcon(painter, rect.adjusted(pad, pad, -pad, -pad))

        rect.moveLeft(rect.left() + self.width() / 2)
        pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 0.0)
        if self.item.isVisible():
            painter.setRenderHint(painter.RenderHint.Antialiasing, False)
            painter.setPen(pen)
            painter.setBrush(self.detections.opts["brush"])
            painter.drawRect(rect.adjusted(pad, pad, -pad, -pad))
        else:
            self.drawHiddenIcon(painter, rect.adjusted(pad, pad, -pad, -pad))
