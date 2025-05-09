import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets


class StaticRectItemSample(pyqtgraph.GraphicsWidget):
    def __init__(self, brush: QtGui.QBrush):
        super().__init__()
        self.item = None
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


class HistogramItemSample(pyqtgraph.ItemSample):
    """Legend item for a histogram and its fit."""

    def __init__(
        self,
        histograms: list[pyqtgraph.PlotDataItem],
        fit: pyqtgraph.PlotCurveItem | None = None,
        limits: list[pyqtgraph.InfiniteLine] | None = None,
        size: float = 20.0,
        pad: float = 2.0,
    ):
        super().__init__(histograms[0])
        self.other_items = histograms[1:]
        self.item_fit = fit
        self.item_limits = []
        if limits is not None:
            self.item_limits.extend(limits)

        self.size = size
        self.pad = pad
        self.setFixedWidth(self.size * 2 + self.pad)

    def setLimitsVisible(self, visible: bool):
        for limit in self.item_limits:
            limit.setVisible(visible)
        self.update()

    def boundingRect(self):
        return QtCore.QRectF(0, 0, self.size * 3 + self.pad * 2, self.size)

    def mouseClickEvent(
        self, event: QtGui.QMouseEvent
    ):  # Dumb pyqtgraph class, use pos()
        """Use the mouseClick event to toggle the visibility of the plotItem"""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if len(self.item_limits) > 0 and QtCore.QRectF(
                0, 0, self.size, self.size
            ).contains(event.pos()):
                if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                    self.setLimitsVisible(True)
                    for item in self.parentItem().childItems():
                        if isinstance(item, HistogramItemSample) and item != self:
                            item.setLimitsVisible(False)
                else:
                    visible = any(limit.isVisible() for limit in self.item_limits)
                    self.setLimitsVisible(not visible)
                self.update()
            elif self.item_fit is not None and QtCore.QRectF(
                self.size + self.pad, 0, self.size, self.size
            ).contains(event.pos()):
                if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                    self.item_fit.setVisible(True)
                    for item in self.parentItem().childItems():
                        if isinstance(item, HistogramItemSample) and item != self:
                            item.item_fit.setVisible(False)
                            item.update()
                else:
                    self.item_fit.setVisible(not self.item_fit.isVisible())
            elif QtCore.QRectF(
                (self.size + self.pad) * 2.0, 0, self.size, self.size
            ).contains(event.pos()):
                if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                    self.item.setVisible(True)
                    for item in self.other_items:
                        item.setVisible(True)
                    for item in self.parentItem().childItems():
                        if isinstance(item, HistogramItemSample) and item != self:
                            item.item.setVisible(False)
                            for it in item.other_items:
                                it.setVisible(False)
                            item.update()
                else:
                    visible = not self.item.isVisible()
                    self.item.setVisible(visible)
                    for item in self.other_items:
                        item.setVisible(visible)
                self.update()

        event.accept()
        self.update()

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ) -> None:
        painter.setRenderHint(painter.RenderHint.Antialiasing)

        offset = 0.0
        if len(self.item_limits) > 0:
            if not all(limit.isVisible() for limit in self.item_limits):
                icon = pyqtgraph.icons.invisibleEye.qicon
                painter.drawPixmap(
                    QtCore.QPoint(offset + self.pad, self.pad),
                    icon.pixmap(self.size - self.pad * 2.0, self.size - self.pad * 2.0),
                )
            else:
                painter.setPen(self.item_limits[0].pen)
                painter.drawLine(
                    offset + self.size / 2.0,
                    self.pad,
                    offset + self.size / 2.0,
                    self.size - self.pad,
                )

        offset = self.size + self.pad
        if self.item_fit is not None:
            opts = self.item_fit.opts
            if not self.item_fit.isVisible():
                icon = pyqtgraph.icons.invisibleEye.qicon
                painter.drawPixmap(
                    QtCore.QPoint(offset + self.pad, self.pad),
                    icon.pixmap(self.size - self.pad * 2.0, self.size - self.pad * 2.0),
                )
            else:
                path = self.pathForFit(
                    offset + self.pad,
                    self.pad,
                    offset + self.size - self.pad * 2.0,
                    self.size - self.pad * 2.0,
                )
                painter.strokePath(path, pyqtgraph.mkPen(opts["pen"]))

        offset = (self.size + self.pad) * 2.0
        if not self.item.isVisible():
            icon = pyqtgraph.icons.invisibleEye.qicon
            painter.drawPixmap(
                QtCore.QPoint(offset + self.pad, self.pad),
                icon.pixmap(self.size - self.pad * 2.0, self.size - self.pad * 2.0),
            )
        else:
            opts = self.item.opts
            painter.setBrush(pyqtgraph.mkBrush(opts["brush"]))
            painter.setPen(pyqtgraph.mkPen(opts["pen"]))
            painter.drawPath(
                self.pathForHist(
                    offset + self.pad,
                    self.pad,
                    offset + self.size - self.pad * 2.0,
                    self.size - self.pad * 2.0,
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

    def pathForHist(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> QtGui.QPainterPath:
        dx = (x1 - x0) / 3.0
        h = y1 - y0
        path = QtGui.QPainterPath()
        path.addRect(QtCore.QRectF(x0, y0 + h / 2.0, dx, h / 2.0))
        path.addRect(QtCore.QRectF(x0 + dx, y1 - h / 3.0, dx, h / 3.0))
        path.addRect(QtCore.QRectF(x0 + 2.0 * dx, y0, dx, h))
        return path

    def addLimit(self, limit: pyqtgraph.InfiniteLine) -> None:
        self.item_limits.append(limit)
        self.prepareGeometryChange()

    def setFit(self, fit: pyqtgraph.PlotCurveItem | None) -> None:
        self.item_fit = fit
        self.prepareGeometryChange()


class MultipleItemSampleProxy(pyqtgraph.ItemSample):
    """Draws a colored block legend and links visibility to multiple items."""

    def __init__(
        self,
        brush: QtGui.QBrush | QtGui.QColor,
        items: list[pyqtgraph.PlotDataItem] | None = None,
    ):
        if isinstance(brush, QtGui.QColor):
            brush = QtGui.QBrush(brush)

        super().__init__(
            pyqtgraph.BarGraphItem(
                pen=QtGui.QPen(QtCore.Qt.PenStyle.NoPen),
                brush=brush,
                x=0,
                y=0,
                width=0,
                height=0,
            )
        )
        self.items = []
        if items is not None:
            self.items.extend(items)

    def setItemsVisible(self, visible: bool) -> None:
        self.item.setVisible(visible)
        for item in self.items:
            item.setVisible(visible)
        self.update()

    def addItem(self, item: pyqtgraph.PlotDataItem) -> None:
        self.items.append(item)
        item.setVisible(self.item.isVisible())
        self.update()

    def mouseClickEvent(self, event: QtGui.QMouseEvent):
        """Use the mouseClick event to toggle the visibility of the plotItem"""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            visible = self.item.isVisible()
            if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                self.setItemsVisible(True)
                for item in self.parentItem().childItems():
                    if isinstance(item, MultipleItemSampleProxy) and item != self:
                        item.setItemsVisible(False)
            else:
                self.setItemsVisible(not visible)

            event.accept()
