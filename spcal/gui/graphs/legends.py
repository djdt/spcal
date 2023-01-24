from typing import List

import pyqtgraph
from PySide6 import QtCore, QtGui


class HistogramItemSample(pyqtgraph.ItemSample):
    """Legend item for a histogram and its fit."""

    def __init__(
        self,
        histogram: pyqtgraph.PlotDataItem,
        fit: pyqtgraph.PlotCurveItem | None = None,
        limits: List[pyqtgraph.InfiniteLine] | None = None,
    ):
        super().__init__(histogram)
        self.setFixedWidth(20)
        self.item_fit = fit
        self.item_limits = limits

        self.size = 20.0
        self.pad = 2.0

    def boundingRect(self):
        return QtCore.QRectF(0, 0, self.size * 2 + self.pad, self.size)

    def mouseClickEvent(
        self, event: QtGui.QMouseEvent
    ):  # Dumb pyqtgraph class, use pos()
        """Use the mouseClick event to toggle the visibility of the plotItem"""
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return

        if self.item_limits is not None and QtCore.QRectF(
            0, 0, self.size, self.size
        ).contains(event.pos()):
            visible = any(limit.isVisible() for limit in self.item_limits)
            for limit in self.item_limits:
                limit.setVisible(not visible)
        elif self.item_fit is not None and QtCore.QRectF(
            self.size + self.pad, 0, self.size, self.size
        ).contains(event.pos()):
            visible = self.item_fit.isVisible()
            self.item_fit.setVisible(not visible)
        elif QtCore.QRectF(
            (self.size + self.pad) * 2.0, 0, self.size, self.size
        ).contains(event.pos()):
            visible = self.item.isVisible()
            self.item.setVisible(not visible)

        event.accept()
        self.update()

    def paint(self, painter: QtGui.QPainter, *args):
        opts = self.item.opts
        if opts.get("antialias"):
            painter.setRenderHint(painter.RenderHint.Antialiasing)

        offset = 0.0
        if self.item_fit is not None:
            opts = self.item_fit.opts
            if not self.item_fit.isVisible():
                icon = pyqtgraph.icons.invisibleEye.qicon
                painter.drawPixmap(
                    QtCore.QPoint(offset + self.pad, self.pad),
                    icon.pixmap(self.size - self.pad * 2.0, self.size - self.pad * 2.0),
                )
            else:
                painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
                path = self.pathForFit(
                    offset + self.pad,
                    self.pad,
                    offset + self.size - self.pad * 2.0,
                    self.size - self.pad * 2.0,
                )
                painter.strokePath(path, pyqtgraph.mkPen(opts["pen"]))

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
                painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
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
            painter.setBrush(pyqtgraph.mkBrush(opts["brush"]))
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

    def setFit(self, fit: pyqtgraph.PlotCurveItem | None) -> None:
        self.item_fit = fit
        self.prepareGeometryChange()


class MultipleItemSampleProxy(pyqtgraph.ItemSample):
    """Draws a colored block legend and links visibility to multiple items."""

    def __init__(
        self,
        brush: QtGui.QBrush | QtGui.QColor,
        items: List[pyqtgraph.PlotDataItem] | None = None,
    ):
        if isinstance(brush, QtGui.QColor):
            brush = QtGui.QBrush(brush)

        super().__init__(pyqtgraph.BarGraphItem(brush=brush))
        self.items = []
        if items is not None:
            self.items.extend(items)

    def addItem(self, item: pyqtgraph.PlotDataItem) -> None:
        self.items.append(item)
        item.setVisible(self.item.isVisible())
        self.update()

    def mouseClickEvent(self, event: QtGui.QMouseEvent):
        """Use the mouseClick event to toggle the visibility of the plotItem"""
        visible = self.item.isVisible()
        for item in self.items:
            item.setVisible(not visible)
        super().mouseClickEvent(event)
