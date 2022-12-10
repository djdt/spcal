from typing import List

import pyqtgraph
from PySide6 import QtCore, QtGui


class HistogramItemSample(pyqtgraph.ItemSample):
    """Legend item for a histogram and its fit."""

    def __init__(
        self,
        histogram: pyqtgraph.PlotDataItem,
        fit: pyqtgraph.PlotCurveItem | None = None,
    ):
        super().__init__(histogram)
        self.setFixedWidth(20)
        self.item2 = fit

    def mouseClickEvent(
        self, event: QtGui.QMouseEvent
    ):  # Dumb pyqtgraph class, use pos()
        """Use the mouseClick event to toggle the visibility of the plotItem"""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if QtCore.QRectF(0, 0, 20, 20).contains(event.pos()):
                visible = self.item.isVisible()
                self.item.setVisible(not visible)
            elif self.item2 is not None and QtCore.QRectF(23, 0, 20, 20).contains(
                event.pos()
            ):
                visible = self.item2.isVisible()
                self.item2.setVisible(not visible)

        event.accept()
        self.update()

    def setFit(self, fit: pyqtgraph.PlotCurveItem | None) -> None:
        self.item2 = fit
        self.prepareGeometryChange()

    def boundingRect(self):
        return QtCore.QRectF(0, 0, 43, 20)

    def paint(self, painter: QtGui.QPainter, *args):
        opts = self.item.opts
        if opts.get("antialias"):
            painter.setRenderHint(painter.RenderHint.Antialiasing)

        visible = self.item.isVisible()
        if not visible:
            icon = pyqtgraph.icons.invisibleEye.qicon
            painter.drawPixmap(QtCore.QPoint(1, 1), icon.pixmap(18, 18))
        else:
            painter.setBrush(pyqtgraph.mkBrush(opts["brush"]))
            painter.drawRect(QtCore.QRectF(2, 2, 18, 18))

        if self.item2 is not None:
            visible2 = self.item2.isVisible()
            opts = self.item2.opts
            if not visible2:
                icon = pyqtgraph.icons.invisibleEye.qicon
                painter.drawPixmap(QtCore.QPoint(20 + 3 + 1, 1), icon.pixmap(18, 18))
            else:
                painter.setPen(pyqtgraph.mkPen(opts["pen"]))
                painter.drawLine(23, 11, 40, 11)


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