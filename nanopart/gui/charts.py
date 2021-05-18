import numpy as np
from pathlib import Path

from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCharts import QtCharts

from nanopart.gui.util import array_to_polygonf

from typing import List, Union


class ParticleChartView(QtCharts.QChartView):
    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        event.accept()
        action_copy_to_clipboard = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("insert-image"), "Copy to Clipboard"
        )
        action_copy_to_clipboard.setToolTip(
            "Copy the chart as an image to the clipboard."
        )
        action_copy_to_clipboard.triggered.connect(self.copyToClipboard)

        action_zoom_reset = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("zoom-original"), "Reset View"
        )
        action_zoom_reset.setToolTip("Reset the charts view.")
        action_zoom_reset.triggered.connect(self.chart().zoomReset)

        menu = QtWidgets.QMenu(self)
        menu.addAction(action_copy_to_clipboard)
        menu.addAction(action_zoom_reset)
        menu.exec_(event.globalPos())

    def copyToClipboard(self) -> None:
        QtWidgets.QApplication.clipboard().setPixmap(self.grab(self.viewport().rect()))

    def saveToFile(self, path: Union[str, Path]) -> None:
        if isinstance(path, str):
            path = Path(path)

        pixmap = QtGui.QPixmap(self.viewport().size())
        pixmap.fill()
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        self.render(painter)
        pixmap.save(str(path.absolute()))
        painter.end()


class NiceValueAxis(QtCharts.QValueAxis):
    nicenums = [1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 7.5]

    def __init__(self, nticks: int = 6, parent: QtCore.QObject = None):
        super().__init__(parent)
        self.nticks = nticks

        self.setLabelFormat("%.4g")
        self.setTickType(QtCharts.QValueAxis.TicksDynamic)
        self.setTickAnchor(0.0)
        self.setTickInterval(1e3)

    def setRange(self, amin: float, amax: float) -> None:
        self.fixValues(amin, amax)
        super().setRange(amin, amax)

    def fixValues(self, amin: float, amax: float) -> None:
        delta = amax - amin

        interval = delta / self.nticks
        pwr = 10 ** int(np.log10(interval) - (1 if interval < 1.0 else 0))
        interval = interval / pwr

        idx = np.searchsorted(NiceValueAxis.nicenums, interval)
        idx = min(idx, len(NiceValueAxis.nicenums) - 1)

        interval = NiceValueAxis.nicenums[idx] * pwr
        anchor = int(amin / interval) * interval

        self.setTickAnchor(anchor)
        self.setTickInterval(interval)


class ParticleChart(QtCharts.QChart):
    def __init__(self, parent: QtWidgets.QGraphicsItem = None):
        super().__init__(parent)
        self.setMinimumSize(640, 320)

        self.xaxis = QtCharts.QValueAxis()
        self.xaxis.setGridLineVisible(False)
        self.xaxis.rangeChanged.connect(self.updateYRange)
        self.xaxis.setLabelsVisible(False)
        self.yaxis = QtCharts.QValueAxis()
        self.yaxis.setGridLineVisible(False)
        self.yaxis.setLabelFormat("%.2g")
        self.yaxis.setTitleText("Response")

        self.addAxis(self.xaxis, QtCore.Qt.AlignBottom)
        self.addAxis(self.yaxis, QtCore.Qt.AlignLeft)

        self.yvalues: np.ndarray = None

        self.series = QtCharts.QLineSeries()
        self.series.setPen(QtGui.QPen(QtCore.Qt.black, 1.0))
        self.series.setUseOpenGL(True)  # Speed for many line?
        self.addSeries(self.series)
        self.series.attachAxis(self.xaxis)
        self.series.attachAxis(self.yaxis)

        self.scatter_series = QtCharts.QScatterSeries()
        self.scatter_series.setMarkerSize(5)
        self.scatter_series.setColor(QtCore.Qt.red)
        self.scatter_series.setUseOpenGL(True)
        self.addSeries(self.scatter_series)
        self.scatter_series.attachAxis(self.xaxis)
        self.scatter_series.attachAxis(self.yaxis)

        self.ub = QtCharts.QLineSeries()
        self.lc = QtCharts.QLineSeries()
        self.ld = QtCharts.QLineSeries()

        for series, color, label in zip(
            [self.ub, self.lc, self.ld],
            [QtGui.QColor(255, 0, 0), QtGui.QColor(0, 255, 0), QtGui.QColor(0, 0, 255)],
            ["μb", "Lc", "Ld"],
        ):
            series.setPen(QtGui.QPen(color, 1.0, QtCore.Qt.DashLine))
            series.setName(label)
            series.setUseOpenGL(True)
            self.addSeries(series)
            series.attachAxis(self.xaxis)
            series.attachAxis(self.yaxis)

        self.vlines: List[QtCharts.QLineSeries] = []

        # Clean legend
        self.legend().setMarkerShape(QtCharts.QLegend.MarkerShapeFromSeries)
        self.legend().markers(self.series)[0].setVisible(False)
        self.legend().markers(self.scatter_series)[0].setVisible(False)

    def clearVerticalLines(self) -> None:
        for line in self.vlines:
            self.removeSeries(line)
        self.vlines.clear()

    def setData(self, ys: np.ndarray) -> None:
        xs = np.arange(ys.size)
        self.yvalues = ys
        data = np.stack((xs, ys), axis=1)
        poly = array_to_polygonf(data)
        self.series.replace(poly)

    def setScatter(self, xs: np.ndarray, ys: np.ndarray) -> None:
        data = np.stack((xs, ys), axis=1)
        poly = array_to_polygonf(data)
        self.scatter_series.replace(poly)

    def setBackground(self, xs: np.ndarray, ub: Union[float, np.ndarray]) -> None:
        if isinstance(ub, float):
            ub = np.full(xs.size, ub)

        data = np.stack((xs, ub), axis=1)
        poly = array_to_polygonf(data)
        self.ub.replace(poly)

    def setLimitCritical(self, xs: np.ndarray, lc: Union[float, np.ndarray]) -> None:
        if isinstance(lc, float):
            lc = np.full(xs.size, lc)

        data = np.stack((xs, lc), axis=1)
        poly = array_to_polygonf(data)
        self.lc.replace(poly)

    def setLimitDetection(self, xs: np.ndarray, ld: Union[float, np.ndarray]) -> None:
        if isinstance(ld, float):
            ld = np.full(xs.size, ld)

        data = np.stack((xs, ld), axis=1)
        poly = array_to_polygonf(data)
        self.ld.replace(poly)

    def drawVerticalLines(
        self,
        values: List[float],
        names: List[str] = None,
        pens: List[QtGui.QPen] = None,
        visible_in_legend: List[bool] = None,
    ) -> None:
        self.clearVerticalLines()

        for i, value in enumerate(values):
            line = QtCharts.QLineSeries()
            if pens is not None:
                line.setPen(pens[i])
            if names is not None:
                line.setName(names[i])

            self.addSeries(line)
            line.attachAxis(self.xaxis)
            line.attachAxis(self.yaxis)

            if visible_in_legend is not None and not visible_in_legend[i]:
                self.legend().markers(line)[0].setVisible(False)
            self.vlines.append(line)

        self.setVerticalLines(values)

    def setVerticalLines(self, values: List[float]) -> None:
        ymin, ymax = 0, 1e99
        for line, value in zip(self.vlines, values):
            line.replace([QtCore.QPointF(value, ymin), QtCore.QPointF(value, ymax)])
        self.update()

    def updateYRange(self, xmin: float = None, xmax: float = None) -> None:
        if self.yvalues is None:
            return
        if xmin is None:
            xmin = self.xaxis.min()
        if xmax is None:
            xmax = self.xaxis.max()

        xmin = max(xmin, 0)
        xmax = min(xmax, self.series.count())

        ymax = np.nanmax(self.yvalues[int(xmin) : int(xmax)])
        self.yaxis.setRange(0.0, ymax)


class ParticleHistogram(QtCharts.QChart):
    def __init__(self, parent: QtWidgets.QGraphicsItem = None):
        super().__init__(parent)
        self.setMinimumSize(640, 320)

        self._xaxis = QtCharts.QValueAxis()
        self._xaxis.setVisible(False)

        self.xaxis = NiceValueAxis()
        self.xaxis.setTitleText("Size (nm)")
        self.xaxis.setGridLineVisible(False)

        self.yaxis = QtCharts.QValueAxis()
        self.yaxis.setGridLineVisible(False)
        self.yaxis.setTitleText("Count")
        self.yaxis.setLabelFormat("%d")

        self.addAxis(self._xaxis, QtCore.Qt.AlignBottom)
        self.addAxis(self.xaxis, QtCore.Qt.AlignBottom)
        self.addAxis(self.yaxis, QtCore.Qt.AlignLeft)

        self.series = QtCharts.QBarSeries()
        self.series.setBarWidth(0.9)
        self.addSeries(self.series)
        self.series.attachAxis(self._xaxis)
        self.series.attachAxis(self.yaxis)

        self.set = QtCharts.QBarSet("sizes")
        self.set.setColor(QtCore.Qt.black)
        self.set.hovered.connect(self.barHovered)
        self.series.append(self.set)

        self.fit = QtCharts.QSplineSeries()
        self.fit.setPen(QtGui.QPen(QtGui.QColor(255, 172, 0), 2.0))
        self.addSeries(self.fit)
        self.fit.attachAxis(self.xaxis)
        self.fit.attachAxis(self.yaxis)

        self.label_fit = QtWidgets.QGraphicsTextItem(self)
        self.label_fit.setFont(QtGui.QFont("sans", 12, italic=False))
        self.label_fit.setZValue(99)
        self.label_fit.setDefaultTextColor(QtGui.QColor(255, 172, 0))
        self.xaxis.rangeChanged.connect(self.updateFitLabelPos)

        self.label_hovered = QtWidgets.QGraphicsTextItem(self)
        self.plotAreaChanged.connect(self.updateHoveredLabelPos)

        self.vlines: List[QtCharts.QLineSeries] = []

        # Clean legend
        self.legend().setMarkerShape(QtCharts.QLegend.MarkerShapeFromSeries)
        self.legend().markers(self.series)[0].setVisible(False)

    def updateHoveredLabelPos(self) -> None:
        a = QtCore.QRectF()
        a.topLeft()
        self.label_hovered.setPos(self.plotArea().topRight())

    def updateFitLabelPos(self) -> None:
        pos = self.label_fit.data(0)
        if pos is not None:
            pos = self.mapToPosition(pos, series=self.fit)
            pos -= QtCore.QPointF(
                0,
                self.label_fit.boundingRect().height(),
            )
            self.label_fit.setPos(pos)

    def clearVerticalLines(self) -> None:
        for line in self.vlines:
            self.removeSeries(line)
        self.vlines.clear()

    def drawVerticalLines(
        self,
        values: List[float],
        names: List[str] = None,
        pens: List[QtGui.QPen] = None,
        visible_in_legend: List[bool] = None,
    ) -> None:
        self.clearVerticalLines()

        for i, value in enumerate(values):
            line = QtCharts.QLineSeries()
            if pens is not None:
                line.setPen(pens[i])
            if names is not None:
                line.setName(names[i])

            self.addSeries(line)
            line.attachAxis(self.xaxis)
            line.attachAxis(self.yaxis)

            if visible_in_legend is not None and not visible_in_legend[i]:
                self.legend().markers(line)[0].setVisible(False)
            self.vlines.append(line)

        self.setVerticalLines(values)

    def setVerticalLines(self, values: List[float]) -> None:
        ymin, ymax = self.yaxis.min(), self.yaxis.max()
        for line, value in zip(self.vlines, values):
            line.replace([QtCore.QPointF(value, ymin), QtCore.QPointF(value, ymax)])
        self.update()

    def setData(
        self, data: np.ndarray, bins: np.ndarray, xmin: float = None, xmax: float = None
    ) -> None:
        if xmin is None:
            xmin = bins[0]
        if xmax is None:
            xmax = bins[-1]

        binwidth = bins[1] - bins[0]

        # Pad histogram with zeros to fit xmin, xmax
        _xpad = (int((bins[0] - xmin) / binwidth), int((xmax - bins[-1]) / binwidth))
        data = np.pad(data, _xpad, constant_values=0)

        # Recenter the series on xaxis
        xmin = bins[0] - _xpad[0] * binwidth + binwidth / 2.0
        xmax = bins[-1] + _xpad[1] * binwidth + binwidth / 2.0

        self.set.remove(0, self.set.count())
        self.set.append(list(data))

        self._xaxis.setRange(-0.5, data.size - 0.5)
        self.xaxis.setRange(xmin, xmax)
        self.yaxis.setRange(0.0, np.amax(data))
        self.yaxis.applyNiceNumbers()

    def setFit(self, bins: np.ndarray, fit: np.ndarray) -> None:
        poly = array_to_polygonf(np.stack((bins, fit), axis=1))
        self.fit.replace(poly)

        idx = np.argmax(fit)
        x, y = bins[idx], fit[idx]
        self.label_fit.setPlainText(f"{x:.4g}")
        self.label_fit.setData(0, QtCore.QPointF(x, y))

        self.updateFitLabelPos()

    def barHovered(self, state: bool, index: int) -> None:
        self.label_hovered.setVisible(state)
        if state:
            x = np.round(
                index * ((self.xaxis.max() - self.xaxis.min()) / self.set.count())
                + self.xaxis.min(),
                2,
            )
            y = self.set.at(index)
            text = f"{x:.4g}, {y:.4g}"
            self.label_hovered.setPlainText(text)
            self.label_hovered.setPos(
                self.plotArea().topRight()
                - QtCore.QPointF(
                    self.label_hovered.boundingRect().width(),
                    self.label_hovered.boundingRect().height(),
                )
            )
