import numpy as np

from PySide2 import QtCore, QtWidgets
from PySide2.QtCharts import QtCharts

from nanopart.gui.util import array_to_polygonf

from typing import List, Union


class ParticleChart(QtCharts.QChart):
    def __init__(self, parent: QtWidgets.QGraphicsItem = None):
        super().__init__(parent)
        self.setMinimumSize(640, 320)

        self.legend().setVisible(False)

        self.xaxis = QtCharts.QValueAxis()
        self.xaxis.setGridLineVisible(False)
        self.xaxis.rangeChanged.connect(self.updateYRange)
        self.xaxis.setLabelsVisible(False)
        self.yaxis = QtCharts.QValueAxis()
        self.yaxis.setGridLineVisible(False)
        self.yaxis.setLabelFormat("%.2g")

        self.addAxis(self.xaxis, QtCore.Qt.AlignBottom)
        self.addAxis(self.yaxis, QtCore.Qt.AlignLeft)

        self.yvalues: np.ndarray = None

        self.series = QtCharts.QLineSeries()
        self.series.setUseOpenGL(True)  # Speed for many line?
        self.addSeries(self.series)
        self.series.attachAxis(self.xaxis)
        self.series.attachAxis(self.yaxis)

        self.lines: List[QtCharts.QLineSeries] = []
        for i in range(3):
            line = QtCharts.QLineSeries()
            self.addSeries(line)
            line.attachAxis(self.xaxis)
            line.attachAxis(self.yaxis)
            self.lines.append(line)

        self.trims: List[QtCharts.QLineSeries] = []
        for i in range(2):
            line = QtCharts.QLineSeries()
            self.addSeries(line)
            line.attachAxis(self.xaxis)
            line.attachAxis(self.yaxis)
            self.trims.append(line)

    def setData(self, points: np.ndarray) -> None:
        self.yvalues = points[:, 1]
        poly = array_to_polygonf(points)
        self.series.replace(poly)

    def setTrim(self, left: int, right: int) -> None:
        ymin, ymax = 0, 1e99

        for line, value in zip(self.trims, [left, right]):
            line.replace([QtCore.QPointF(value, ymin), QtCore.QPointF(value, ymax)])
        self.update()

    def setLines(self, ub: float, lc: float, ld: float) -> None:
        xmin, xmax = 0, self.series.count()

        for line, value in zip(self.lines, [ub, lc, ld]):
            line.replace([QtCore.QPointF(xmin, value), QtCore.QPointF(xmax, value)])
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

        ymax = np.max(self.yvalues[int(xmin) : int(xmax)])
        self.yaxis.setRange(0.0, ymax)


class ParticleResultsChart(QtCharts.QChart):
    def __init__(self, parent: QtWidgets.QGraphicsItem = None):
        super().__init__(parent)
        self.setMinimumSize(640, 320)

        self.legend().setVisible(False)

        self._xaxis = QtCharts.QValueAxis()
        self._xaxis.setVisible(False)
        self.xaxis = QtCharts.QValueAxis()
        self.xaxis.setGridLineVisible(False)

        self.yaxis = QtCharts.QValueAxis()
        self.yaxis.setGridLineVisible(False)
        self.yaxis.setLabelFormat("%d")

        self.addAxis(self._xaxis, QtCore.Qt.AlignBottom)
        self.addAxis(self.xaxis, QtCore.Qt.AlignBottom)
        self.addAxis(self.yaxis, QtCore.Qt.AlignLeft)

        self.series = QtCharts.QBarSeries()
        self.series.setBarWidth(1.0)
        self.addSeries(self.series)
        self.series.attachAxis(self._xaxis)
        self.series.attachAxis(self.yaxis)

    def setData(
        self,
        data: np.ndarray,
        bins: Union[int, str] = "auto",
        min_bins: int = 64,
        max_bins: int = 256,
    ) -> None:
        # vmin, vmax = np.percentile(data, 5), np.percentile(data, 95)
        vmin, vmax = np.percentile(data, 0), np.percentile(data, 100)

        barset = QtCharts.QBarSet("histogram")
        # barset.setColor(sequential[1])
        # barset.setLabelColor(light_theme["text"])

        bin_edges = np.histogram_bin_edges(data, bins=bins, range=(vmin, vmax))
        if bin_edges.size > max_bins:
            bin_edges = np.histogram_bin_edges(data, bins=max_bins, range=(vmin, vmax))
        elif bin_edges.size < min_bins:
            bin_edges = np.histogram_bin_edges(data, bins=min_bins, range=(vmin, vmax))

        hist, edges = np.histogram(data, bins=bin_edges)
        barset.append(list(hist))

        self.series.clear()
        self.series.append(barset)

        self._xaxis.setRange(-0.5, hist.size - 0.5)
        self.xaxis.setRange(edges[0], edges[-1])
        self.yaxis.setRange(0, np.amax(hist))
        self.yaxis.applyNiceNumbers()

        self.update()
