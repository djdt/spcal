import numpy as np

from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCharts import QtCharts

from nanopart.gui.util import array_to_polygonf

from typing import List, Union


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

        self.addAxis(self.xaxis, QtCore.Qt.AlignBottom)
        self.addAxis(self.yaxis, QtCore.Qt.AlignLeft)

        self.yvalues: np.ndarray = None

        self.series = QtCharts.QLineSeries()
        self.series.setPen(QtGui.QPen(QtCore.Qt.black, 1.0))
        self.series.setUseOpenGL(True)  # Speed for many line?
        self.addSeries(self.series)
        self.series.attachAxis(self.xaxis)
        self.series.attachAxis(self.yaxis)

        self.hlines: List[QtCharts.QLineSeries] = []
        self.vlines: List[QtCharts.QLineSeries] = []

        # Clean legend
        self.legend().setMarkerShape(QtCharts.QLegend.MarkerShapeFromSeries)
        self.legend().markers(self.series)[0].setVisible(False)

    def clearHorizontalLines(self) -> None:
        for line in self.hlines:
            self.removeSeries(line)
        self.hlines.clear()

    def clearVerticalLines(self) -> None:
        for line in self.vlines:
            self.removeSeries(line)
        self.vlines.clear()

    def setData(self, points: np.ndarray) -> None:
        self.yvalues = points[:, 1]
        poly = array_to_polygonf(points)
        self.series.replace(poly)

    def drawHorizontalLines(
        self,
        values: List[float],
        colors: List[QtGui.QColor] = None,
        names: List[str] = None,
        styles: List[QtCore.Qt.PenStyle] = None,
        visible_in_legend: bool = True,
    ) -> None:

        # Clear lines
        self.clearHorizontalLines()

        for i, value in enumerate(values):
            pen = QtGui.QPen()
            if colors is not None:
                pen.setColor(colors[i])
            if styles is not None:
                pen.setStyle(styles[i])
            line = QtCharts.QLineSeries()
            line.setPen(pen)
            if names is not None:
                line.setName(names[i])

            self.addSeries(line)
            line.attachAxis(self.xaxis)
            line.attachAxis(self.yaxis)
            if not visible_in_legend:
                self.legend().markers(line)[0].setVisible(False)
            self.hlines.append(line)

        self.setHorizontalLines(values)

    def drawVerticalLines(
        self,
        values: List[float],
        colors: List[QtGui.QColor] = None,
        names: List[str] = None,
        styles: List[QtCore.Qt.PenStyle] = None,
        visible_in_legend: bool = True,
    ) -> None:
        self.clearVerticalLines()

        for i, value in enumerate(values):
            pen = QtGui.QPen()
            if colors is not None:
                pen.setColor(colors[i])
            if styles is not None:
                pen.setStyle(styles[i])
            line = QtCharts.QLineSeries()
            line.setPen(pen)
            if names is not None:
                line.setName(names[i])

            self.addSeries(line)
            line.attachAxis(self.xaxis)
            line.attachAxis(self.yaxis)

            if not visible_in_legend:
                self.legend().markers(line)[0].setVisible(False)
            self.vlines.append(line)

        self.setVerticalLines(values)

    def setHorizontalLines(self, values: List[float]) -> None:
        xmin, xmax = 0, self.series.count()
        for line, value in zip(self.hlines, values):
            line.replace([QtCore.QPointF(xmin, value), QtCore.QPointF(xmax, value)])
        self.update()

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

        ymax = np.max(self.yvalues[int(xmin) : int(xmax)])
        self.yaxis.setRange(0.0, ymax)


class ParticleHistogram(QtCharts.QChart):
    def __init__(self, parent: QtWidgets.QGraphicsItem = None):
        super().__init__(parent)
        self.setMinimumSize(640, 320)

        self._xaxis = QtCharts.QValueAxis()
        self._xaxis.setVisible(False)
        self.xaxis = QtCharts.QValueAxis()
        self.xaxis.setTitleText("Size (nm)")
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

        self.set = QtCharts.QBarSet("sizes")
        self.set.setColor(QtCore.Qt.black)
        self.set.hovered.connect(self.barHovered)
        self.series.append(self.set)

        self.label_series = QtCharts.QScatterSeries()
        self.label_series.append(0, 0)
        self.label_series.setBrush(QtGui.QBrush(QtCore.Qt.black, QtCore.Qt.NoBrush))
        self.label_series.setPen(QtCore.Qt.NoPen)
        font = QtGui.QFont("sans", 12)
        self.label_series.setPointLabelsFont(font)
        self.label_series.setPointLabelsColor(QtGui.QColor(255, 128, 128))
        self.label_series.setPointLabelsFormat("(@xPoint, @yPoint)")

        self.addSeries(self.label_series)
        self.label_series.attachAxis(self.xaxis)
        self.label_series.attachAxis(self.yaxis)

        self.fit = QtCharts.QLineSeries()
        self.addSeries(self.fit)
        self.fit.attachAxis(self.xaxis)
        self.fit.attachAxis(self.yaxis)

        self.vlines: List[QtCharts.QLineSeries] = []

        # Clean legend
        self.legend().setMarkerShape(QtCharts.QLegend.MarkerShapeFromSeries)
        self.legend().markers(self.series)[0].setVisible(False)
        self.legend().markers(self.label_series)[0].setVisible(False)

    def clearVerticalLines(self) -> None:
        for line in self.vlines:
            self.removeSeries(line)
        self.vlines.clear()

    def drawVerticalLines(
        self,
        values: List[float],
        colors: List[QtGui.QColor] = None,
        names: List[str] = None,
        styles: List[QtCore.Qt.PenStyle] = None,
        visible_in_legend: bool = True,
    ) -> None:
        self.clearVerticalLines()

        for i, value in enumerate(values):
            pen = QtGui.QPen()
            if colors is not None:
                pen.setColor(colors[i])
            if styles is not None:
                pen.setStyle(styles[i])
            line = QtCharts.QLineSeries()
            line.setPen(pen)
            if names is not None:
                line.setName(names[i])

            self.addSeries(line)
            line.attachAxis(self.xaxis)
            line.attachAxis(self.yaxis)

            if not visible_in_legend:
                self.legend().markers(line)[0].setVisible(False)
            self.vlines.append(line)

        self.setVerticalLines(values)

    def setVerticalLines(self, values: List[float]) -> None:
        ymin, ymax = self.yaxis.min(), self.yaxis.max()
        for line, value in zip(self.vlines, values):
            line.replace([QtCore.QPointF(value, ymin), QtCore.QPointF(value, ymax)])
        self.update()

    def setData(self, data: np.ndarray, bins: np.ndarray) -> None:
        self.set.remove(0, self.set.count())
        self.set.append(list(data))

        self._xaxis.setRange(-0.5, data.size - 0.5)
        self.xaxis.setRange(bins[0], bins[-1])
        self.yaxis.setRange(0, np.amax(data))
        self.yaxis.applyNiceNumbers()

    def setFit(self, bins: np.ndarray, fit: np.ndarray) -> None:
        poly = array_to_polygonf(np.stack((bins, fit), axis=1))
        self.fit.replace(poly)
        pass

    def barHovered(self, state: bool, index: int) -> None:
        self.label_series.setPointLabelsVisible(state)
        x = np.round(index * (self.xaxis.max() / self.set.count()), 2)
        y = self.set.at(index)
        if state:
            self.label_series.replace(0, x, y)
