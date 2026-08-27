import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.util import text_for_mz


class SingleIonAreaScatterPlot(pyqtgraph.ScatterPlotItem):
    pointHovered = QtCore.Signal(QtCore.QPointF, int)
    pointClicked = QtCore.Signal(QtCore.QPointF, int)

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
    ):
        super().__init__(x=x, y=y, pen=pen, brush=brush)
        self.setAcceptHoverEvents(True)
        self.opts["mouseWidth"] = 50.0

        self.label = pyqtgraph.TextItem(anchor=(0.5, 1))
        self.label.setParentItem(self)
        self.label.setVisible(False)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        points: list[pyqtgraph.SpotItem] = self.pointsAt(event.pos())
        if len(points) > 0:
            self.pointClicked.emit(points[0].pos(), points[0].index())

    def hoverMoveEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent):
        points: list[pyqtgraph.SpotItem] = self.pointsAt(event.pos())
        if len(points) == 0:
            self.label.setVisible(False)
            return

        self.label.setPos(points[0].pos())
        self.label.setText(text_for_mz(points[0].pos().x()))
        self.label.setVisible(True)

        self.pointHovered.emit(
            QtCore.QPointF(points[0].pos().x(), points[0].pos().y()),
            int(points[0].index()),
        )


class SingleIonAreaScatterView(SinglePlotGraphicsView):
    pointHovered = QtCore.Signal(QtCore.QPointF, int)
    pointClicked = QtCore.Signal(QtCore.QPointF, int)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(
            "Extracted Parameters",
            xlabel="m/z",
            ylabel="Shape (σ)",
            parent=parent,
        )
        self.plot.yaxis.autoSIPrefix = False

        self.points: SingleIonAreaScatterPlot | None = None
        self.lines: dict[str, pyqtgraph.PlotCurveItem] = {}
        self.plot.getViewBox().setLimits(xMin=0.0, yMin=0.0)

    #     self.pointHovered.connect(self.onPointHovered)
    #
    # def onPointHovered(self, pos: QtCore.QPointF, index: int):
    #     self.label.setPos(pos)

    def clear(self):
        super().clear()
        self.points = None
        self.lines.clear()

    def drawData(
        self,
        x: np.ndarray,
        y: np.ndarray,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
    ):
        if self.points is not None:
            self.plot.removeItem(self.points)

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0)
            pen.setCosmetic(True)
        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.GlobalColor.black)

        self.points = SingleIonAreaScatterPlot(x=x, y=y, pen=pen, brush=brush)
        self.points.pointHovered.connect(self.pointHovered)
        self.points.pointClicked.connect(self.pointClicked)
        self.plot.addItem(self.points)

        self.setDataLimits(-0.05, 1.05, -0.05, 1.05)

    # def setColors(self, indicies: np.ndarray, brushes: list[QtGui.QBrush]):
    #     if self.points is None:
    #         return
    #
    #     self.points.setBrush([brushes[i] for i in indicies])
    #
    # def setValid(self, valid: np.ndarray):
    #     if self.points is None:
    #         return
    #     brush_valid = QtGui.QBrush(QtCore.Qt.GlobalColor.black)
    #     brush_invalid = QtGui.QBrush(QtCore.Qt.GlobalColor.red)
    #     brushes = [brush_valid if x else brush_invalid for x in valid]
    #     self.points.setBrush(brushes)

    def drawMaxDifference(
        self,
        poly: np.polynomial.Polynomial,
        max_difference: float,
        pen: QtGui.QPen | None = None,
    ):
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.red, 1.0)
            pen.setCosmetic(True)

        if "max_diff" not in self.lines:
            max_diff = pyqtgraph.PlotCurveItem(pen=pen, connect="pairs", antialias=True)
            self.plot.addItem(max_diff)
            self.lines["max_diff"] = max_diff

        xs = [poly.domain[0], poly.domain[-1], poly.domain[0], poly.domain[-1]]
        ys = poly(xs)
        ys += [
            max_difference,
            max_difference,
            -max_difference,
            -max_difference,
        ]
        self.lines["max_diff"].setData(x=xs, y=ys)
        self.lines["max_diff"].setPen(pen)

    def drawInterpolationLine(
        self, xs: np.ndarray, ys: np.ndarray, pen: QtGui.QPen | None = None
    ):
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.blue, 1.0)
            pen.setCosmetic(True)

        if "interp" not in self.lines:
            interp = pyqtgraph.PlotCurveItem(x=xs, y=ys, pen=pen, skipFiniteCheck=True)
            self.plot.addItem(interp)
            self.lines["interp"] = interp
        else:
            self.lines["interp"].setData(x=xs, y=ys)
