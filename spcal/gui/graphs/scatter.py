import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.graphs.base import SinglePlotGraphicsView

from spcal.pratt import Parser, ParserException, Reducer, ReducerException
from spcal.processing.result import SPCalProcessingResult


class PolygonSelectionItem(QtWidgets.QGraphicsWidget):
    def __init__(
        self,
        pen: QtGui.QPen,
        brush: QtGui.QBrush | None = None,
        parent: QtWidgets.QGraphicsObject | None = None,
    ):
        super().__init__(parent)

        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush)

        self.pen = pen
        self.brush = brush

        self.poly = QtGui.QPolygonF()

    def boundingRect(self) -> QtCore.QRectF:
        return self.poly.boundingRect()

    def shape(self) -> QtGui.QPainterPath:
        path = QtGui.QPainterPath()
        path.addPolygon(self.poly)
        return path

    def addPoint(self, point: QtCore.QPointF):
        self.poly.append(point)

    def containsPoints(self, points: np.ndarray) -> np.ndarray:
        """Test an array of points, shape (N, 2).

        This is a much lazier implementation than in pewpew."""

        rect = self.boundingRect()

        valid = np.logical_and(
            np.logical_and(points[:, 0] > rect.left(), points[:, 0] < rect.right()),
            np.logical_and(points[:, 1] > rect.bottom(), points[:, 1] < rect.top()),
        )
        test_idx = np.flatnonzero(~valid)

        valid_test_points = [
            self.poly.containsPoint(
                QtCore.QPointF(points[i][0], points[i][1]),
                QtCore.Qt.FillRule.OddEvenFill,
            )
            for i in test_idx
        ]
        valid[test_idx] = valid_test_points
        return valid

    def paint(
        self,
        painter: QtGui.QPainter,
        options: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ):
        painter.save()
        painter.setPen(self.pen)
        painter.setBrush(self.brush)
        painter.drawConvexPolygon(self.poly)
        painter.restore()


class ScatterView(SinglePlotGraphicsView):
    def __init__(
        self, font: QtGui.QFont | None = None, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__("Scatter", font=font, parent=parent)

        self.poly: PolygonSelectionItem | None = None

    def drawArrays(
        self,
        x: np.ndarray,
        y: np.ndarray,
        label_x: str,
        label_y: str,
        unit_x: str = "",
        unit_y: str = "",
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
    ):
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0)
            pen.setCosmetic(True)
        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.GlobalColor.black)

        valid = np.logical_and(x != 0, y != 0)

        self.data_for_export[f"x_{label_x}"] = x[valid]
        self.data_for_export[f"y_{label_y}"] = y[valid]

        self.drawScatter(x[valid], y[valid], pen=pen, brush=brush)

        self.plot.xaxis.setLabel(label_x, unit_x)
        self.plot.yaxis.setLabel(label_y, unit_y)

        self.setDataLimits(-0.05, 1.05, -0.05, 1.05)
        self.zoomReset()

    # def drawResults(
    #     self,
    #     result_x: SPCalProcessingResult,
    #     result_y: SPCalProcessingResult,
    #     key: str,
    #     pen: QtGui.QPen | None = None,
    #     brush: QtGui.QBrush | None = None,
    # ):
    #     if result_x.peak_indicies is None or result_y.peak_indicies is None:
    #         raise ValueError("peak_indicies have not been generated")
    #
    #     npeaks = result_x.number_peak_indicies
    #     x, y = np.zeros(npeaks, dtype=np.float32), np.zeros(npeaks, dtype=np.float32)
    #     np.add.at(
    #         x,
    #         result_x.peak_indicies[result_x.filter_indicies],
    #         result_x.calibrated(key),
    #     )
    #     np.add.at(
    #         y,
    #         result_y.peak_indicies[result_y.filter_indicies],
    #         result_y.calibrated(key),
    #     )
    #
    #     self.drawArrays(x, y, str(result_x.isotope), str(result_y.isotope), pen, brush)

    def drawResultsExpr(
        self,
        results: list[SPCalProcessingResult],
        text_x: str,
        text_y: str,
        key_x: str,
        key_y: str,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
    ):
        parser = Parser(variables=[str(result.isotope) for result in results])

        def get_reduction(text: str, key: str) -> np.ndarray | None:
            reducer = Reducer()
            reducer.variables = {
                str(result.isotope): result.calibrateTo(result.peakValues(), key)
                for result in results
                if result.canCalibrate(key)
            }
            try:
                expr = parser.parse(text)
                x = reducer.reduce(expr)
                if isinstance(x, np.ndarray):
                    return x
                else:
                    return None
            except (ReducerException, ParserException):
                return None

        x = get_reduction(text_x, key_x)
        y = get_reduction(text_y, key_y)

        if x is None or y is None:
            return

        self.drawArrays(x, y, text_x, text_y)

    # def mousePressEvent(self, event: QtGui.QMouseEvent):  # type: ignore , more pyqtgraph
    #     if event.button() == QtCore.Qt.MouseButton.LeftButton:
    #         # self.band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Shape.Rectangle, self)
    #         # self.band.setGeometry(QtCore.QRect(event.pos(), QtCore.QSize(0,0)))
    #         # # pen = QtGui.QPen(QtCore.Qt.GlobalColor.red, 0.0)
    #         # # rb
    #         # #
    #         # # self.poly = PolygonSelectionItem(pen)
    #         # # self.plot.addItem(self.poly)
    #         # # self.poly.addPoint(event.pos())
    #         self.setDragMode(QtWidgets.QGraphicsView.DragMode.RubberBandDrag)
    #
    #         event.accept()
    #     else:
    #         super().mousePressEvent(event)

    # def mouseMoveEvent(self, event: QtGui.QMouseEvent):
    #     if event.button() == QtCore.Qt.MouseButton.LeftButton and self.band is not None:
    #         self.band.setGeometry(
    #             QtCore.QRect(self.band.geometry().topLeft(), event.pos())
    #         )
    #     super().mouseMoveEvent(event)
    #
    # def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
    #     if self.dragMode() == QtWidgets.QGraphicsView.DragMode.RubberBandDrag:
    #         rect = self.rubberBandRect().normalized()
    #     if self.poly is not None:
    #         self.removeItem

    def drawFit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        degree: int = 1,
        logx: bool = False,
        logy: bool = False,
        weighting: str = "none",
        set_title: bool = True,
        pen: QtGui.QPen | None = None,
    ):
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.red, 1.0)
            pen.setCosmetic(True)

        if weighting == "none":
            w = np.ones_like(x)
        elif weighting == "1/x":
            w = 1.0 / x
        elif weighting == "1/x²":
            w = 1.0 / (x**2)
        elif weighting == "1/y":
            w = 1.0 / y
        elif weighting == "1/y²":
            w = 1.0 / (y**2)
        else:
            raise ValueError(f"drawFit: unknown weighting '{weighting}'")

        poly = np.polynomial.Polynomial.fit(x, y, degree, w=w / w.sum())
        poly = poly.convert()

        xmin, xmax = np.amin(x), np.amax(x)
        sx = np.linspace(xmin, xmax, 1000)

        sy = poly(sx)

        if logx:
            sx = np.log10(sx)
        if logy:
            sy = np.log10(sy)

        curve = pyqtgraph.PlotCurveItem(
            x=sx, y=sy, pen=pen, connect="all", skipFiniteCheck=True
        )
        self.plot.addItem(curve)

        if set_title:
            self.plot.setTitle(f"{x.size} points; y = {poly}")
