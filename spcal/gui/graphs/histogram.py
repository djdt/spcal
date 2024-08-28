import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.fit import fit_lognormal, fit_normal, lognormal_pdf, normal_pdf
from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.legends import HistogramItemSample
from spcal.gui.graphs.viewbox import ViewBoxForceScaleAtZero


class HistogramView(SinglePlotGraphicsView):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(
            "Histogram",
            xlabel="Signal (counts)",
            ylabel="No. Events",
            viewbox=ViewBoxForceScaleAtZero(),
            parent=parent,
        )
        self.plot.setLimits(xMin=0.0, yMin=0.0)

    def draw(
        self,
        data: np.ndarray,
        filtered_data: np.ndarray,
        bins: str | np.ndarray = "auto",
        bar_width: float = 0.5,
        bar_offset: float = 0.0,
        name: str | None = None,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
        draw_fit: str | None = None,
        draw_limits: dict[str, float] | None = None,
        fit_visible: bool = True,
        limits_visible: bool = True,
        draw_filtered: bool = False,
    ) -> None:
        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.black)

        hist, edges = np.histogram(data, bins)
        curve = self.drawData(
            hist,
            edges,
            bar_width=bar_width,
            bar_offset=bar_offset,
            pen=pen,
            brush=brush,
        )
        if draw_filtered:
            hist_filt, edges_filt = np.histogram(filtered_data, bins)
            curve_filt = self.drawData(
                hist_filt,
                edges_filt,
                bar_width=bar_width,
                bar_offset=bar_offset,
                pen=pen,
                brush=QtGui.QBrush(QtGui.QColor(128, 128, 128, 128)),
            )
            legend = HistogramItemSample([curve, curve_filt])
        else:
            legend = HistogramItemSample([curve])

        if draw_fit is not None:
            pen = QtGui.QPen(brush.color().darker(), 2.0)
            pen.setCosmetic(True)
            curve = self.drawFit(
                hist, edges, data.size, fit_type=draw_fit, pen=pen, visible=fit_visible
            )
            legend.setFit(curve)

        if draw_limits is not None:
            pen = QtGui.QPen(brush.color().darker(), 2.0)
            pen.setCosmetic(True)
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
            for label, lim in draw_limits.items():
                limit = self.drawLimit(lim, label, pen=pen, visible=limits_visible)
                legend.addLimit(limit)

        self.plot.legend.addItem(legend, name)
        if name is not None:
            self.export_data[name] = hist
            self.export_data[name + "_bins"] = edges[1:]
            if draw_filtered:
                self.export_data[name + "_filtered"] = hist_filt
                self.export_data[name + "_filtered_bins"] = edges_filt[1:]

    def drawData(
        self,
        hist: np.ndarray,
        edges: np.ndarray,
        bar_width: float = 0.5,
        bar_offset: float = 0.0,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
    ) -> pyqtgraph.PlotCurveItem:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)

        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.black)

        assert bar_width > 0.0 and bar_width <= 1.0
        assert bar_offset >= 0.0 and bar_offset < 1.0

        widths = np.diff(edges)

        x = np.repeat(edges, 2)

        # Calculate bar start and end points for width / offset
        x[1:-1:2] += widths * ((1.0 - bar_width) / 2.0 + bar_offset)
        x[2::2] -= widths * ((1.0 - bar_width) / 2.0 - bar_offset)

        y = np.zeros(hist.size * 2 + 1, dtype=hist.dtype)
        y[1:-1:2] = hist

        curve = pyqtgraph.PlotCurveItem(
            x=x,
            y=y,
            stepMode="center",
            fillLevel=0,
            fillOutline=True,
            pen=pen,
            brush=brush,
            skipFiniteCheck=True,
        )

        self.plot.addItem(curve)
        return curve

    def drawFit(
        self,
        hist: np.ndarray,
        edges: np.ndarray,
        size: int,
        fit_type: str = "log normal",
        visible: bool = True,
        pen: QtGui.QPen | None = None,
    ) -> pyqtgraph.PlotCurveItem:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)

        centers = (edges[:-1] + edges[1:]) / 2.0
        bin_width = edges[1] - edges[0]
        # scale for density
        hist = hist / bin_width / size
        xs = np.linspace(centers[0] - bin_width, centers[-1] + bin_width, 1024)
        if fit_type == "normal":
            fit = fit_normal(centers, hist)
            ys = normal_pdf(xs * fit[2], fit[0], fit[1])
        elif fit_type == "log normal":
            fit = fit_lognormal(centers, hist)
            ys = lognormal_pdf(xs + fit[2], fit[0], fit[1])
        else:
            raise ValueError(f"drawFit: unknown fit {fit_type}")

        # rescale
        ys = ys * bin_width * size
        curve = pyqtgraph.PlotCurveItem(
            x=xs,
            y=np.nan_to_num(ys),
            pen=pen,
            connect="all",
            skipFiniteCheck=True,
            antialias=True,
        )
        curve.setVisible(visible)
        self.plot.addItem(curve)
        return curve

    def drawLimit(
        self,
        limit: float,
        label: str,
        name: str | None = None,
        pos: float = 0.95,
        pen: QtGui.QPen | None = None,
        visible: bool = True,
    ) -> pyqtgraph.InfiniteLine:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 2.0, QtCore.Qt.PenStyle.DashLine)
            pen.setCosmetic(True)

        line = pyqtgraph.InfiniteLine(
            (limit, 1.0),  # This bypasses the fuzzy comparision in QGraphicsItem
            label=label,
            labelOpts={"position": pos, "color": "black"},
            pen=pen,
        )
        line.setVisible(visible)
        self.plot.addItem(line)
        return line
