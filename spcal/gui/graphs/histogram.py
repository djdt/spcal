import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.dists import lognormal, normal
from spcal.fit import fit_lognormal, fit_normal
from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.legends import HistogramItemSample
from spcal.gui.graphs.viewbox import ViewBoxForceScaleAtZero
from spcal.processing import SPCalProcessingResult


class HistogramView(SinglePlotGraphicsView):
    def __init__(
        self, font: QtGui.QFont | None = None, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(
            "Histogram",
            xlabel="Signal (counts)",
            ylabel="No. Events",
            viewbox=ViewBoxForceScaleAtZero(),
            font=font,
            parent=parent,
        )
        self.has_image_export = True
        assert self.plot.vb is not None
        self.plot.vb.setLimits(xMin=0.0, yMin=0.0)

    def drawResult(
        self,
        result: SPCalProcessingResult,
        key: str = "signal",
        bins: str | np.ndarray = "auto",
        width: float = 1.0,
        offset: float = 0.0,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
        scatter_size: float = 6.0,
        scatter_symbol: str = "t",
    ):
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0)
            pen.setCosmetic(True)
        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.GlobalColor.red)

        if not result.canCalibrate(key):
            return

        signals = result.calibrated(key)
        counts, edges = np.histogram(signals, bins)

        curve = self.drawHistogram(
            counts, edges, width=width, offset=offset, pen=pen, brush=brush
        )

        if self.plot.legend is not None:
            fm = self.fontMetrics()
            legend = HistogramItemSample([curve], size=fm.height())
            self.plot.legend.addItem(legend, str(result.isotope))

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
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.black)
            pen.setCosmetic(True)
        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.GlobalColor.black)

        fm = QtGui.QFontMetrics(self.font)

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
            legend = HistogramItemSample([curve, curve_filt], size=fm.height())
        else:
            legend = HistogramItemSample([curve], size=fm.height())

        if draw_fit is not None:
            fit_pen = QtGui.QPen(brush.color().darker(), 2.0 * pen.widthF())
            fit_pen.setCosmetic(True)
            curve = self.drawFit(
                hist,
                edges,
                data.size,
                fit_type=draw_fit,
                pen=fit_pen,
                visible=fit_visible,
            )
            legend.setFit(curve)

        if draw_limits is not None:
            lim_pen = QtGui.QPen(brush.color().darker(), 2.0 * pen.widthF())
            lim_pen.setCosmetic(True)
            lim_pen.setStyle(QtCore.Qt.PenStyle.DashLine)
            for label, lim in draw_limits.items():
                limit = self.drawLimit(lim, label, pen=lim_pen, visible=limits_visible)
                legend.addLimit(limit)

        self.plot.legend.addItem(legend, name)
        if name is not None:
            self.export_data[name] = hist
            self.export_data[name + "_bins"] = edges[1:]
            if draw_filtered:
                self.export_data[name + "_filtered"] = hist_filt
                self.export_data[name + "_filtered_bins"] = edges_filt[1:]

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
            ys = normal.pdf(xs * fit[2], fit[0], fit[1])
        elif fit_type == "log normal":
            fit = fit_lognormal(centers, hist)
            xs_fit = xs - fit[2]
            xs_fit[xs_fit <= 0.0] = np.nan
            ys = lognormal.pdf(xs_fit, fit[0], fit[1])
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
