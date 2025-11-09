import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.dists import lognormal, normal
from spcal.fit import fit_lognormal, fit_normal
from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.legends import HistogramItemSample
from spcal.gui.graphs.viewbox import ViewBoxForceScaleAtZero
from spcal.isotope import SPCalIsotope
from spcal.processing import SPCalProcessingResult


def bins_for_values(
    values: list[np.ndarray],
    width: float | None = None,
    percentile_max: float = 98.0,
    min_bins: int = 10,
    max_bins: int = 1000,
):
    percentiles = np.stack(
        [np.percentile(x, [0, 25, 75, percentile_max]) for x in values], axis=0
    )
    sizes = np.array([x.size for x in values])
    if width is None:
        width = float(
            np.median([2.0 * (percentiles[:, 2] - percentiles[:, 1]) / np.cbrt(sizes)])
        )
    data_range = np.amin(percentiles[:, 0]), np.amax(percentiles[:, 3])

    while (data_range[1] - data_range[0]) / width > max_bins:
        width *= 2
    while (data_range[1] - data_range[0]) / width < min_bins:
        width /= 2

    bins = np.arange(data_range[0], data_range[1], width)
    bins -= bins[0] % width
    return bins


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

        self.bin_widths: dict[str, float] = {}
        self.max_percentile = 95.0

    # def drawFit(
    #     self,
    #     hist: np.ndarray,
    #     edges: np.ndarray,
    #     size: int,
    #     fit_type: str = "log normal",
    #     visible: bool = True,
    #     pen: QtGui.QPen | None = None,
    # ) -> pyqtgraph.PlotCurveItem:
    #     if pen is None:
    #         pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0)
    #         pen.setCosmetic(True)
    #
    #     centers = (edges[:-1] + edges[1:]) / 2.0
    #     bin_width = edges[1] - edges[0]
    #     # scale for density
    #     hist = hist / bin_width / size
    #     xs = np.linspace(centers[0] - bin_width, centers[-1] + bin_width, 1024)
    #     if fit_type == "normal":
    #         fit = fit_normal(centers, hist)
    #         ys = normal.pdf(xs * fit[2], fit[0], fit[1])
    #     elif fit_type == "log normal":
    #         fit = fit_lognormal(centers, hist)
    #         xs_fit = xs - fit[2]
    #         xs_fit[xs_fit <= 0.0] = np.nan
    #         ys = lognormal.pdf(xs_fit, fit[0], fit[1])
    #     else:
    #         raise ValueError(f"unknown fit {fit_type}")
    #
    #     # rescale
    #     ys = ys * bin_width * size
    #     curve = pyqtgraph.PlotCurveItem(
    #         x=xs,
    #         y=np.nan_to_num(ys),
    #         pen=pen,
    #         connect="all",
    #         skipFiniteCheck=True,
    #         antialias=True,
    #     )
    #     curve.setVisible(visible)
    #     self.plot.addItem(curve)
    #     return curve

    def drawResult(
        self,
        result: SPCalProcessingResult,
        key: str = "signal",
        bins: str | np.ndarray = "auto",
        range: tuple[float, float] | None = None,
        width: float = 1.0,
        offset: float = 0.0,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
        draw_fit: bool = False,
    ):
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0)
            pen.setCosmetic(True)
        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.GlobalColor.red)

        if not result.canCalibrate(key):
            return

        signals = result.calibrated(key)
        counts, edges = np.histogram(signals, bins, range=range)

        curve = self.drawHistogram(
            counts, edges, width=width, offset=offset, pen=pen, brush=brush
        )

        if self.plot.legend is not None:
            fm = self.fontMetrics()
            legend = HistogramItemSample([curve], size=fm.height())
            self.plot.legend.addItem(legend, str(result.isotope))

        self.setDataLimits(xMax=1.0)

    def drawResults(
        self,
        results: list[SPCalProcessingResult],
        key: str = "signal",
        pen: QtGui.QPen | None = None,
        brushes: list[QtGui.QBrush] | None = None,
    ):
        # Limit maximum / minimum number of bins
        values = [result.calibrated(key) for result in results]
        bins = bins_for_values(
            values, self.bin_widths.get(key, None), self.max_percentile
        )

        for i, result in enumerate(results):
            brush = brushes[i] if brushes is not None else None
            # width = 1.0 / len(results)
            self.drawResult(
                result,
                key,
                bins,
                width=1.0,
                offset=0.0,
                pen=pen,
                brush=brush,
            )
        self.zoomReset()
