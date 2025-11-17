import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.legends import HistogramItemSample
from spcal.gui.graphs.viewbox import ViewBoxForceScaleAtZero
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
            xlabel="Signal (cts)",
            ylabel="No. Events",
            viewbox=ViewBoxForceScaleAtZero(),
            font=font,
            parent=parent,
        )
        self.has_image_export = True
        assert self.plot.vb is not None
        self.plot.vb.setLimits(xMin=0.0, yMin=0.0)

        # options
        self.bin_widths: dict[str, float | None] = {}
        self.max_percentile = 95.0
        self.draw_filtered = True

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
    ):
        label, unit, mult = SinglePlotGraphicsView.UNIT_LABELS[key]

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0)
            pen.setCosmetic(True)
        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.GlobalColor.red)

        if not result.canCalibrate(key):
            return

        signals = result.calibrated(key, filtered=False) * mult
        mask = np.zeros(signals.size, dtype=bool)
        mask[result.filter_indicies] = True
        counts, edges = np.histogram(signals[mask], bins, range=range)

        curve = self.drawHistogram(
            counts, edges, width=width, offset=offset, pen=pen, brush=brush
        )

        if self.plot.legend is not None:
            fm = self.fontMetrics()
            legend = HistogramItemSample([curve], size=fm.height())  # type: ignore , curce is subtype of dataitem
            self.plot.legend.addItem(legend, str(result.isotope))

        if self.draw_filtered:
            counts, edges = np.histogram(signals[~mask], bins, range=range)

            filtered_brush = QtGui.QBrush(brush)
            filtered_brush.setColor(QtCore.Qt.GlobalColor.gray)

            curve = self.drawHistogram(
                counts, edges, width=width, offset=offset, pen=pen, brush=filtered_brush
            )

        self.plot.xaxis.setLabel(label, unit)
        self.setDataLimits(xMax=1.0)

    def drawResults(
        self,
        results: list[SPCalProcessingResult],
        key: str = "signal",
        pen: QtGui.QPen | None = None,
        brushes: list[QtGui.QBrush] | None = None,
    ):
        label, unit, mult = SinglePlotGraphicsView.UNIT_LABELS[key]
        # Limit maximum / minimum number of bins
        values = [result.calibrated(key) * mult for result in results]
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
