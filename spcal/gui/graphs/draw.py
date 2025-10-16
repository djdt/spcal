import logging

import numpy as np
from PySide6 import QtCore, QtGui

from spcal.detection import detection_maxima
from spcal.gui.graphs import color_schemes, symbols
from spcal.gui.graphs.histogram import HistogramView
from spcal.gui.graphs.particle import ParticleView
from spcal.result import SPCalResult

logger = logging.getLogger(__name__)


def draw_particle_view(
    graph: ParticleView | None,
    results: dict[str, SPCalResult],
    regions: np.ndarray,
    dwell: float,
    show_markers: bool = False,
    font: QtGui.QFont | None = None,
    scale: float = 1.0,
    colors: list[QtGui.QColor] | None = None,
) -> ParticleView:
    if graph is None:
        graph = ParticleView(xscale=dwell, font=font)

    if colors is None:
        scheme = color_schemes[QtCore.QSettings().value("colorscheme", "IBM Carbon")]
        colors = [QtGui.QColor(scheme[i % len(scheme)]) for i in range(len(results))]

    names = tuple(results.keys())
    xs = np.arange(results[names[0]].events)

    for color, (name, result) in zip(colors, results.items()):
        index = names.index(name)
        pen = QtGui.QPen(color, 1.2 * scale)
        pen.setCosmetic(True)
        graph.drawSignal(name, xs, result.responses, pen=pen)

    if show_markers:
        for color, (name, result) in zip(colors, results.items()):
            index = names.index(name)
            brush = QtGui.QBrush(color)
            symbol = symbols[index % len(symbols)]

            maxima = detection_maxima(
                result.responses, regions[np.flatnonzero(result.detections)]
            )
            graph.drawMaxima(
                name,
                xs[maxima],
                result.responses[maxima],
                brush=brush,
                symbol=symbol,
                size=8.0 * scale,
            )
    return graph


def draw_histogram_view(
    graph: HistogramView | None,
    results: dict[str, SPCalResult],
    key: str,
    mode_label: tuple[str, str, float],
    filter_idx: np.ndarray | None = None,
    histogram_options: dict | None = None,
    font: QtGui.QFont | None = None,
    scale: float = 1.0,
    colors: list[QtGui.QColor] | None = None,
    show_fit: bool = True,
    show_limits: bool = True,
) -> HistogramView | None:
    # defaults
    options: dict = {
        "draw filtered": False,
        "mode": "overlay",
        "fit": "log normal",
        "bin widths": {
            "signal": None,
            "mass": None,
            "size": None,
            "volume": None,
        },
        "percentile": 95.0,
    }

    if graph is None:
        graph = HistogramView(font=font)

    if colors is None:
        scheme = color_schemes[QtCore.QSettings().value("colorscheme", "IBM Carbon")]
        colors = [QtGui.QColor(scheme[i % len(scheme)]) for i in range(len(results))]

    if histogram_options is not None:
        options.update(histogram_options)

    if filter_idx is None:
        filter_idx = np.arange(next(iter(results.values())).detections.size)

    bin_width = options["bin widths"][key]
    label, unit, modifier = mode_label

    names = tuple(results.keys())
    graph_data = {
        k: np.maximum(0.0, v.calibrated(key, use_indicies=False))
        for k, v in results.items()
        if (k in names and v.canCalibrate(key))
    }

    graph_idx = {
        k: np.intersect1d(filter_idx, np.nonzero(v), assume_unique=True)
        for k, v in graph_data.items()
    }
    # remove filtered
    for k, v in graph_idx.items():
        if v.size == 0:
            graph_data.pop(k)

    if len(graph_data) == 0:
        return None

    # median FD bin width
    if bin_width is None:
        bin_width = np.median(
            [
                2.0
                * np.subtract(
                    *np.percentile(graph_data[name][graph_idx[name]], [75, 25])
                )
                / np.cbrt(graph_idx[name].size)
                for name in graph_data
            ]
        )
    # Limit maximum / minimum number of bins
    data_range = 0.0
    for name, data in graph_data.items():
        ptp = np.percentile(data[graph_idx[name]], options["percentile"]) - np.amin(
            data[graph_idx[name]]
        )
        if ptp > data_range:
            data_range = ptp

    if data_range == 0.0:  # prevent drawing if no range, i.e. one point
        return None

    min_bins, max_bins = 10, 1000
    if bin_width < data_range / max_bins:
        logger.warning(f"drawGraphHist: exceeded maximum bins, setting to {max_bins}")
        bin_width = data_range / max_bins
    elif bin_width > data_range / min_bins:
        logger.warning(f"drawGraphHist: less than minimum bins, setting to {min_bins}")
        bin_width = data_range / min_bins
    bin_width *= modifier  # convert to base unit (kg -> g)

    for i, (name, data) in enumerate(graph_data.items()):
        max_bin = np.percentile(data[graph_idx[name]], options["percentile"])
        min_bin = np.amin(data[data > 0.0])
        bins = np.arange(min_bin * modifier, max_bin * modifier + bin_width, bin_width)
        if bins.size == 1:
            bins = np.array([bins[0], max_bin])
        bins -= bins[0] % bin_width  # align bins
        if options["mode"] == "overlay":
            width = 1.0 / len(graph_data)
            offset = i * width
        elif options["mode"] == "single":
            width = 1.0
            offset = 0.0
        else:
            raise ValueError("drawGraphHist: invalid draw mode")

        lod = results[name].convertTo(results[name].limits.detection_limit, key)

        non_zero = np.flatnonzero(data)

        limits = {"mean": np.mean(data[graph_idx[name]]) * modifier}
        if isinstance(lod, np.ndarray):
            limits["LOD (mean)"] = np.nanmean(lod) * modifier
        else:
            limits["LOD"] = lod * modifier

        # Auto SI prefix does not work with squared (or cubed) units
        graph.xaxis.enableAutoSIPrefix(key not in ["signal", "volume"])
        graph.xaxis.setLabel(text=label, units=unit)

        pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, scale)
        pen.setCosmetic(True)

        graph.draw(
            data[graph_idx[name]] * modifier,
            filtered_data=data[np.setdiff1d(non_zero, filter_idx, assume_unique=True)]
            * modifier,
            bins=bins,
            bar_width=width,
            bar_offset=offset,
            pen=pen,
            brush=QtGui.QBrush(colors[i]),
            name=name,
            draw_fit=options["fit"],
            fit_visible=show_fit,
            draw_limits=limits,
            limits_visible=show_limits,
            draw_filtered=options["draw filtered"],
        )
    return graph
