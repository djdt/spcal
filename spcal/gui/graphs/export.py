import numpy as np
from PySide6 import QtCore, QtGui

from spcal.detection import detection_maxima
from spcal.gui.graphs import color_schemes, symbols
from spcal.gui.graphs.particle import ParticleView
from spcal.result import SPCalResult


def draw_particle_view(
    results: dict[str, SPCalResult],
    regions: np.ndarray,
    dwell: float,
    font: QtGui.QFont,
    scale: float,
    draw_markers: bool = False,
) -> ParticleView:

    graph = ParticleView(xscale=dwell, font=font)

    scheme = color_schemes[QtCore.QSettings().value("colorscheme", "IBM Carbon")]

    names = tuple(results.keys())
    xs = np.arange(results[names[0]].events)

    for name, result in results.items():
        index = names.index(name)
        pen = QtGui.QPen(QtGui.QColor(scheme[index % len(scheme)]), 2.0 * scale)
        pen.setCosmetic(True)
        graph.drawSignal(name, xs, result.responses, pen=pen)

    if draw_markers:
        for name, result in results.items():
            index = names.index(name)
            brush = QtGui.QBrush(QtGui.QColor(scheme[index % len(scheme)]))
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
                size=6.0 * scale,
            )
    return graph

def draw_histogram_view(
    results: dict[str, SPCalResult],
        )

        for i, (name, data) in enumerate(graph_data.items()):
            color = self.colorForName(name)
            max_bin = np.percentile(
                data[graph_idx[name]], self.graph_options["histogram"]["percentile"]
            )
            bins = np.arange(
                data.min() * modifier, max_bin * modifier + bin_width, bin_width
            )
            if bins.size == 1:
                bins = np.array([bins[0], max_bin])
            bins -= bins[0] % bin_width  # align bins
            if self.graph_options["histogram"]["mode"] == "overlay":
                width = 1.0 / len(graph_data)
                offset = i * width
            elif self.graph_options["histogram"]["mode"] == "single":
                width = 1.0
                offset = 0.0
            else:
                raise ValueError("drawGraphHist: invalid draw mode")

            lod = self.results[name].convertTo(
                self.results[name].limits.detection_limit, key
            )

            non_zero = np.flatnonzero(data)

            limits = {"mean": np.mean(data[graph_idx[name]]) * modifier}
            if isinstance(lod, np.ndarray):
                limits["LOD (mean)"] = np.nanmean(lod) * modifier
            else:
                limits["LOD"] = lod * modifier

            # Auto SI prefix does not work with squared (or cubed) units
            self.graph_hist.xaxis.enableAutoSIPrefix(mode not in ["Signal", "Volume"])

            self.graph_hist.xaxis.setLabel(text=label, units=unit)
            self.graph_hist.draw(
                data[graph_idx[name]] * modifier,
                filtered_data=data[np.setdiff1d(non_zero, idx, assume_unique=True)]
                * modifier,
                bins=bins,
                bar_width=width,
                bar_offset=offset,
                brush=QtGui.QBrush(color),
                name=name,
                draw_fit=self.graph_options["histogram"]["fit"],
                fit_visible=self.graph_options["histogram"]["mode"] == "single",
                draw_limits=limits,
                limits_visible=self.graph_options["histogram"]["mode"] == "single",
                draw_filtered=self.graph_options["histogram"]["draw filtered"],
            )
