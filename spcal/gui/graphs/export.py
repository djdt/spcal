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

# def draw_histogram_view(
#     results: dict[str, SPCalResult],
#         )
