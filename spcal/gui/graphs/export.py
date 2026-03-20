from PySide6 import QtCore, QtGui
import numpy as np
from pathlib import Path

from spcal.processing import CALIBRATION_KEYS
from spcal.processing.result import SPCalProcessingResult
from spcal.gui.graphs.base import SinglePlotGraphicsView


def export_histograms_for_results(
    results: list[SPCalProcessingResult],
    output_dir: Path,
    file_name: str,
    size: QtCore.QSize | None = None,
    dpi: int = 96,
    pen: QtGui.QPen | None = None,
    brush: QtGui.QBrush | None = None,
    font: QtGui.QFont | None = None,
    font_pen: QtGui.QPen | None = None,
    background_color: QtGui.QColor
    | QtCore.Qt.GlobalColor = QtCore.Qt.GlobalColor.white,
):
    if not output_dir.is_dir():
        raise FileNotFoundError("output_dir must be a directory")

    if font is None:
        font = QtGui.QFont()

    if pen is None:
        pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0)
        pen.setCosmetic(True)

    if font_pen is None:
        font_pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0)

    if brush is None:
        brush = QtGui.QBrush(QtCore.Qt.GlobalColor.darkGray)

    font_scale = dpi / QtGui.QFontMetrics(font).fontDpi()
    font.setPointSizeF(font.pointSize() * font_scale)

    view = SinglePlotGraphicsView("", xlabel="Signal", ylabel="Count", font=font)
    font_pen.setCosmetic(True)
    for axis in [view.plot.xaxis, view.plot.yaxis]:
        axis.setTextPen(font_pen)
        axis.setPen(font_pen)
        axis.setTickPen(font_pen)
        if axis.label is not None:
            axis.label.setDefaultTextColor(font_pen.color())

    if size is not None:
        view.plot.resize(size)

    for key in CALIBRATION_KEYS:
        for result in results:
            if not result.canCalibrate(key):
                continue

            x = result.calibrated(key)
            top = np.percentile(x, 98.0)
            counts, edges = np.histogram(x, range=(0.0, top), bins="fd")

            view.plot.clear()
            view.plot.drawHistogram(counts, edges, width=1.0, pen=pen, brush=brush)

            path = output_dir.joinpath(f"{file_name}_{key}_{result.isotope}.png")
            view.exportImage(path, size=size, background=background_color)
