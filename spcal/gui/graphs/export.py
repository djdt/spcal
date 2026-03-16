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
    keys: list[str] | None = None,
    max_percentile: float = 98.0,
    pen: QtGui.QPen | None = None,
    width: int = 1200,
    background: QtGui.QColor | QtCore.Qt.GlobalColor = QtCore.Qt.GlobalColor.white,
):
    if not output_dir.is_dir():
        output_dir = output_dir.parent

    if keys is None:
        keys = CALIBRATION_KEYS
    if pen is None:
        pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0)
        pen.setCosmetic(True)

    view = SinglePlotGraphicsView("", xlabel="Signal", ylabel="Count")

    for key in keys:
        for result in results:
            if not result.canCalibrate(key):
                continue
            x = result.calibrated(key)

            view.plot.clear()
            top = np.percentile(x, max_percentile)
            counts, edges = np.histogram(x, range=(0.0, top), bins="fd")
            view.plot.drawHistogram(counts, edges, width=1.0)
            path = output_dir.joinpath(f"{file_name}_{key}_{result.isotope}.png")
            view.exportImage(path, background=background)
