from pathlib import Path
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.legends import HistogramItemSample
from spcal.gui.graphs.viewbox import ViewBoxForceScaleAtZero
from spcal.isotope import ISOTOPE_TABLE, SPCalIsotope
from spcal.processing import SPCalProcessingResult
from spcal.datafile import SPCalDataFile, SPCalNuDataFile, SPCalTOFWERKDataFile
from spcal.processing.method import SPCalProcessingMethod


class SpectraView(SinglePlotGraphicsView):
    def __init__(
        self, font: QtGui.QFont | None = None, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(
            "Spectra",
            xlabel="Mass (m/z)",
            ylabel="Signal (cts)",
            # viewbox=ViewBoxForceScaleAtZero(),
            font=font,
            parent=parent,
        )
        assert self.plot.vb is not None
        # self.plot.vb.setLimits(xMin=0.0, yMin=0.0)

    def spectraForTOFWERKFile(
        self, data_file: SPCalTOFWERKDataFile, regions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        peak_sums = np.add.reduceat(data_file.signals, regions.ravel(), axis=0)[::2]
        sums = np.nansum(peak_sums, axis=0)
        return data_file.masses, sums

    def spectraForNuFile(
        self, data_file: SPCalNuDataFile, regions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        peak_sums = np.add.reduceat(data_file.signals, regions.ravel(), axis=0)[::2]
        sums = np.nansum(peak_sums, axis=0)
        return data_file.masses, sums

    def drawDataFile(
        self,
        data_file: SPCalDataFile,
        regions: np.ndarray,
        min_value: float = 0.5,
        pen:QtGui.QPen|None=None,
        reverse: bool = False,
    ):
        if isinstance(data_file, SPCalNuDataFile):
            xs, ys = self.spectraForNuFile(data_file, regions)
        elif isinstance(data_file, SPCalTOFWERKDataFile):
            xs, ys = self.spectraForTOFWERKFile(data_file, regions)
        else:
            raise NotImplementedError

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.black,2.0*self.devicePixelRatio())
            pen.setCosmetic(True)

        mask = ys > min_value
        xs = xs[mask]
        ys = ys[mask]

        xs = np.repeat(xs, 2)
        if reverse:
            ys *= -1.0
        ys = np.stack((np.zeros_like(ys), ys), axis=1).ravel()

        curve = self.drawCurve(xs, ys, pen=pen, connect="pairs")


if __name__ == "__main__":
    app = QtWidgets.QApplication()

    df = SPCalNuDataFile.load(
        Path("/home/tom/Downloads/14-38-58 UPW + 80nm Au 90nm UCNP many particles")
    )
    # df = SPCalTOFWERKDataFile.load(
    #     Path("/home/tom/Downloads/Single cell_blank_2025-08-27_15h39m38s.h5")
    # )
    method = SPCalProcessingMethod()
    method.limit_options.compound_poisson_kws["sigma"] = 0.65

    results = method.processDataFile(df, [ISOTOPE_TABLE[("Yb", 172)]])

    graph = SpectraView()

    regions = results[ISOTOPE_TABLE[("Yb", 172)]].regions

    bg_regions = regions.ravel()[1::-1]

    graph.drawDataFile(df, regions)
    graph.drawDataFile(df, bg_regions, reverse=True)

    graph.show()
    app.exec()
