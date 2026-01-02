from pathlib import Path
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph

from spcal.calc import search_sorted_closest

from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.datafile import SPCalDataFile, SPCalNuDataFile, SPCalTOFWERKDataFile


class SpectraPlotItem(pyqtgraph.PlotCurveItem):
    def __init__(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        pen: QtGui.QPen,
        parent: QtWidgets.QGraphicsItem | None = None,
    ):
        xs = np.repeat(xs, 2)
        ys = np.stack((np.zeros_like(ys), ys), axis=1).ravel()

        super().__init__(xs, ys, connect="pairs", pen=pen, parent=parent)
        self.opts["mouseWidth"] = 50.0

        self.setAcceptHoverEvents(True)

        self.label = pyqtgraph.TextItem(anchor=(0.5, 1))
        self.label.setParentItem(self)
        self.label.setVisible(False)

    def hoverMoveEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent):  # type: ignore
        if self.mouseShape().contains(event.pos()):  # type: ignore
            closest_dist = 9999.9
            for polygon in self.getPath().toSubpathPolygons():
                dist = QtCore.QLineF(
                    self.mapToDevice(polygon.at(1)),  # type: ignore
                    self.mapToDevice(event.pos()),  # type: ignore
                ).length()
                if dist < closest_dist:
                    self.label.setPos(polygon.at(1))
                    self.label.setText(f"{polygon.at(1).x():.4g}")
                    self.label.setVisible(True)
                    closest_dist = dist
        else:
            self.label.setVisible(False)

    def hoverLeaveEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent):
        self.label.setVisible(False)


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
        pen: QtGui.QPen | None = None,
        reverse: bool = False,
    ):
        if isinstance(data_file, SPCalNuDataFile):
            xs, ys = self.spectraForNuFile(data_file, regions)
        elif isinstance(data_file, SPCalTOFWERKDataFile):
            xs, ys = self.spectraForTOFWERKFile(data_file, regions)
        else:
            raise NotImplementedError

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 2.0 * self.devicePixelRatio())
            pen.setCosmetic(True)

        mask = ys > min_value
        xs = xs[mask]
        ys = ys[mask]

        self.data_for_export["m/z"] = xs
        self.data_for_export["intensity"] = ys

        curve = SpectraPlotItem(xs, ys, pen)
        self.plot.addItem(curve)

        # scatter = pyqtgraph.ScatterPlotItem(
        #     xs,
        #     ys,
        #     # pen=QtGui.QPen(QtCore.Qt.PenStyle.NoPen),
        #     # brush=QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush),
        #     hoverable=True,
        #     # hoverSize=100,
        #     tip="{x:.3g} {y:.3g} {}".format,
        # )
        # scatter.setAcceptHoverEvents(True)
        # self.plot.addItem(scatter)

        # xs = np.repeat(xs, 2)
        # if reverse:
        #     ys *= -1.0
        # ys = np.stack((np.zeros_like(ys), ys), axis=1).ravel()
        #
        # self.drawCurve(xs, ys, pen=pen, connect="pairs")


if __name__ == "__main__":
    from spcal.isotope import ISOTOPE_TABLE
    from spcal.datafile import SPCalDataFile, SPCalNuDataFile, SPCalTOFWERKDataFile
    from spcal.processing.method import SPCalProcessingMethod

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
