from pathlib import Path
import bottleneck as bn
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph

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
            closest_pos = None
            event_pos = self.mapToDevice(event.pos())  # type: ignore
            for polygon in self.getPath().toSubpathPolygons():
                dist = QtCore.QLineF(
                    self.mapToDevice(polygon.at(1)),  # type: ignore
                    event_pos,  # type: ignore
                ).length()
                if dist < closest_dist:
                    closest_dist = dist
                    closest_pos = polygon.at(1)
            if closest_pos is not None and closest_dist < self.opts["mouseWidth"]:
                self.label.setPos(closest_pos)
                self.label.setText(f"{closest_pos.x():.4g}")
                self.label.setVisible(True)
            else:
                self.label.setVisible(False)
        else:
            self.label.setVisible(False)

    def hoverLeaveEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent):  # type: ignore
        self.label.setVisible(False)


class SpectraView(SinglePlotGraphicsView):
    def __init__(
        self, font: QtGui.QFont | None = None, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(
            "Spectra",
            xlabel="Mass (m/z)",
            ylabel="Signal (cts)",
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
        sums = np.zeros_like(data_file.masses)
        for region in regions:
            sums += bn.nansum(data_file.signals[region[0] : region[1]], axis=0) / (
                region[1] - region[0]
            )
        return data_file.masses, sums

    def drawDataFile(
        self,
        data_file: SPCalDataFile,
        regions: np.ndarray,
        min_value: float = 0.5,
        pen: QtGui.QPen | None = None,
        negative: bool = False,
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

        if self.plot.vb is None:
            return
        self.plot.vb.setLimits(xMin=xs[0] - 1, xMax=xs[-1] + 1)

        mask = ys > min_value
        xs = xs[mask]
        ys = ys[mask]


        if negative:
            ys *= -1.0
        else:
            self.data_for_export["m/z"] = xs
            self.data_for_export["intensity"] = ys

        curve = SpectraPlotItem(xs, ys, pen)
        self.plot.addItem(curve)
