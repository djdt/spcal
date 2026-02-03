import bottleneck as bn
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph

from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.datafile import SPCalDataFile, SPCalNuDataFile, SPCalTOFWERKDataFile

from spcal.isotope import ISOTOPE_TABLE


def text_for_mz(mz: float) -> str:
    text = f"{mz:.2f}"
    possible_isotopes = [
        iso
        for iso in ISOTOPE_TABLE.values()
        if abs(iso.mass - mz) < 0.1
        and iso.composition is not None
        and iso.composition > 0.05
    ]
    if len(possible_isotopes) > 0:
        text += "(" + ",".join(iso.symbol for iso in possible_isotopes) + ")"
    return text


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
                self.label.setText(text_for_mz(closest_pos.x()))
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

        # options
        self.subtract_background = True

    def spectraForTOFFile(
        self, data_file: SPCalNuDataFile | SPCalTOFWERKDataFile, regions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        sums = np.zeros_like(data_file.masses)

        for region in regions:
            sums += bn.nansum(data_file.signals[region[0] : region[1]], axis=0)

        counts = np.sum(regions[:, 1] - regions[:, 0])
        sums /= counts

        if self.subtract_background:
            bg_regions = np.reshape(regions.ravel()[1:-1], (-1, 2))
            bg = np.zeros_like(data_file.masses)
            for region in bg_regions:
                bg += bn.nansum(data_file.signals[region[0] : region[1]], axis=0)

            bg_counts = np.sum(bg_regions[:, 1] - bg_regions[:, 0])
            sums -= bg / bg_counts

        return data_file.masses, sums

    def drawDataFile(
        self,
        data_file: SPCalDataFile,
        regions: np.ndarray,
        min_value: float = 0.5,
        pen: QtGui.QPen | None = None,
        negative: bool = False,
    ):
        if isinstance(data_file, (SPCalNuDataFile, SPCalTOFWERKDataFile)):
            xs, ys = self.spectraForTOFFile(data_file, regions)
        else:
            raise NotImplementedError

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 2.0 * self.devicePixelRatio())
            pen.setCosmetic(True)

        if self.plot.vb is None:
            return

        self.plot.vb.setLimits(xMin=xs[0] - 1, xMax=xs[-1] + 1)

        mask = np.abs(ys) > min_value
        xs = xs[mask]
        ys = ys[mask]

        if negative:
            ys *= -1.0

        self.data_for_export["m/z"] = xs
        self.data_for_export["intensity"] = ys

        curve = SpectraPlotItem(xs, ys, pen)
        self.plot.addItem(curve)
