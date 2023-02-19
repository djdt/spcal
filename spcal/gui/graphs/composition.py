from typing import Dict, List

import numpy as np
import numpy.lib.recfunctions as rfn
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.cluster import agglomerative_cluster, prepare_data_for_clustering
from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.items import PieChart
from spcal.gui.graphs.legends import StaticRectItemSample


class CompositionView(SinglePlotGraphicsView):
    def __init__(self, parent: QtWidgets.QWidget | None = None):

        super().__init__("Detection Compositions", parent=parent)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setAspectLocked(1.0)
        self.plot.legend.setSampleType(StaticRectItemSample)
        self.xaxis.hide()
        self.yaxis.hide()

        self.pies: List[PieChart] = []

    def draw(
        self,
        data: Dict[str, np.ndarray],
        distance: float = 0.03,
        min_size: float | str = "5%",
        pen: QtGui.QPen | None = None,
        brushes: List[QtGui.QBrush] | None = None,
    ) -> None:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)
        if brushes is None:
            brushes = [QtGui.QBrush() for _ in data.keys()]
        assert len(brushes) >= len(data.keys())

        self.pies.clear()

        fractions = prepare_data_for_clustering(data)

        if fractions.shape[0] == 1:  # single peak
            means, counts = fractions, np.array([1])
        elif fractions.shape[1] == 1:  # single element
            means, counts = np.array([[1.0]]), np.array([np.count_nonzero(fractions)])
        else:
            means, stds, counts = agglomerative_cluster(fractions, distance)

        # Get minimum size as number
        if isinstance(min_size, str) and min_size.endswith("%"):
            min_size = fractions.shape[0] * float(min_size.rstrip("%")) / 100.0
        elif isinstance(min_size, str | float):
            min_size = float(min_size)
        else:
            raise ValueError("draw: min_size is neither float nor a % str")

        compositions = np.empty(
            counts.size, dtype=[(name, np.float64) for name in data]
        )
        for i, name in enumerate(data):
            compositions[name] = means[:, i]

        mask = counts > min_size
        compositions = compositions[mask]
        counts = counts[mask]

        if counts.size == 0:
            return

        size = 100.0
        radii = np.sqrt(counts * np.pi)
        radii = radii / np.amax(radii) * size
        spacing = size * 2.0

        for i, (count, radius, comp) in enumerate(zip(counts, radii, compositions)):
            pie = PieChart(radius, rfn.structured_to_unstructured(comp), brushes)
            pie.setPos(i * spacing, 0)
            label = pyqtgraph.TextItem(f"{count}", color="black", anchor=(0.5, 0.0))
            label.setPos(i * spacing, -size)
            self.plot.addItem(pie)
            self.plot.addItem(label)
            self.pies.append(pie)

        # link all pie hovers
        for i in range(len(self.pies)):
            for j in range(len(self.pies)):
                if i == j:
                    continue
                self.pies[i].hovered.connect(self.pies[j].setHoveredIdx)

        # Add legend for each pie
        assert compositions.dtype.names is not None
        for name, brush in zip(compositions.dtype.names, brushes):
            if np.sum(compositions[name]) > 0.0:
                self.plot.legend.addItem(StaticRectItemSample(brush), name)

    def zoomReset(self) -> None:  # No plotdata
        self.plot.autoRange()
