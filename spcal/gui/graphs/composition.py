import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.cluster import cluster_information, prepare_data_for_clustering
from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.items import BarChart, PieChart
from spcal.gui.graphs.legends import StaticRectItemSample
from spcal.gui.modelviews.basic import BasicTable
from spcal.gui.util import create_action
from spcal.processing import SPCalProcessingResult


class CompositionDetailDialog(QtWidgets.QDialog):
    def __init__(
        self,
        export_data: dict[str, np.ndarray],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Compostion Detail")
        self.setMinimumSize(QtCore.QSize(480, 640))

        names = [k[:-5] for k in export_data.keys() if "_mean" in k]
        nrows = export_data["count"].size

        self.table = BasicTable()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setRowCount(nrows)
        self.table.setColumnCount(len(names) + 1)

        self.table.setHorizontalHeaderLabels(["Count"] + names)
        for i in range(nrows):
            item = QtWidgets.QTableWidgetItem(f"{export_data['count'][i]}")
            self.table.setItem(i, 0, item)
            for j, name in enumerate(names):
                mean = export_data[name + "_mean"][i]
                std = export_data[name + "_std"][i]
                item = QtWidgets.QTableWidgetItem(f"{mean:.2f} ± {std:.2f}")
                self.table.setItem(i, j + 1, item)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.table)
        self.setLayout(layout)

    def showEvent(self, event: QtGui.QShowEvent):
        super().showEvent(event)


class CompositionView(SinglePlotGraphicsView):
    def __init__(
        self, font: QtGui.QFont | None = None, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__("Detection Compositions", font=font, parent=parent)
        assert self.plot.vb is not None
        self.plot.vb.setMouseEnabled(x=False, y=False)
        self.plot.vb.setAspectLocked(True)
        self.plot.vb.invertY(True)
        assert self.plot.legend is not None
        self.plot.legend.setSampleType(StaticRectItemSample)
        self.plot.xaxis.hide()
        self.plot.yaxis.hide()

        # options
        self.mode = "pie"
        self.min_size: float | str = "5%"

        self.action_show_comp_dialog = create_action(
            "office-chart-pie",
            "Composition Detail",
            "Open a dialog displaying compositions in detail.",
            self.dialogDetail,
        )

        self.context_menu_actions.append(self.action_show_comp_dialog)

    def clear(self):
        super().clear()
        self.cluster_info = None

    def dialogDetail(self) -> QtWidgets.QDialog:
        dlg = CompositionDetailDialog(self.data_for_export, parent=self)
        dlg.open()
        return dlg

    def drawResults(
        self,
        results: list[SPCalProcessingResult],
        clusters: np.ndarray,
        key: str = "signal",
        pen: QtGui.QPen | None = None,
        brushes: list[QtGui.QBrush] | None = None,
    ):
        if brushes is None:
            brushes = [QtGui.QBrush(QtCore.Qt.GlobalColor.red) for _ in results]
        npeaks = (
            np.amax(
                [
                    result.peak_indicies[-1]
                    for result in results
                    if result.peak_indicies is not None
                ]
            )
            + 1
        )
        peak_data = np.zeros((npeaks, len(results)), np.float32)
        for i, result in enumerate(results):
            if result.peak_indicies is None:
                raise ValueError(
                    "cannot cluster, peak_indicies have not been generated"
                )
            if not result.canCalibrate(key):
                continue
            np.add.at(
                peak_data[:, i],
                result.peak_indicies[result.filter_indicies],
                result.calibrated(key),
            )

        valid = np.any(peak_data != 0, axis=1)
        peak_data = peak_data[valid]

        X = prepare_data_for_clustering(peak_data)
        means, stds, counts = cluster_information(X, clusters[valid])
        self.data_for_export["count"] = counts
        for i, result in enumerate(results):
            self.data_for_export[str(result.isotope) + "_mean"] = means[:, i]
            self.data_for_export[str(result.isotope) + "_std"] = stds[:, i]

        if isinstance(self.min_size, str) and self.min_size.endswith("%"):
            min_size = X.shape[0] * float(self.min_size.rstrip("%")) / 100.0
        else:
            min_size = float(self.min_size)

        mask = counts > min_size
        compositions = means[mask]
        counts = counts[mask]

        if counts.size == 0:
            return

        size = 100.0
        spacing = size * 2.0

        items = []
        if self.mode == "pie":
            radii = np.sqrt(counts * np.pi)
            radii = radii / np.amax(radii) * size

            for i, (count, radius, comp) in enumerate(zip(counts, radii, compositions)):
                item = PieChart(
                    radius,
                    comp,
                    brushes,
                    labels=[str(result.isotope) for result in results],
                    font=self.font(),
                )
                item.setPos(i * spacing, -size)
                label = pyqtgraph.TextItem(
                    f"idx {i + 1}: {count}",
                    color="black",
                    anchor=(0.5, 0.0),
                    ensureInBounds=True,
                )
                label.setPos(i * spacing, 0)
                self.plot.addItem(item)
                self.plot.addItem(label)
                items.append(item)
        elif self.mode == "bar":
            heights = counts / np.amax(counts) * size
            width = spacing / 2.0

            for i, (count, height, comp) in enumerate(
                zip(counts, heights, compositions)
            ):
                item = BarChart(
                    height,
                    width,
                    comp,
                    brushes,
                    labels=[str(result.isotope) for result in results],
                    font=self.font(),
                )
                item.setPos(i * spacing - width / 2.0, -height)
                label = pyqtgraph.TextItem(
                    f"idx {i + 1}: {count}",
                    color="black",
                    anchor=(0.5, 0.0),
                    ensureInBounds=True,
                )
                label.setPos(i * spacing, 0)
                self.plot.addItem(item)
                self.plot.addItem(label)
                items.append(item)
        else:
            raise ValueError("Composition mode must be 'pie' or 'bar'.")

        # link all hovers
        for i in range(len(items)):
            for j in range(len(items)):
                if i == j:
                    continue
                items[i].hovered.connect(items[j].setHoveredIdx)

        # Add legend for each pie
        if self.plot.legend is not None:
            for result, brush in zip(results, brushes):
                self.plot.legend.addItem(StaticRectItemSample(brush), str(result.isotope))

    def zoomReset(self):  # No plotdata
        if self.plot.vb is not None:
            self.plot.vb.autoRange()
