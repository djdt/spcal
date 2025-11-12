import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.cluster import cluster_information, prepare_data_for_clustering
from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.items import BarChart, PieChart
from spcal.gui.graphs.legends import StaticRectItemSample
from spcal.gui.modelviews.basic import BasicTable
from spcal.gui.util import create_action
from spcal.isotope import SPCalIsotope
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

    def showEvent(self, event: QtGui.QShowEvent) -> None:
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
        self.xaxis.hide()
        self.yaxis.hide()

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

    def dialogDetail(self) -> QtWidgets.QDialog:
        dlg = CompositionDetailDialog(self.export_data, parent=self)
        dlg.open()
        return dlg

    def drawResults(
        self,
        results: dict[SPCalIsotope, SPCalProcessingResult],
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
                    for result in results.values()
                    if result.peak_indicies is not None
                ]
            )
            + 1
        )
        peak_data = np.zeros((npeaks, len(results)), np.float32)
        for i, result in enumerate(results.values()):
            if result.peak_indicies is None:
                raise ValueError(
                    "cannot cluster, peak_indicies have not been generated"
                )
            np.add.at(
                peak_data[:, i],
                result.peak_indicies[result.filter_indicies],
                result.calibrated(key),
            )

        valid = np.any(peak_data != 0, axis=1)
        peak_data = peak_data[valid]

        X = prepare_data_for_clustering(peak_data)
        means, stds, counts = cluster_information(X, clusters[valid])

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

        pies = []
        if self.mode == "pie":
            radii = np.sqrt(counts * np.pi)
            radii = radii / np.amax(radii) * size

            for i, (count, radius, comp) in enumerate(zip(counts, radii, compositions)):
                pie = PieChart(
                    radius,
                    comp,
                    brushes,
                    labels=[str(result.isotope) for result in results.values()],
                    font=self.font(),
                )
                pie.setPos(i * spacing, 0)
                label = pyqtgraph.TextItem(
                    f"idx {i + 1}: {count}", color="black", anchor=(0.5, 0.0)
                )
                label.setPos(i * spacing, -size)
                self.plot.addItem(pie)
                self.plot.addItem(label)
                pies.append(pie)
        elif self.mode == "bar":
            heights = counts / np.amax(counts) * size
            width = spacing / 2.0

            for i, (count, height, comp) in enumerate(
                zip(counts, heights, compositions)
            ):
                pie = BarChart(height, width, comp, brushes, font=self.font())
                pie.setPos(i * spacing - width / 2.0, -size)
                label = pyqtgraph.TextItem(f"{count}", color="black", anchor=(0.5, 0.0))
                label.setPos(i * spacing, -size)
                self.plot.addItem(pie)
                self.plot.addItem(label)
                pies.append(pie)
        else:
            raise ValueError("Composition mode must be 'pie' or 'bar'.")

        # link all hovers
        # todo link hover to legend
        for i in range(len(pies)):
            for j in range(len(pies)):
                if i == j:
                    continue
                pies[i].hovered.connect(pies[j].setHoveredIdx)

        # Add legend for each pie
        if self.plot.legend is not None:
            for isotope, composition, brush in zip(
                results.keys(), compositions, brushes
            ):
                if np.sum(composition) > 0.0:
                    self.plot.legend.addItem(StaticRectItemSample(brush), str(isotope))

    def zoomReset(self) -> None:  # No plotdata
        if self.plot.vb is not None:
            self.plot.vb.autoRange()
