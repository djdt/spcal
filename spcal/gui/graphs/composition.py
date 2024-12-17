import numpy as np
import numpy.lib.recfunctions as rfn
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.cluster import cluster_information, prepare_data_for_clustering
from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.items import BarChart, PieChart
from spcal.gui.graphs.legends import StaticRectItemSample
from spcal.gui.modelviews import BasicTable
from spcal.gui.util import create_action


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
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setAspectLocked(1.0)
        self.plot.legend.setSampleType(StaticRectItemSample)
        self.xaxis.hide()
        self.yaxis.hide()

        self.pies: list[PieChart] = []

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

    def draw(
        self,
        data: dict[str, np.ndarray],
        T: np.ndarray,
        min_size: float | str = "5%",
        mode: str = "pie",
        pen: QtGui.QPen | None = None,
        brushes: list[QtGui.QBrush] | None = None,
    ) -> None:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)
        if brushes is None:
            brushes = [QtGui.QBrush() for _ in data.keys()]
        assert len(brushes) >= len(data.keys())

        self.pies.clear()
        self.export_data.clear()

        X = prepare_data_for_clustering(data)
        means, stds, counts = cluster_information(X, T)

        self.export_data["count"] = counts
        for i, name in enumerate(data.keys()):
            self.export_data[name + "_mean"] = means[:, i]
            self.export_data[name + "_std"] = stds[:, i]

        # Get minimum size as number
        if isinstance(min_size, str) and min_size.endswith("%"):
            min_size = X.shape[0] * float(min_size.rstrip("%")) / 100.0
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
        spacing = size * 2.0

        if mode == "pie":
            radii = np.sqrt(counts * np.pi)
            radii = radii / np.amax(radii) * size

            for i, (count, radius, comp) in enumerate(zip(counts, radii, compositions)):
                pie = PieChart(radius, rfn.structured_to_unstructured(comp), brushes)
                pie.setPos(i * spacing, 0)
                label = pyqtgraph.TextItem(
                    f"idx {i+1}: {count}", color="black", anchor=(0.5, 0.0)
                )
                label.setPos(i * spacing, -size)
                self.plot.addItem(pie)
                self.plot.addItem(label)
                self.pies.append(pie)
        elif mode == "bar":
            heights = counts / np.amax(counts) * size
            width = spacing / 2.0

            for i, (count, height, comp) in enumerate(
                zip(counts, heights, compositions)
            ):
                pie = BarChart(
                    height, width, rfn.structured_to_unstructured(comp), brushes
                )
                pie.setPos(i * spacing - width / 2.0, -size)
                label = pyqtgraph.TextItem(f"{count}", color="black", anchor=(0.5, 0.0))
                label.setPos(i * spacing, -size)
                self.plot.addItem(pie)
                self.plot.addItem(label)
                self.pies.append(pie)
        else:
            raise ValueError("Composition mode must be 'pie' or 'bar'.")

        # link all hovers
        # todo link hover to legend
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
