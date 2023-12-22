from pathlib import Path

import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.util import create_action


class SinglePlotGraphicsView(pyqtgraph.GraphicsView):
    def __init__(
        self,
        title: str,
        xlabel: str = "",
        ylabel: str = "",
        xunits: str | None = None,
        yunits: str | None = None,
        viewbox: pyqtgraph.ViewBox | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(background="white", parent=parent)

        pen = QtGui.QPen(QtCore.Qt.black, 1.0)
        pen.setCosmetic(True)

        self.export_data: dict[str, np.ndarray] = {}

        self.xaxis = pyqtgraph.AxisItem("bottom", pen=pen, textPen=pen, tick_pen=pen)
        self.xaxis.setLabel(xlabel, units=xunits)

        self.yaxis = pyqtgraph.AxisItem("left", pen=pen, textPen=pen, tick_pen=pen)
        self.yaxis.setLabel(ylabel, units=yunits)

        self.plot = pyqtgraph.PlotItem(
            title=title,
            name="central_plot",
            axisItems={"bottom": self.xaxis, "left": self.yaxis},
            viewBox=viewbox,
            parent=parent,
        )
        # Common options
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.addLegend(
            offset=(-5, 5), verSpacing=-5, colCount=1, labelTextColor="black"
        )

        self.setCentralWidget(self.plot)

        self.action_copy_image = create_action(
            "insert-image",
            "Copy Image to Clipboard",
            "Copy an image of the plot to the clipboard.",
            self.copyToClipboard,
        )

        self.action_show_legend = create_action(
            "view-hidden",
            "Hide Legend",
            "Toggle visibility of the legend.",
            lambda: None,
        )

        self.action_export_data = create_action(
            "document-export",
            "Export Data",
            "Save currently loaded data to file.",
            self.exportData,
        )

        self.context_menu_actions: list[QtGui.QAction] = []

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        menu = QtWidgets.QMenu(self)
        menu.addAction(self.action_copy_image)
        if self.plot.legend is not None:
            if self.plot.legend.isVisible():
                self.action_show_legend.setIcon(QtGui.QIcon.fromTheme("view-hidden"))
                self.action_show_legend.setText("Hide Legend")
                self.action_show_legend.triggered.connect(
                    lambda: self.plot.legend.setVisible(False)
                )
            else:
                self.action_show_legend.setIcon(QtGui.QIcon.fromTheme("view-visible"))
                self.action_show_legend.setText("Show Legend")
                self.action_show_legend.triggered.connect(
                    lambda: self.plot.legend.setVisible(True)
                )

            menu.addAction(self.action_show_legend)
            if self.readyForExport():
                menu.addSeparator()
                menu.addAction(self.action_export_data)

            if len(self.context_menu_actions) > 0:
                menu.addSeparator()
            for action in self.context_menu_actions:
                menu.addAction(action)

        event.accept()
        menu.popup(event.globalPos())

    def copyToClipboard(self) -> None:
        """Copy current view to system clipboard."""
        pixmap = QtGui.QPixmap(self.viewport().size())
        painter = QtGui.QPainter(pixmap)
        self.render(painter)
        painter.end()
        QtWidgets.QApplication.clipboard().setPixmap(pixmap)  # type: ignore

    def clear(self) -> None:
        self.plot.legend.clear()
        self.plot.clear()
        self.export_data.clear()

    def dataBounds(self) -> tuple[float, float, float, float]:
        items = [item for item in self.plot.listDataItems() if item.isVisible()]
        bx = np.array([item.dataBounds(0) for item in items], dtype=float)
        by = np.array([item.dataBounds(1) for item in items], dtype=float)
        bx, by = np.nan_to_num(bx), np.nan_to_num(by)
        # Just in case
        if len(bx) == 0 or len(by) == 0:
            return (0, 1, 0, 1)
        return (
            np.amin(bx[:, 0]),
            np.amax(bx[:, 1]),
            np.amin(by[:, 0]),
            np.amax(by[:, 1]),
        )

    def dataRect(self) -> QtCore.QRectF:
        x0, x1, y0, y1 = self.dataBounds()
        return QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)

    def exportData(self) -> None:
        dir = QtCore.QSettings().value("RecentFiles/1/path", None)
        dir = str(Path(dir).parent) if dir is not None else ""
        path, filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Data",
            dir,
            "CSV Documents(*.csv);;Numpy archives(*.npz);;All Files(*)",
        )
        if path == "":
            return

        path = Path(path)

        filter_suffix = filter[filter.rfind(".") : -1]
        if filter_suffix != "":  # append suffix if missing
            path = path.with_suffix(filter_suffix)

        data = self.dataForExport()
        names = list(data.keys())

        if path.suffix.lower() == ".csv":
            header = "\t".join(name for name in names)
            stack = np.full(
                (max(d.size for d in data.values()), len(data)),
                np.nan,
                dtype=np.float32,
            )
            for i, x in enumerate(data.values()):
                stack[: x.size, i] = x
            np.savetxt(
                path, stack, delimiter="\t", comments="", header=header, fmt="%.16g"
            )
        elif path.suffix.lower() == ".npz":
            np.savez_compressed(
                path,
                **{k: v for k, v in data.items()},
            )
        else:
            raise ValueError("dialogExportData: file suffix must be '.npz' or '.csv'.")

    def dataForExport(self) -> dict[str, np.ndarray]:
        return self.export_data

    def readyForExport(self) -> bool:
        return len(self.export_data) > 0

    def setLimits(self, **kwargs) -> None:
        self.plot.setLimits(**kwargs)

    def setDataLimits(
        self,
        xMin: float | None = None,
        xMax: float | None = None,
        yMin: float | None = None,
        yMax: float | None = None,
    ) -> None:
        """Set all plots limits in range 0.0 - 1.0."""
        bounds = self.dataBounds()
        dx = bounds[1] - bounds[0]
        dy = bounds[3] - bounds[2]
        limits = {}
        if xMin is not None:
            limits["xMin"] = bounds[0] + dx * xMin
        if xMax is not None:
            limits["xMax"] = bounds[0] + dx * xMax
        if yMin is not None:
            limits["yMin"] = bounds[2] + dy * yMin
        if yMax is not None:
            limits["yMax"] = bounds[2] + dy * yMax
        self.setLimits(**limits)

    def zoomReset(self) -> None:
        x, y = self.plot.vb.state["autoRange"][0], self.plot.vb.state["autoRange"][1]
        self.plot.autoRange()
        self.plot.enableAutoRange(x=x, y=y)


# class MultiPlotGraphicsView(SPCalGraphicsView):
#     def __init__(
#         self, minimum_plot_height: int = 160, parent: QtWidgets.QWidget | None = None
#     ):
#         self.minimum_plot_height = minimum_plot_height
#         self.layout = pyqtgraph.GraphicsLayout()
#         self.plots: dict[str, pyqtgraph.PlotItem] = {}

#         self.limits: dict[str, float] = {}

#         super().__init__(parent=parent)

#         self.setCentralWidget(self.layout)
#         self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

#     def addPlot(
#         self,
#         name: str,
#         plot: pyqtgraph.PlotItem,
#         xlink: bool = False,
#         ylink: bool = False,
#         expand_limits: bool = True,
#     ) -> None:
#         if xlink:
#             plot.setXLink(self.layout.getItem(0, 0))
#         if ylink:
#             plot.setYLink(self.layout.getItem(0, 0))

#         self.plots[name] = plot
#         self.layout.addItem(plot)
#         self.layout.nextRow()
#         self.resizeEvent(QtGui.QResizeEvent(QtCore.QSize(0, 0), QtCore.QSize(0, 0)))

#     def dataBounds(self) -> tuple[float, float, float, float]:
#         items = [
#             item
#             for plot in self.plots.values()
#             for item in plot.listDataItems()
#             if item.isVisible()
#         ]
#         bx = np.asarray([item.dataBounds(0) for item in items])
#         by = np.asarray([item.dataBounds(1) for item in items])
#         return (
#             np.amin(bx[:, 0]),
#             np.amax(bx[:, 1]),
#             np.amin(by[:, 0]),
#             np.amax(by[:, 1]),
#         )

#     def dataRect(self) -> QtCore.QRectF:
#         x0, x1, y0, y1 = self.dataBounds()
#         return QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)

#     def setLimits(self, **kwargs) -> None:
#         for plot in self.plots.values():
#             plot.setLimits(**kwargs)

#     def setDataLimits(
#         self,
#         xMin: float | None = None,
#         xMax: float | None = None,
#         yMin: float | None = None,
#         yMax: float | None = None,
#     ) -> None:
#         """Set all plots limits in range 0.0 - 1.0."""
#         bounds = self.dataBounds()
#         dx = bounds[1] - bounds[0]
#         dy = bounds[3] - bounds[2]
#         limits = {}
#         if xMin is not None:
#             limits["xMin"] = bounds[0] + dx * xMin
#         if xMax is not None:
#             limits["xMax"] = bounds[0] + dx * xMax
#         if yMin is not None:
#             limits["yMin"] = bounds[2] + dy * yMin
#         if yMax is not None:
#             limits["yMax"] = bounds[2] + dy * yMax
#         self.setLimits(**limits)

#     # Taken from pyqtgraph.widgets.MultiPlotWidget
#     def setRange(self, *args, **kwds):
#         pyqtgraph.GraphicsView.setRange(self, *args, **kwds)
#         if self.centralWidget is not None:
#             r = self.range
#             minHeight = self.layout.currentRow * self.minimum_plot_height
#             if r.height() < minHeight:
#                 r.setHeight(minHeight)
#                 r.setWidth(r.width() - self.verticalScrollBar().width())
#             self.centralWidget.setGeometry(r)

#     # Taken from pyqtgraph.widgets.MultiPlotWidget
#     def resizeEvent(self, event: QtGui.QResizeEvent):
#         if self.closed:
#             return
#         if self.autoPixelRange:
#             self.range = QtCore.QRectF(0, 0, self.size().width(), self.size().height())
#         MultiPlotGraphicsView.setRange(
#             self, self.range, padding=0, disableAutoPixel=False
#         )  # we do this because some subclasses like to redefine
#         # setRange in an incompatible way.
#         self.updateMatrix()

#     def clear(self) -> None:
#         self.layout.clear()
#         self.plots = {}

#     def legends(self) -> list[pyqtgraph.LegendItem]:
#         return [plot.legend for plot in self.plots.values()]

#     def zoomReset(self) -> None:
#         for plot in self.plots.values():
#             plot.autoRange()
