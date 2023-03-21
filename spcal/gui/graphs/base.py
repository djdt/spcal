from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.util import create_action


class PlotCurveItemFix(pyqtgraph.PlotCurveItem):
    """Temporary class to fix error in pyqtgraph"""

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        import math
        import warnings

        # Need this to run as fast as possible.
        # check cache first:
        cache = self._boundsCache[ax]
        if cache is not None and cache[0] == (frac, orthoRange):
            return cache[1]

        (x, y) = self.getData()
        if x is None or len(x) == 0:
            return (None, None)

        if ax == 0:
            d = x
            d2 = y
        elif ax == 1:
            d = y
            d2 = x
        else:
            raise ValueError("Invalid axis value")

        # If an orthogonal range is specified, mask the data now
        if orthoRange is not None:
            mask = (d2 >= orthoRange[0]) * (d2 <= orthoRange[1])
            if self.opts.get("stepMode", None) == "center":
                mask = mask[:-1]  # len(y) == len(x) - 1 when stepMode is center
            d = d[mask]
            # d2 = d2[mask]

        if len(d) == 0:
            return (None, None)

        # Get min/max (or percentiles) of the requested data range
        if frac >= 1.0:
            # include complete data range
            # first try faster nanmin/max function, then cut out infs if needed.
            with warnings.catch_warnings():
                # All-NaN data is acceptable; Explicit numpy warning is not needed.
                warnings.simplefilter("ignore")
                b = (np.nanmin(d), np.nanmax(d))
            if math.isinf(b[0]) or math.isinf(b[1]):
                mask = np.isfinite(d)
                d = d[mask]
                if len(d) == 0:
                    return (None, None)
                b = (d.min(), d.max())

        elif frac <= 0.0:
            raise Exception(
                "Value for parameter 'frac' must be > 0. (got %s)" % str(frac)
            )
        else:
            # include a percentile of data range
            mask = np.isfinite(d)
            d = d[mask]
            if len(d) == 0:
                return (None, None)
            b = np.percentile(d, [50 * (1 - frac), 50 * (1 + frac)])

        # adjust for fill level
        if ax == 1 and self.opts["fillLevel"] not in [None, "enclosed"]:
            b = (min(b[0], self.opts["fillLevel"]), max(b[1], self.opts["fillLevel"]))

        # Add pen width only if it is non-cosmetic.
        pen = self.opts["pen"]
        spen = self.opts["shadowPen"]
        if (
            pen is not None
            and not pen.isCosmetic()
            and pen.style() != QtCore.Qt.PenStyle.NoPen
        ):
            b = (b[0] - pen.widthF() * 0.7072, b[1] + pen.widthF() * 0.7072)
        if (
            spen is not None
            and not spen.isCosmetic()
            and spen.style() != QtCore.Qt.PenStyle.NoPen
        ):
            b = (b[0] - spen.widthF() * 0.7072, b[1] + spen.widthF() * 0.7072)

        self._boundsCache[ax] = [(frac, orthoRange), b]
        return b


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

    def dataBounds(self) -> Tuple[float, float, float, float]:
        items = [item for item in self.plot.listDataItems() if item.isVisible()]
        bx = np.asarray([item.dataBounds(0) for item in items])
        by = np.asarray([item.dataBounds(1) for item in items])
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
            stack = np.stack(list(data.values()), axis=-1)
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

    def dataForExport(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def readyForExport(self) -> bool:
        return False

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
#         self.plots: Dict[str, pyqtgraph.PlotItem] = {}

#         self.limits: Dict[str, float] = {}

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

#     def dataBounds(self) -> Tuple[float, float, float, float]:
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

#     def legends(self) -> List[pyqtgraph.LegendItem]:
#         return [plot.legend for plot in self.plots.values()]

#     def zoomReset(self) -> None:
#         for plot in self.plots.values():
#             plot.autoRange()
