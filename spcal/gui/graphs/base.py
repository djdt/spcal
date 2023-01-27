from typing import Dict, List, Tuple

import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.util import create_action


class SPCalGraphicsView(pyqtgraph.GraphicsView):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent, background="white")

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
            self.toggleLegendVisible,
        )

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        menu = QtWidgets.QMenu(self)
        menu.addAction(self.action_copy_image)
        if all(legend is not None for legend in self.legends()):
            if any(legend.isVisible() for legend in self.legends()):
                self.action_show_legend.setIcon(QtGui.QIcon.fromTheme("view-hidden"))
                self.action_show_legend.setText("Hide Legend")
            else:
                self.action_show_legend.setIcon(QtGui.QIcon.fromTheme("view-visible"))
                self.action_show_legend.setText("Show Legend")

            menu.addAction(self.action_show_legend)
        menu.exec(event.globalPos())

    def copyToClipboard(self) -> None:
        """Copy current view to system clipboard."""
        pixmap = QtGui.QPixmap(self.viewport().size())
        painter = QtGui.QPainter(pixmap)
        self.render(painter)
        painter.end()
        QtWidgets.QApplication.clipboard().setPixmap(pixmap)  # type: ignore

    def legends(self) -> List[pyqtgraph.LegendItem]:
        return []

    def toggleLegendVisible(self) -> None:
        for legend in self.legends():
            legend.setVisible(not legend.isVisible())


class SinglePlotGraphicsView(SPCalGraphicsView):
    def __init__(
        self,
        title: str,
        xlabel: str,
        ylabel: str,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent=parent)

        pen = QtGui.QPen(QtCore.Qt.black, 1.0)
        pen.setCosmetic(True)

        self.xaxis = pyqtgraph.AxisItem("bottom", pen=pen, textPen=pen, tick_pen=pen)
        self.xaxis.setLabel(xlabel)

        self.yaxis = pyqtgraph.AxisItem("left", pen=pen, textPen=pen, tick_pen=pen)
        self.yaxis.setLabel(ylabel)

        self.plot = pyqtgraph.PlotItem(
            title=title,
            name="central_plot",
            axisItems={"bottom": self.xaxis, "left": self.yaxis},
            enableMenu=False,
            parent=parent,
        )
        self.plot.hideButtons()
        self.plot.setMouseEnabled(x=False)
        self.plot.addLegend(
            offset=(-5, 5), verSpacing=-5, colCount=1, labelTextColor="black"
        )

        self.setCentralWidget(self.plot)

    def clear(self) -> None:
        self.plot.clear()
        self.plot.legend.clear()

    def legends(self) -> List[pyqtgraph.LegendItem]:
        return [self.plot.legend]

    def zoomReset(self) -> None:
        self.plot.setRange(
            xRange=self.plot.vb.state["limits"]["xLimits"],
            yRange=self.plot.vb.state["limits"]["yLimits"],
            disableAutoRange=False,
        )


class MultiPlotGraphicsView(SPCalGraphicsView):
    def __init__(
        self, minimum_plot_height: int = 160, parent: QtWidgets.QWidget | None = None
    ):
        self.minimum_plot_height = minimum_plot_height
        self.layout = pyqtgraph.GraphicsLayout()
        self.plots: Dict[str, pyqtgraph.PlotItem] = {}

        super().__init__(parent=parent)

        self.setCentralWidget(self.layout)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    # Taken from pyqtgraph.widgets.MultiPlotWidget
    def setRange(self, *args, **kwds):
        pyqtgraph.GraphicsView.setRange(self, *args, **kwds)
        if self.centralWidget is not None:
            r = self.range
            minHeight = self.layout.currentRow * self.minimum_plot_height
            if r.height() < minHeight:
                r.setHeight(minHeight)
                r.setWidth(r.width() - self.verticalScrollBar().width())
            self.centralWidget.setGeometry(r)

    # Taken from pyqtgraph.widgets.MultiPlotWidget
    def resizeEvent(self, event: QtGui.QResizeEvent):
        if self.closed:
            return
        if self.autoPixelRange:
            self.range = QtCore.QRectF(0, 0, self.size().width(), self.size().height())
        MultiPlotGraphicsView.setRange(
            self, self.range, padding=0, disableAutoPixel=False
        )  # we do this because some subclasses like to redefine
        # setRange in an incompatible way.
        self.updateMatrix()

    def clear(self) -> None:
        self.layout.clear()
        self.plots = {}

    def legends(self) -> List[pyqtgraph.LegendItem]:
        return [plot.legend for plot in self.plots.values()]

    def bounds(self) -> Tuple[float, float, float, float]:
        bounds = np.array(
            [plot.vb.childrenBounds() for plot in self.plots.values()], dtype=float
        )
        if np.all(np.isnan(bounds)):
            return 0.0, 1.0, 0.0, 1.0
        return (
            np.nanmin(bounds[:, 0, 0]),
            np.nanmax(bounds[:, 0, 1]),
            np.nanmin(bounds[:, 1, 0]),
            np.nanmax(bounds[:, 1, 1]),
        )

    def zoomReset(self) -> None:
        for plot in self.plots.values():
            plot.vb.autoRange()
        # if self.layout.getItem(0, 0) is None:
        #     return
        # xmin, xmax, ymin, ymax = self.bounds()

        # for plot in self.plots.values():
        #     plot.setLimits(xMin=xmin, xMax=xmax, yMin=ymin, yMax=ymax)

        # self.layout.getItem(0, 0).setRange(
        #     xRange=(xmin, xmax),
        #     yRange=(ymin, ymax),
        #     disableAutoRange=True,
        # )
