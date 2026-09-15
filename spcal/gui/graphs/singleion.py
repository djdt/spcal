import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.util import text_for_mz

# fmt: off
guide_data = np.array(
    [
        [0.9574, 1.0593], [0.9591, 1.0616], [0.9607, 1.0643], [0.9622, 1.0673], [0.9634, 1.0705],
        [0.9644, 1.0738], [0.9652, 1.0772], [0.9657, 1.0807], [0.9659, 1.0841], [0.9659, 1.0874],
        [0.9656, 1.0904], [0.965, 1.0932], [0.9642, 1.0956], [0.9632, 1.0976], [0.9619, 1.0992],
        [0.9605, 1.1003], [0.9591, 1.101], [0.9577, 1.1012], [0.9565, 1.1008], [0.9555, 1.0999],
        [0.9549, 1.0985], [0.9547, 1.0967], [0.9546, 1.0945], [0.9547, 1.0919], [0.9549, 1.0891],
        [0.9552, 1.0861], [0.9556, 1.0828], [0.956, 1.0793], [0.9564, 1.0756], [0.9568, 1.0717],
        [0.9571, 1.0676], [0.9573, 1.0634], [0.9576, 1.0592], [0.9577, 1.055], [0.9579, 1.0509],
        [0.958, 1.0469], [0.9581, 1.043], [0.9582, 1.0394], [0.9584, 1.036], [0.9586, 1.0329],
        [0.9589, 1.0301], [0.9594, 1.0277], [0.9599, 1.0257], [0.9607, 1.0241], [0.9615, 1.0229],
        [0.9626, 1.0221], [0.9639, 1.0217], [0.9653, 1.0216], [0.967, 1.0219], [0.9688, 1.0225],
        [0.9707, 1.0233], [0.9728, 1.0245], [0.9749, 1.0259], [0.9771, 1.0276], [0.9792, 1.0293],
        [0.9812, 1.0311], [0.9831, 1.033], [0.9848, 1.0348], [0.9862, 1.0365], [0.9874, 1.038],
        [0.9883, 1.0394], [0.989, 1.0406], [0.9895, 1.0415], [0.9897, 1.0421], [0.9896, 1.0425],
        [0.9894, 1.0426], [0.989, 1.0425], [0.9885, 1.0421], [0.9878, 1.0416], [0.9869, 1.0409],
        [0.986, 1.0401], [0.9849, 1.0392], [0.9838, 1.0381], [0.9826, 1.0371], [0.9814, 1.0359],
        [0.9802, 1.0346], [0.979, 1.0333], [0.9779, 1.0319], [0.9768, 1.0305], [0.9757, 1.029],
        [0.9748, 1.0275], [0.9738, 1.026], [0.9729, 1.0245], [0.9721, 1.023], [0.9714, 1.0214],
        [0.9707, 1.0199], [0.9701, 1.0184], [0.9696, 1.0169], [0.9691, 1.0154], [0.9687, 1.014],
        [0.9684, 1.0126], [0.9681, 1.0113], [0.9679, 1.0101], [0.9677, 1.009], [0.9675, 1.0081],
        [0.9673, 1.0073], [0.9672, 1.0067], [0.9671, 1.0063], [0.9669, 1.0061], [0.9668, 1.006],
        [0.9667, 1.006], [0.9667, 1.0062], [0.9666, 1.0065], [0.9666, 1.0069], [0.9666, 1.0074],
        [0.9666, 1.008], [0.9666, 1.0086], [0.9667, 1.0093], [0.9668, 1.01], [0.9669, 1.0108],
        [0.967, 1.0117], [0.9671, 1.0126], [0.9672, 1.0136], [0.9673, 1.0147], [0.9675, 1.0159],
        [0.9676, 1.0171], [0.9678, 1.0184], [0.9679, 1.0197], [0.968, 1.0211], [0.9681, 1.0225],
        [0.9682, 1.0239], [0.9682, 1.0253], [0.9683, 1.0267], [0.9683, 1.028], [0.9684, 1.0292],
        [0.9684, 1.0303], [0.9685, 1.0314], [0.9685, 1.0323], [0.9686, 1.0332], [0.9686, 1.034],
        [0.9687, 1.0348], [0.9688, 1.0354], [0.9689, 1.036], [0.9691, 1.0365], [0.9694, 1.0368],
        [0.9698, 1.0371], [0.9703, 1.0374], [0.9708, 1.0376], [0.9714, 1.0378], [0.972, 1.038],
        [0.9727, 1.0382], [0.9734, 1.0384], [0.9742, 1.0386], [0.975, 1.0388], [0.9757, 1.0389],
        [0.9765, 1.0391], [0.9772, 1.0392], [0.9779, 1.0392], [0.9785, 1.0392], [0.979, 1.0391]
    ]
)
# fmt: on


class SingleIonAreaScatterPlot(pyqtgraph.ScatterPlotItem):
    pointHovered = QtCore.Signal(QtCore.QPointF, int)
    pointClicked = QtCore.Signal(QtCore.QPointF, int)

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
    ):
        super().__init__(x=x, y=y, pen=pen, brush=brush)
        self.setAcceptHoverEvents(True)
        self.opts["mouseWidth"] = 50.0

        self.label = pyqtgraph.TextItem(anchor=(0.5, 1))
        self.label.setParentItem(self)
        self.label.setVisible(False)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        points: list[pyqtgraph.SpotItem] = self.pointsAt(event.pos())
        if len(points) > 0:
            self.pointClicked.emit(points[0].pos(), points[0].index())

    def hoverMoveEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent):
        points: list[pyqtgraph.SpotItem] = self.pointsAt(event.pos())
        if len(points) == 0:
            self.label.setVisible(False)
            return

        self.label.setPos(points[0].pos())
        self.label.setText(text_for_mz(points[0].pos().x()))
        self.label.setVisible(True)

        self.pointHovered.emit(
            QtCore.QPointF(points[0].pos().x(), points[0].pos().y()),
            int(points[0].index()),
        )


class SingleIonAreaScatterView(SinglePlotGraphicsView):
    pointHovered = QtCore.Signal(QtCore.QPointF, int)
    pointClicked = QtCore.Signal(QtCore.QPointF, int)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(
            "Extracted Parameters",
            xlabel="m/z",
            ylabel="Shape (σ)",
            parent=parent,
        )
        self.plot.yaxis.autoSIPrefix = False

        pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0)
        pen.setCosmetic(True)
        brush = QtGui.QBrush(QtCore.Qt.GlobalColor.black)

        self.points = SingleIonAreaScatterPlot(
            x=np.array([0]), y=np.array([0]), pen=pen, brush=brush
        )
        self.points.pointHovered.connect(self.pointHovered)
        self.points.pointClicked.connect(self.pointClicked)
        self.plot.addItem(self.points)

        pen = QtGui.QPen(QtCore.Qt.GlobalColor.red, 1.0)
        pen.setCosmetic(True)

        self.plot.getViewBox().setLimits(xMin=0.0, yMin=0.0)

    #     self.pointHovered.connect(self.onPointHovered)
    #
    # def onPointHovered(self, pos: QtCore.QPointF, index: int):
    #     self.label.setPos(pos)

    def clear(self):
        self.points.clear()

    def drawData(self, x: np.ndarray, y: np.ndarray):
        self.points.setData(x=x, y=y)
        self.setDataLimits(-0.05, 1.05, -0.05, 1.05)
