
from pyqtgraph import ViewBox
from PySide6 import QtCore


class LimitBoundViewBox(ViewBox):
    """Viewbox that autoRanges to any set limits."""

    def childrenBounds(self, frac=None, orthoRange=(None, None), items=None):
        bounds = super().childrenBounds(frac=frac, orthoRange=orthoRange, items=items)
        limits = self.state["limits"]["xLimits"], self.state["limits"]["yLimits"]
        for i in range(2):
            if bounds[i] is not None:
                if limits[i][0] != -1e307:  # and limits[i][0] < bounds[i][0]:
                    bounds[i][0] = limits[i][0]
                if limits[i][1] != +1e307:  # and limits[i][1] > bounds[i][1]:
                    bounds[i][1] = limits[i][1]
        return bounds


class ViewBoxForceScaleAtZero(LimitBoundViewBox):
    """Viewbox that forces the bottom to be 0."""

    def scaleBy(
        self,
        s: list[float] | None = None,
        center: QtCore.QPointF | None = None,
        x: float | None = None,
        y: float | None = None,
    ) -> None:
        if center is not None:
            center.setY(0.0)
        super().scaleBy(s, center, x, y)

    def translateBy(
        self,
        t: QtCore.QPointF | None = None,
        x: float | None = None,
        y: float | None = None,
    ) -> None:
        if t is not None:
            t.setY(0.0)
        if y is not None:
            y = 0.0
        super().translateBy(t, x, y)
