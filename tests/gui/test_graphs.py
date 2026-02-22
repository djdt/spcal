import numpy as np

# from PySide6 import QtWidgets, QtGui
from pytestqt.qtbot import QtBot


from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.particle import ExclusionRegion, ParticleView
from spcal.gui.graphs.legends import ParticleItemSample, HistogramItemSample


def test_graph_base(qtbot: QtBot):
    view = SinglePlotGraphicsView("test")
    qtbot.addWidget(view)

    with qtbot.waitExposed(view):
        view.show()


def test_graph_particle(qtbot: QtBot, default_method, random_result_generator):
    view = ParticleView()
    qtbot.addWidget(view)

    with qtbot.waitExposed(view):
        view.show()

    result = random_result_generator(default_method, size=10000, number=100)
    view.drawResult(result)

    assert view.plot.legend is not None
    assert len(view.plot.legend.items) == 1
    assert len(view.data_for_export) == 4

    # Test overlapping legend label
    item, label = view.plot.legend.items[0]
    assert isinstance(item, ParticleItemSample)
    assert not item.sceneBoundingRect().intersects(label.sceneBoundingRect())

    result = random_result_generator(default_method, size=1000, number=10)
    result.signals[:50] = np.nan

    # Test overlapping
    view.drawResult(result)
    assert len(view.plot.legend.items) == 2
    item, _ = view.plot.legend.items[0]
    item2, _ = view.plot.legend.items[1]
    assert not item.sceneBoundingRect().intersects(item2.sceneBoundingRect())

    view.addExclusionRegion(0.1, 0.3)
    assert view.exclusionRegions() == [(0.1, 0.3)]

    for item in view.plot.items:
        if isinstance(item, ExclusionRegion):
            item.requestRemoval.emit()

    assert view.exclusionRegions() == []

    view.clear()
