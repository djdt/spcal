import numpy as np
from pytestqt.qtbot import QtBot

from spcal.gui.main import SPCalWindow

data = np.empty(1000, dtype=[("A", float), ("B", float)])
data["A"] = 0.0
data["B"] = np.random.random(1000) * 0.1
data["A"][50::100] = 10.0
data["B"][25::200] += 10.0
data["B"][200:400] = np.nan


def test_sample_trimming(qtbot: QtBot):
    window = SPCalWindow()
    qtbot.add_widget(window)
    with qtbot.wait_exposed(window):
        window.show()

    window.sample.loadData(data, options={"path": "fake.txt", "dwelltime": 1e-3})

    assert np.count_nonzero(window.sample.detections["A"]) == 10
    assert np.count_nonzero(window.sample.detections["B"]) == 4

    # Trim out one event from a and b
    window.sample.graph.region.setRegion((100, 1000))
    assert np.count_nonzero(window.sample.detections["A"]) == 9
    assert np.count_nonzero(window.sample.detections["B"]) == 3

    # Trim out all events
    window.sample.graph.region.setRegion((60, 140))
    assert np.count_nonzero(window.sample.detections["A"]) == 0
    assert np.count_nonzero(window.sample.detections["B"]) == 0

    # Trim to all NaN for B
    window.sample.graph.region.setRegion((220, 380))
    assert np.count_nonzero(window.sample.detections["A"]) == 2
    assert np.count_nonzero(window.sample.detections["B"]) == 0
