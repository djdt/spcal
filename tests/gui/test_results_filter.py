import numpy as np
import numpy.lib.recfunctions as rfn
import pytest
from pytestqt.qtbot import QtBot

from spcal.gui.main import SPCalWindow
from spcal.result import ClusterFilter, Filter


# Clustering doesn't like the fake data
@pytest.mark.filterwarnings("ignore:invalid value")
def test_results_filters(qtbot: QtBot):
    window = SPCalWindow()
    qtbot.add_widget(window)
    with qtbot.wait_exposed(window):
        window.show()

    data = np.full((1000, 3), 0.01, dtype=np.float32)
    data[5::10, 0] += 100 + np.arange(0, 100)
    data[5::10, 1] += 100 + np.arange(0, 100) * 2
    data[5::10, 2] += 100 + np.tile([10, 20, 30, 40, 50], 20)

    data = rfn.unstructured_to_structured(
        data, dtype=[("A", np.float32), ("B", np.float32), ("C", np.float32)]
    )

    window.options.efficiency_method.setCurrentText("Manual Input")
    window.options.limit_method.setCurrentText("Manual Input")
    window.options.efficiency.setValue(0.1)
    window.options.uptake.setBaseValue(1.0)

    window.sample.loadData(data, {"path": "test/fake_data.csv", "dwelltime": 0.001})

    # Set values
    window.sample.io["A"].density.setBaseValue(1.0)
    window.sample.io["A"].response.setBaseValue(1.0)
    window.sample.io["A"].lod_count.setValue(0.02)
    window.sample.io["B"].lod_count.setValue(0.02)
    window.sample.io["C"].lod_count.setValue(0.02)
    window.sample.updateLimits()

    # Update results
    window.results.graph_options["histogram"]["fit"] = None
    window.results.graph_options["composition"]["distance"] = 0.01
    window.tabs.setCurrentWidget(window.results)

    for result in window.results.results.values():
        assert result.number == 100

    window.results.setFilters([[Filter("A", "signal", ">=", 149.0)]], [])
    for result in window.results.results.values():
        assert result.number == 50

    window.results.setFilters([[Filter("A", "signal", ">", 159.0)]], [])
    for result in window.results.results.values():
        assert result.number == 40

    window.results.setFilters(
        [[Filter("A", "signal", ">", 149.0), Filter("B", "signal", "<", 239.0)]], []
    )
    for result in window.results.results.values():
        assert result.number == 20

    window.results.setFilters(
        [
            [Filter("A", "signal", ">", 149.0), Filter("B", "signal", "<", 239.0)],
            [Filter("C", "signal", ">", 150.0)],
        ],
        [],
    )
    for result in window.results.results.values():
        assert result.number == 20

    window.results.setFilters([[Filter("A", "mass", ">", 0.019)]], [])
    for result in window.results.results.values():
        assert result.number == 9

    window.results.setFilters([], [])
    for result in window.results.results.values():
        assert result.number == 100

    # Cluster filter
    counts = np.bincount(window.results.clusters["signal"])
    idx = np.argsort(counts)[::-1]
    for i, c in enumerate(counts[idx]):
        window.results.setFilters([], [ClusterFilter(i, "signal")])
        for result in window.results.results.values():
            assert result.number == c

    filters = [ClusterFilter(i, "signal") for i in np.arange(counts.size)]
    window.results.setFilters([], filters)
    for result in window.results.results.values():
        assert result.number == 100

    window.results.setFilters(
        [[Filter("A", "signal", ">=", 149.0)]], [ClusterFilter(0, "signal")]
    )
    for result in window.results.results.values():
        assert result.number < 50
