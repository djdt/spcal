from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from spcal.gui.main import SPCalWindow


@pytest.fixture(scope="module")
def spcal_window() -> SPCalWindow:
    data = np.empty(1000, dtype=[("A", np.float32), ("B", np.float32)])
    for name in ["A", "B"]:
        data[name] = np.random.random(1000)
        data[name][50::100] += 20.0

    window = SPCalWindow()
    window.sample.loadData(data, {"path": "test/fake_data.csv", "dwelltime": 0.001})
    return window


def test_image_export_particle_single(
    qtbot: QtBot, spcal_window: SPCalWindow, tmp_path: Path
):
    qtbot.add_widget(spcal_window)
    spcal_window.sample.setDrawMode("single")
    spcal_window.sample.exportGraphImage(
        tmp_path.joinpath("image_export_particle_single.png"),
        QtCore.QSize(600, 400),
        96,
        {"transparent background": True},
    )


def test_image_export_particle_multiple(
    qtbot: QtBot, spcal_window: SPCalWindow, tmp_path: Path
):
    qtbot.add_widget(spcal_window)
    spcal_window.sample.setDrawMode("overlay")
    spcal_window.sample.exportGraphImage(
        tmp_path.joinpath("image_export_particle_multiple.png"),
        QtCore.QSize(600, 400),
        96,
        {
            "transparent background": False,
            "show detections": False,
            "show legend": False,
        },
    )


def test_image_export_histogram_single(
    qtbot: QtBot, spcal_window: SPCalWindow, tmp_path: Path
):
    qtbot.add_widget(spcal_window)
    spcal_window.tabs.setCurrentWidget(spcal_window.results)
    spcal_window.results.setHistDrawMode("single")
    spcal_window.results.exportGraphHistImage(
        tmp_path.joinpath("image_export_hist_single.png"),
        QtCore.QSize(600, 400),
        96,
        {"transparent background": True},
    )


def test_image_export_histogram_multiple(
    qtbot: QtBot, spcal_window: SPCalWindow, tmp_path: Path
):
    qtbot.add_widget(spcal_window)
    spcal_window.results.setHistDrawMode("overlay")
    spcal_window.tabs.setCurrentWidget(spcal_window.results)
    spcal_window.results.exportGraphHistImage(
        tmp_path.joinpath("image_export_hist_multiple.png"),
        QtCore.QSize(600, 400),
        96,
        {
            "show legend": False,
            "show fit": False,
            "show limits": False,
            "transparent background": True,
        },
    )
