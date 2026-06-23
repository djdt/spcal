from typing import Callable
from pathlib import Path
from PySide6 import QtGui

from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from spcal.datafile import SPCalDataFile
from spcal.gui.graphs.particle import ParticleView
from spcal.gui.graphs.histogram import HistogramView
from spcal.processing.method import SPCalProcessingMethod


def png_size(path: Path):
    with path.open("rb") as fp:
        data = fp.read(24)

    w = int.from_bytes(data[16:20])
    h = int.from_bytes(data[20:])

    return w, h


def test_image_export_particle(
    qtbot: QtBot,
    tmp_path: Path,
    default_method: SPCalProcessingMethod,
    random_datafile_generator: Callable[..., SPCalDataFile],
):
    view = ParticleView()
    output = tmp_path.joinpath("particle_image.png")
    df = random_datafile_generator()
    results = default_method.processDataFile(df)

    for result in results.values():
        view.drawResult(result, label=str(result.isotope))

    with qtbot.waitExposed(view):
        view.show()

    view.exportImageWithOptions(
        output,
        dpi=300,
        size=QtCore.QSize(1800, 1200),
        font=QtGui.QFont("serif"),
    )

    assert output.exists()
    assert png_size(output) == (1800, 1200)


def test_image_export_histogram(
    qtbot: QtBot,
    tmp_path: Path,
    default_method: SPCalProcessingMethod,
    random_datafile_generator: Callable[..., SPCalDataFile],
):
    view = HistogramView()
    output = tmp_path.joinpath("histogram_image.png")
    df = random_datafile_generator()
    results = default_method.processDataFile(df)

    for result in results.values():
        view.drawResult(result, label=str(result.isotope))

    with qtbot.waitExposed(view):
        view.show()

    view.exportImageWithOptions(
        output,
        dpi=96,
        size=QtCore.QSize(600, 400),
        font=QtGui.QFont("sans"),
    )

    assert output.exists()
    assert png_size(output) == (600, 400)
