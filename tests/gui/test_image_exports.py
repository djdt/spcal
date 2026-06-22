
from typing import Callable
from pathlib import Path
from PySide6 import QtGui
import numpy as np

from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from spcal.datafile import SPCalNuDataFile, SPCalDataFile
from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.calibration import CalibrationView
from spcal.gui.graphs.items import BarChart, HoverableChartItem, PieChart
from spcal.gui.graphs.particle import ExclusionRegion, ParticleView
from spcal.gui.graphs.histogram import HistogramView
from spcal.gui.graphs.composition import CompositionView
from spcal.gui.graphs.scatter import ScatterView
from spcal.gui.graphs.spectra import SpectraView
from spcal.gui.graphs.legends import ParticleItemSample, HistogramItemSample
from spcal.isotope import ISOTOPE_TABLE
from spcal.processing.method import SPCalProcessingMethod
from spcal.processing.options import SPCalIsotopeOptions


def test_image_export_particle(qtbot: QtBot. tmp_path: Path,
    default_method: SPCalProcessingMethod,
    random_datafile_generator: Callable[..., SPCalDataFile]):
    pass
