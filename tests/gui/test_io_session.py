from pathlib import Path

import numpy as np
import numpy.lib.recfunctions as rfn
import pytest
from pytestqt.qt_compat import qt_api
from pytestqt.qtbot import QtBot

from spcal.gui.main import SPCalWindow
from spcal.io.session import restoreSession, saveSession
from spcal.npdb import db
from spcal.result import ClusterFilter, Filter

npz = np.load(Path(__file__).parent.parent.joinpath("data/tofwerk_auag.npz"))
data = rfn.unstructured_to_structured(
    np.stack((npz["107Ag"], npz["197Au"]), axis=1),
    dtype=[("Au", np.float32), ("Ag", np.float32)],
)


@pytest.fixture(scope="session")
def tmp_session_path(tmp_path_factory):
    return tmp_path_factory.mktemp("session").joinpath("tmp.spcal")


def test_save_session(tmp_session_path: Path, qtbot: QtBot):
    qt_api.QtWidgets.QApplication.setApplicationVersion("99.99.99")

    window = SPCalWindow()
    qtbot.addWidget(window)

    window.options.efficiency_method.setCurrentText("Reference Particle")
    window.options.uptake.setBaseValue(0.2)
    window.options.gaussian.alpha.setValue(0.003)
    window.options.poisson.alpha.setValue(0.004)
    window.options.compound_poisson.alpha.setValue(0.005)
    window.options.compound_poisson.single_ion_dist = np.arange(10)
    window.options.compound_poisson.lognormal_sigma.setValue(0.567)

    window.options.limit_accumulation = "Detection Threshold"
    window.options.points_required = 5

    window.sample.loadData(data, {"path": "test/data.csv", "dwelltime": 0.1})
    window.sample.io["Ag"].response.setBaseValue(1.0)
    window.sample.io["Au"].response.setBaseValue(2.0)
    window.sample.io["Au"].density.setBaseValue(3.0)

    window.sample.import_options["isotopes"] = db["isotopes"][[138, 275]]
    window.sample.import_options["dictdict"] = {"a": {"b": "c"}}

    window.reference.loadData(data.copy(), {"path": "test/ref.csv", "dwelltime": 0.1})
    window.reference.io["Au"].density.setBaseValue(4.0)

    window.results.updateResults()

    window.results.filters = [
        [Filter("Au", "signal", ">", 1.0)],
        [Filter("Ag", "mass", "<", 1000.0)],
    ]
    window.results.cluster_filters = [ClusterFilter(0, "signal")]

    window.sample.addExpression("{Ag+Au}", "+ Ag Au")
    window.reference.addExpression("{Ag+Au}", "+ Ag Au")

    window.sample.io.combo_name.setCurrentText("Au_mod")
    window.sample.io.combo_name.editingFinished()

    saveSession(
        tmp_session_path,
        window.options,
        window.sample,
        window.reference,
        window.results,
    )


def test_restore_session(tmp_session_path: Path, qtbot: QtBot):
    window = SPCalWindow()
    qtbot.addWidget(window)

    restoreSession(
        tmp_session_path,
        window.options,
        window.sample,
        window.reference,
        window.results,
    )

    assert window.options.efficiency_method.currentText() == "Reference Particle"
    assert window.options.event_time.baseValue() == 0.1
    assert window.options.uptake.baseValue() == 0.2
    assert window.options.gaussian.alpha.value() == 0.003
    assert window.options.poisson.alpha.value() == 0.004
    assert window.options.compound_poisson.alpha.value() == 0.005
    assert np.all(window.options.compound_poisson.single_ion_dist == np.arange(10))
    assert window.options.compound_poisson.lognormal_sigma.value() == 0.567

    assert window.options.limit_accumulation == "Detection Threshold"
    assert window.options.points_required == 5

    assert window.sample.names == ("Au_mod", "Ag", "{Ag+Au}")
    assert str(window.sample.import_options["path"]) == "test/data.csv"
    assert window.sample.io["Ag"].response.baseValue() == 1.0
    assert window.sample.io["Au_mod"].response.baseValue() == 2.0
    assert window.sample.io["Au_mod"].density.baseValue() == 3.0

    assert np.all(window.sample.import_options["isotopes"]["Isotope"] == [107, 197])
    assert window.sample.import_options["dictdict"]["a"]["b"] == "c"

    assert window.reference.names == ("Au_mod", "Ag", "{Ag+Au}")
    assert str(window.reference.import_options["path"]) == "test/ref.csv"
    assert window.reference.io["Au_mod"].density.baseValue() == 4.0

    assert len(window.results.filters) == 2
    assert window.results.filters[0][0].name == "Au_mod"
    assert window.results.filters[0][0].unit == "signal"
    assert window.results.filters[0][0].operation == ">"
    assert window.results.filters[0][0].value == 1.0
    assert window.results.filters[1][0].name == "Ag"
    assert window.results.filters[1][0].unit == "mass"
    assert window.results.filters[1][0].operation == "<"
    assert window.results.filters[1][0].value == 1000.0
    assert len(window.results.cluster_filters) == 1
    assert window.results.cluster_filters[0].unit == "signal"
    assert window.results.cluster_filters[0].idx == 0

    assert window.sample.current_expr["{Ag+Au}"] == "+ Ag Au"
