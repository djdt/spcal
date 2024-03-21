from pathlib import Path

import numpy as np
from pytestqt.qtbot import QtBot

from spcal.gui.main import SPCalWindow

# Determined experimentally
uptake = 1.567e-6
dwelltime = 1e-4
response = 16.08e9
efficiency = 0.062
density = 19.32e3

UPPER_INTEGER = True

limit = 37.63
if UPPER_INTEGER:
    limit = int(limit) + 1
ref_limit = 12.94
if UPPER_INTEGER:
    ref_limit = int(ref_limit) + 1


def test_sample_with_reference(qtbot: QtBot):
    window = SPCalWindow()
    qtbot.add_widget(window)
    with qtbot.wait_exposed(window):
        window.show()

    npz = np.load(Path(__file__).parent.parent.joinpath("data/agilent_au_data.npz"))
    data = np.array(npz["au50nm"], dtype=[("Au", np.float32), ("Ag", np.float32)])
    ref = np.array(npz["au15nm"], dtype=[("Au", np.float32)])

    window.options.efficiency_method.setCurrentText("Reference Particle")
    window.options.uptake.setBaseValue(uptake)
    assert not window.options.dwelltime.hasAcceptableInput()
    assert not window.options.efficiency.isEnabled()

    window.sample.loadData(data, {"path": "test/data.csv", "dwelltime": dwelltime})
    assert window.sample.io.names() == ["Au", "Ag"]

    # Check values on load
    assert window.options.dwelltime.baseValue() == dwelltime
    assert window.sample.io["Au"].density.baseValue() is None
    assert window.sample.io["Au"].molarmass.baseValue() is None
    assert window.sample.io["Au"].response.baseValue() is None
    assert window.sample.io["Au"].massfraction.value() == 1.0

    assert window.sample.detections["Au"].size == 3072
    assert np.isclose(
        np.mean(window.sample.trimmedResponse("Au")[window.sample.labels == 0]),
        0.3064,
        atol=1e-4,
    )
    assert np.isclose(window.sample.limits["Au"].detection_threshold, limit, atol=1e-2)

    assert window.sample.io["Au"].count.value() == 3072
    assert window.sample.io["Au"].count.error() == np.round(np.sqrt(3072), 0)
    assert np.isclose(
        window.sample.io["Au"].background_count.value(), 0.3064, atol=1e-4
    )
    assert np.isclose(window.sample.io["Au"].lod_count.value(), limit, atol=1e-2)
    assert window.sample.io["Au"].lod_label.text() == "Poisson (alpha=0.001)"

    # Set values
    window.sample.io["Au"].density.setBaseValue(density)
    window.sample.io["Au"].response.setBaseValue(response)
    window.sample.io["Ag"].density.setBaseValue(density)
    window.sample.io["Ag"].response.setBaseValue(response)

    window.reference.loadData(ref, {"path": "test/ref.csv", "dwelltime": dwelltime})
    assert window.reference.io.names() == ["Au"]

    # Check values on load
    assert window.reference.io["Au"].concentration.baseValue() is None
    assert window.reference.io["Au"].diameter.baseValue() is None
    assert window.reference.io["Au"].density.baseValue() is None
    # Should be shared with sample and set
    assert window.reference.io["Au"].response.baseValue() == response
    assert window.reference.io["Au"].massfraction.value() == 1.0

    assert window.reference.detections["Au"].size == 9551
    assert np.isclose(
        np.mean(window.reference.trimmedResponse("Au")[window.reference.labels == 0]),
        0.1984,
        atol=1e-4,
    )
    assert np.isclose(
        window.reference.limits["Au"].detection_threshold, ref_limit, atol=1e-2
    )
    assert window.reference.io["Au"].efficiency.value() is None
    assert window.reference.io["Au"].efficiency_label.text() == ""
    assert window.reference.io["Au"].massresponse.baseValue() is None

    # Set values
    window.reference.io["Au"].density.setBaseValue(density)
    window.reference.io["Au"].diameter.setBaseValue(15e-9)
    assert np.isclose(window.reference.io["Au"].efficiency.value(), 0.05427, atol=1e-5)
    assert window.reference.io["Au"].efficiency_label.text() == "(Mass)"
    assert np.isclose(
        window.reference.io["Au"].massresponse.baseValue(),
        5.2885e-22,
        atol=1e-26,
    )

    # Check results are ready
    assert window.tabs.isTabEnabled(window.tabs.indexOf(window.results))

    window.results.updateResults()
    assert window.results.results["Au"].canCalibrate("size")
    assert window.results.results["Ag"].canCalibrate("size")
    assert window.results.io["Au"].count.value() == 3072
    assert window.results.io["Au"].count.error() == np.round(np.sqrt(3072), 0)
    assert window.results.io["Au"].count_label.text() == "(100 %)"

    # Should test all outputs here
    assert np.isclose(window.results.io["Au"].mean.baseValue(), 2091, atol=1)
    assert np.isclose(window.results.io["Au"].median.baseValue(), 2034, atol=1)
    assert np.isclose(
        window.results.io["Au"].lod.baseValue(), limit - data["Au"].mean(), atol=0.1
    )
    assert np.isclose(window.results.io["Au"].number.baseValue(), 7.946e8, rtol=1e-4)
    assert np.isclose(window.results.io["Au"].conc.baseValue(), 8.788e-10, rtol=1e-4)
    assert np.isclose(
        window.results.io["Au"].background.baseValue(), 1.906e-11, rtol=1e-4
    )

    # Check all
    window.reference.io["Au"].check_use_efficiency_for_all.setChecked(False)

    window.results.updateResults()
    assert not window.results.results["Ag"].canCalibrate("size")
