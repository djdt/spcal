from pathlib import Path

import numpy as np

# from PySide6 import QtCore
from pytestqt.qtbot import QtBot

# from spcal.gui.inputs import InputWidget, ReferenceWidget
from spcal.gui.main import SPCalWindow

# from spcal.gui.options import OptionsWidget
# from spcal.gui.results import ResultsWidget

# Determined experimentally
uptake = 1.567e-6
dwelltime = 1e-4
response = 16.08e9
efficiency = 0.062
density = 19.32e3


def test_sample_with_reference(qtbot: QtBot):
    window = SPCalWindow()
    qtbot.add_widget(window)
    with qtbot.wait_exposed(window):
        window.show()

    npz = np.load(Path(__file__).parent.parent.joinpath("data/agilent_au_data.npz"))
    data = np.array(npz["au50nm"], dtype=[("Au", float), ("Ag", float)])
    ref = np.array(npz["au15nm"], dtype=[("Au", float)])

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
    assert float(window.sample.io["Au"].massfraction.text()) == 1.0

    assert window.sample.io["Au"].count.text().startswith("3112")
    assert window.sample.io["Au"].background_count.text().startswith("0.5224")
    assert window.sample.io["Au"].lod_count.text().startswith("37.63")

    # Set values
    window.sample.io["Au"].density.setBaseValue(density)
    window.sample.io["Au"].response.setBaseValue(response)
    window.sample.io["Ag"].density.setBaseValue(density)
    window.sample.io["Ag"].response.setBaseValue(response)

    # Check if overwriting on re-load
    window.sample.loadData(data, {"path": "test/data.csv", "dwelltime": dwelltime})
    assert window.sample.io.names() == ["Au", "Ag"]
    assert window.sample.io["Au"].density.baseValue() == density
    assert window.sample.io["Au"].response.baseValue() == response

    window.reference.loadData(ref, {"path": "test/ref.csv", "dwelltime": dwelltime})
    assert window.reference.io.names() == ["Au"]

    # Check values on load
    assert window.reference.io["Au"].concentration.baseValue() is None
    assert window.reference.io["Au"].diameter.baseValue() is None
    assert window.reference.io["Au"].density.baseValue() is None
    # Should be shared with sample and set
    assert window.reference.io["Au"].response.baseValue() == response
    assert float(window.reference.io["Au"].massfraction.text()) == 1.0

    assert window.reference.io["Au"].count.text().startswith("10211")
    assert window.reference.io["Au"].background_count.text().startswith("0.5")
    assert window.reference.io["Au"].lod_count.text().startswith("12.94")
    assert window.reference.io["Au"].efficiency.text() == ""
    assert window.reference.io["Au"].massresponse.baseValue() is None

    # Set values
    window.reference.io["Au"].density.setBaseValue(density)
    window.reference.io["Au"].diameter.setBaseValue(15e-9)

    # Check results are ready
    assert window.tabs.isTabEnabled(window.tabs.indexOf(window.results))

    window.results.updateResults()
    assert "size" in window.results.results["Au"].detections
    # Should test all outputs here
    assert np.isclose(
        window.results.results["Au"].detections["size"].mean(), 52.14e-9, atol=0.01e-9
    )
    assert "size" not in window.results.results["Ag"].detections

    # Check all
    window.reference.io["Au"].check_use_efficiency_for_all.setChecked(True)

    window.results.updateResults()
    assert "size" in window.results.results["Ag"].detections
