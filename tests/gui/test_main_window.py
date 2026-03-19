from spcal.datafile import SPCalTOFWERKDataFile
from spcal.processing.options import SPCalIsotopeOptions
from pathlib import Path
from spcal.isotope import ISOTOPE_TABLE
from spcal.processing.method import SPCalProcessingMethod
from pytestqt.qtbot import QtBot
from spcal.gui.mainwindow import SPCalMainWindow


def test_main_window_restore_method(qtbot: QtBot, test_data_path: Path):
    win = SPCalMainWindow()
    win.instrument_options.options_widget.efficiency.setValue(0.1)

    df = SPCalTOFWERKDataFile.load(
        test_data_path.joinpath("tofwerk/tofwerk_testdata.h5")
    )
    df.selected_isotopes = [ISOTOPE_TABLE[("Ru", 101)], ISOTOPE_TABLE[("Ru", 104)]]

    win.files.addDataFile(df)

    qtbot.addWidget(win)
    with qtbot.waitExposed(win):
        win.show()

    method = SPCalProcessingMethod()
    method.isotope_options[ISOTOPE_TABLE[("Ru", 101)]] = SPCalIsotopeOptions(
        1.0, 1.0, 1.0
    )
    method.isotope_options[ISOTOPE_TABLE[("Ru", 102)]] = SPCalIsotopeOptions(
        1.0, 2.0, 1.0
    )
    method.isotope_options[ISOTOPE_TABLE[("Ru", 104)]] = SPCalIsotopeOptions(
        1.0, 3.0, 1.0
    )

    method.instrument_options.uptake = 0.5
    method.instrument_options.efficiency = None

    # no raise due to 102
    win.setCurrentMethod(method)

    assert win.instrument_options.options_widget.uptake.baseValue() == 0.5
    assert win.instrument_options.options_widget.efficiency.value() is None

    assert win.isotope_options.optionForIsotope(
        ISOTOPE_TABLE[("Ru", 101)]
    ) == SPCalIsotopeOptions(1.0, 1.0, 1.0)
    assert win.isotope_options.optionForIsotope(
        ISOTOPE_TABLE[("Ru", 104)]
    ) == SPCalIsotopeOptions(1.0, 3.0, 1.0)
