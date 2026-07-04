from PySide6 import QtWidgets
from pathlib import Path

from pytestqt.qtbot import QtBot

from spcal.datafile import SPCalTOFWERKDataFile
from spcal.processing.options import SPCalIsotopeOptions
from spcal.isotope import ISOTOPE_TABLE, SPCalIsotopeExpression
from spcal.processing.method import SPCalProcessingMethod
from spcal.gui.mainwindow import SPCalMainWindow


def test_main_window_method_dialogs(qtbot: QtBot, test_data_path: Path):
    win = SPCalMainWindow()

    df = SPCalTOFWERKDataFile.load(
        test_data_path.joinpath("tofwerk/tofwerk_testdata.h5")
    )
    df.selected_isotopes = [ISOTOPE_TABLE[("Ru", 101)], ISOTOPE_TABLE[("Ru", 104)]]

    win.files.addDataFile(df)

    qtbot.addWidget(win)
    with qtbot.waitExposed(win):
        win.show()

    dlg = win.about()
    assert isinstance(dlg, QtWidgets.QDialog)
    dlg.close()

    dlg = win.dialogCalculator()
    assert dlg is not None
    dlg.close()

    dlg = win.dialogCustomColors()
    assert dlg is not None
    dlg.close()

    dlg = win.dialogExportResults()
    assert dlg is not None
    dlg.close()

    dlg = win.dialogFilterDetections()
    assert dlg is not None
    dlg.close()

    dlg = win.dialogIonicResponse()
    assert dlg is not None
    dlg.close()

    dlg = win.dialogLoadFile(test_data_path.joinpath("tofwerk/tofwerk_testdata.h5"))
    assert dlg is not None
    dlg.close()

    dlg = win.dialogManualLimits()
    assert dlg is not None
    dlg.close()

    dlg = win.dialogMassFractionCalculator()
    assert dlg is not None
    dlg.close()

    dlg = win.dialogParticleDatabase()
    assert dlg is not None
    dlg.close()

    dlg = win.dialogPeakProperties()
    assert dlg is not None
    dlg.close()

    dlg = win.dialogProcessingOptions()
    assert dlg is not None
    dlg.close()

    dlg = win.dialogTransportEfficiencyCalculator()
    assert dlg is not None
    dlg.close()

    # uses exec
    # dlg = win.dialogSessionLoad()
    #
    # uses exec
    # dlg = win.dialogSessionSave()


def test_main_window_method_functions(qtbot: QtBot, test_data_path: Path):
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

    with qtbot.waitSignal(win.currentMethodChanged, timeout=100):
        win.addExpression(
            SPCalIsotopeExpression(
                "test", (ISOTOPE_TABLE[("Ru", 101)], ISOTOPE_TABLE[("Ru", 104)], "+")
            )
        )
    assert len(win.currentMethod().expressions) == 1

    with qtbot.waitSignal(win.currentMethodChanged, timeout=100):
        win.removeExpressions(win.currentMethod().expressions)
    assert len(win.currentMethod().expressions) == 0

    with qtbot.waitSignal(win.currentMethodChanged, timeout=100):
        win.setGlobalExclusionRegions([(1.0, 2.0)])
    assert win.currentMethod().exclusion_regions == [(1.0, 2.0)]
    with qtbot.assertNotEmitted(win.currentMethodChanged):
        win.setGlobalExclusionRegions([(1.0, 2.0)])

    with qtbot.assertNotEmitted(win.currentMethodChanged):
        win.setExclusionRegions([(2.0, 3.0)], df)
    assert df.exclusion_regions == [(2.0, 3.0)]

    with qtbot.waitSignals(
        [win.isotope_options.optionChanged, win.currentMethodChanged], timeout=100
    ):
        win.setResponses(
            {ISOTOPE_TABLE[("Ru", 101)]: 10.0, ISOTOPE_TABLE[("Fe", 56)]: 2.0}
        )


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
