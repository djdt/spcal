from pathlib import Path

import numpy as np
from PySide6 import QtCore

import pytest
from pytestqt.qtbot import QtBot

from spcal.gui.batch.formatpages import (
    BatchNuWizardPage,
    BatchTOFWERKWizardPage,
    BatchTextWizardPage,
)

from spcal.gui.batch.wizard import (
    BatchFilesWizardPage,
    BatchMethodWizardPage,
    BatchRunWizardPage,
    SPCalBatchProcessingWizard,
)
from spcal.isotope import ISOTOPE_TABLE
from spcal.processing.method import SPCalProcessingMethod
from spcal.processing.options import SPCalIsotopeOptions


def test_batch_wizard_text(
    tmp_path: Path,
    test_data_path: Path,
    default_method: SPCalProcessingMethod,
    qtbot: QtBot,
):
    wiz = SPCalBatchProcessingWizard(None, default_method, [])
    qtbot.addWidget(wiz)
    with qtbot.waitExposed(wiz):
        wiz.show()

    # Data file page
    page = wiz.currentPage()
    assert isinstance(page, BatchFilesWizardPage)

    assert not page.isComplete()

    with pytest.raises(FileNotFoundError):
        page.addFile(test_data_path.joinpath("tofwerk/test_tofwerk.h5"))
    page.addFile(test_data_path.joinpath("text/tofwerk_export_au.csv"))
    assert page.files.count() == 1
    page.addFile(test_data_path.joinpath("text/tofwerk_export_au_bg.csv"))
    assert page.files.count() == 2

    # Test that no valid isotopes are availabled
    page.addFile(test_data_path.joinpath("text/text_onecol.csv"))
    assert page.files.count() == 3

    wiz.next()

    # Text Options page
    page = wiz.currentPage()
    assert isinstance(page, BatchTextWizardPage)

    assert page.table_isotopes.rowCount() == 0
    assert not page.isComplete()

    # Go back and remove file
    wiz.back()
    page = wiz.currentPage()
    assert isinstance(page, BatchFilesWizardPage)

    with qtbot.waitSignal(page.files.model().rowsRemoved, timeout=100):
        qtbot.mouseClick(
            page.files.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            pos=page.files.visualItemRect(page.files.item(2))
            .marginsRemoved(QtCore.QMargins(0, 0, 5, 5))
            .bottomRight(),
        )
    assert page.files.count() == 2

    wiz.next()

    # Text Options page
    page = wiz.currentPage()
    assert isinstance(page, BatchTextWizardPage)

    assert page.isComplete()

    assert len(page.selectedIsotopes()) == 1
    assert page.table_isotopes.item(1, 1).text() == "197Au"  # type: ignore

    page.table_isotopes.item(1, 1).setText("")  # type: ignore
    assert not page.isComplete()
    page.table_isotopes.item(1, 1).setText("197Au")  # type: ignore

    # Prevent error in simulated data, dwelltime can't be read accurately
    page.override_event_time.setChecked(True)

    wiz.next()

    # Method page
    page = wiz.currentPage()
    assert isinstance(page, BatchMethodWizardPage)

    assert page.isComplete()
    assert page.isotope_table.isotope_model.rowCount() == 1

    wiz.next()

    # Run page
    page = wiz.currentPage()
    assert isinstance(page, BatchRunWizardPage)

    assert page.isComplete()
    assert page.output_files.count() == 2

    page.output_name.setText("%DataFile%.csv")
    page.output_dir.setText(str(tmp_path))

    assert page.isComplete()

    wiz.accept()

    wait = 100  # up to one second
    while wait > 0:
        wait -= 1
        if page.status.text() == "Processing complete!":
            break
        qtbot.wait(10)
    assert wait > 0

    assert tmp_path.joinpath("tofwerk_export_au.csv").exists()
    assert tmp_path.joinpath("tofwerk_export_au_bg.csv").exists()

    wiz.close()


def test_batch_wizard_nu(
    tmp_path: Path,
    test_data_path: Path,
    default_method: SPCalProcessingMethod,
    qtbot: QtBot,
):
    wiz = SPCalBatchProcessingWizard(None, default_method, [])
    qtbot.addWidget(wiz)
    with qtbot.waitExposed(wiz):
        wiz.show()

    # Data file page
    page = wiz.currentPage()
    assert isinstance(page, BatchFilesWizardPage)

    assert not page.isComplete()

    page.radio_nu.click()
    page.addFile(test_data_path.joinpath("nu"))
    assert page.files.count() == 1
    assert page.isComplete()

    wiz.next()

    # Nu Options page
    page = wiz.currentPage()
    assert isinstance(page, BatchNuWizardPage)

    assert not page.isComplete()
    assert len(page.table.selectedIsotopes()) == 0

    assert page.cycle_number.value() == 0
    assert page.segment_number.value() == 0
    assert page.cycle_number.maximum() == 1
    assert page.segment_number.maximum() == 1

    assert page.check_blanking.isChecked()
    assert not page.check_chunked.isChecked()
    page.check_chunked.setChecked(True)
    page.chunk_size.setValue(1)

    with qtbot.waitSignal(page.table.isotopesChanged):
        qtbot.mouseClick(
            page.table.buttons["Ce"],
            QtCore.Qt.MouseButton.LeftButton,
        )

    assert len(page.table.selectedIsotopes()) == 1
    assert page.isComplete()

    wiz.next()

    # Method page
    page = wiz.currentPage()
    assert isinstance(page, BatchMethodWizardPage)

    assert page.isComplete()
    wiz.next()

    # Run page
    page = wiz.currentPage()
    assert isinstance(page, BatchRunWizardPage)

    assert page.isComplete()
    assert page.output_files.count() == 1

    page.output_name.setText("test_batch.csv")
    page.output_dir.setText(str(tmp_path))

    assert page.isComplete()

    wiz.accept()

    wait = 100  # up to one second
    while wait > 0:
        wait -= 1
        if page.status.text() == "Processing complete!":
            break
        qtbot.wait(10)
    assert wait > 0

    # 5 chunks = 5 files
    assert len(list(tmp_path.glob("*.csv"))) == 5

    wiz.close()


def test_batch_wizard_tofwerk(
    tmp_path: Path,
    test_data_path: Path,
    default_method: SPCalProcessingMethod,
    qtbot: QtBot,
):
    default_method.limit_options.limit_method = "compound poisson"
    default_method.prominence_required = 0.5
    default_method.isotope_options[ISOTOPE_TABLE[("Ru", 101)]] = SPCalIsotopeOptions(
        None, 1.0, 1.0
    )

    wiz = SPCalBatchProcessingWizard(None, default_method, [])
    qtbot.addWidget(wiz)
    with qtbot.waitExposed(wiz):
        wiz.show()

    # Data file page
    page = wiz.currentPage()
    assert isinstance(page, BatchFilesWizardPage)

    assert not page.isComplete()

    with pytest.raises(FileNotFoundError):
        page.addFile(test_data_path.joinpath("tofwerk/test_tofwerk.h5"))
    with pytest.raises(FileNotFoundError):
        page.addFile(test_data_path.joinpath("tofwerk/tofwerk_testdata.h5"))
    page.radio_tofwerk.click()
    page.addFile(test_data_path.joinpath("tofwerk/tofwerk_testdata.h5"))
    assert page.files.count() == 1
    page.addFile(test_data_path.joinpath("tofwerk/tofwerk_testdata.h5"))
    assert page.files.count() == 2

    with qtbot.waitSignal(page.files.model().rowsRemoved, timeout=100):
        qtbot.mouseClick(
            page.files.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            pos=page.files.visualItemRect(page.files.item(1))
            .marginsRemoved(QtCore.QMargins(0, 0, 5, 5))
            .bottomRight(),
        )
    assert page.files.count() == 1
    assert page.isComplete()

    wiz.next()

    # TOFWERK Options page
    page = wiz.currentPage()
    assert isinstance(page, BatchTOFWERKWizardPage)

    assert not page.isComplete()
    assert len(page.table.selectedIsotopes()) == 0

    with qtbot.waitSignal(page.table.isotopesChanged):
        qtbot.mouseClick(
            page.table.buttons["Ru"],
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.ShiftModifier,
        )

    assert len(page.table.selectedIsotopes()) == 5
    assert page.isComplete()

    wiz.next()

    # Method page
    page = wiz.currentPage()
    assert isinstance(page, BatchMethodWizardPage)

    assert page.isComplete()

    wiz.next()

    # Run page
    page = wiz.currentPage()
    assert isinstance(page, BatchRunWizardPage)

    assert page.isComplete()
    assert page.output_files.count() == 1

    page.output_name.setText("test_batch.csv")
    page.output_dir.setText(str(tmp_path))

    assert page.isComplete()

    wiz.accept()

    wait = 100  # up to one second
    while wait > 0:
        wait -= 1
        if page.status.text() == "Processing complete!":
            break
        qtbot.wait(10)
    assert wait > 0

    assert tmp_path.joinpath(page.output_name.text()).exists()

    wiz.close()


def test_batch_wizard_method_page(qtbot: QtBot):
    method = SPCalProcessingMethod()
    method.instrument_options.uptake = 0.5
    method.instrument_options.efficiency = 0.1

    method.accumulation_method = "detection threshold"
    method.points_required = 2
    method.prominence_required = 0.5

    method.limit_options.window_size = 100
    method.limit_options.max_iterations = 100
    method.limit_options.single_ion_parameters = np.array([])  # dummy

    method.limit_options.gaussian_kws["alpha"] = 1e-3
    method.limit_options.poisson_kws["alpha"] = 1e-3
    method.limit_options.poisson_kws["eta"] = 1
    method.limit_options.compound_poisson_kws["alpha"] = 1e-3
    method.limit_options.compound_poisson_kws["sigma"] = 0.7

    method.isotope_options[ISOTOPE_TABLE[("Ag", 107)]] = SPCalIsotopeOptions(
        1.0, 2.0, 1.0
    )
    method.isotope_options[ISOTOPE_TABLE[("Au", 197)]] = SPCalIsotopeOptions(
        None, 1.0, 0.5
    )

    page = BatchMethodWizardPage(method)
    qtbot.addWidget(page)

    assert page.instrument_options.uptake.baseValue() == 0.5
    assert page.instrument_options.efficiency.value() == 0.1

    assert page.limit_options.points_required == 2
    assert page.limit_options.prominence_required == 0.5
    assert page.limit_options.limit_accumulation == "detection threshold"

    assert page.limit_options.check_window.isChecked()
    assert page.limit_options.window_size.value() == 100
    assert page.limit_options.check_iterative.isChecked()

    assert page.limit_options.gaussian.alpha.value() == 1e-3
    assert page.limit_options.gaussian.sigma.value() == 3.0902
    assert page.limit_options.poisson.alpha.value() == 1e-3
    assert page.limit_options.poisson.eta == 1
    assert page.limit_options.compound.alpha.value() == 1e-3
    assert not page.limit_options.compound.lognormal_sigma.isEnabled()
    assert page.limit_options.compound.lognormal_sigma.value() == 0.7

    model = page.isotope_table.isotope_model
    assert model.rowCount() == 2
    assert model.data(model.index(0, 0)) == 0.001  # g/cm3
    assert model.data(model.index(0, 1)) == 2e-9  # L/ug
    assert model.data(model.index(0, 2)) == 1.0
    assert model.data(model.index(1, 0)) is None
    assert model.data(model.index(1, 1)) == 1e-9
    assert model.data(model.index(1, 2)) == 0.5

    page.close()
