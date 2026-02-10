from pathlib import Path

from PySide6 import QtCore

import pytest
from pytestqt.qtbot import QtBot

from spcal.gui.batch.formatpages import BatchTOFWERKWizardPage, BatchTextWizardPage

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
    default_method.limit_options.limit_method = "compound poisson"
    default_method.prominence_required = 0.5

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
            .marginsRemoved(QtCore.QMargins(5, 0, 0, 5))
            .bottomLeft(),
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
    # default_method.limit_options.limit_method = "compound poisson"
    # default_method.prominence_required = 0.5
    # default_method.isotope_options[ISOTOPE_TABLE[("Ru", 101)]] = SPCalIsotopeOptions(
    #     None, 1.0, 1.0
    # )
    #
    wiz = SPCalBatchProcessingWizard(None, default_method, [])
    qtbot.addWidget(wiz)
    with qtbot.waitExposed(wiz):
        wiz.show()
    #
    # # Data file page
    # page = wiz.currentPage()
    # assert isinstance(page, BatchFilesWizardPage)
    #
    # assert not page.isComplete()
    #
    # with pytest.raises(FileNotFoundError):
    #     page.addFile(test_data_path.joinpath("tofwerk/test_tofwerk.h5"))
    # with pytest.raises(FileNotFoundError):
    #     page.addFile(test_data_path.joinpath("tofwerk/tofwerk_testdata.h5"))
    # page.radio_tofwerk.click()
    # page.addFile(test_data_path.joinpath("tofwerk/tofwerk_testdata.h5"))
    # assert page.files.count() == 1
    # page.addFile(test_data_path.joinpath("tofwerk/tofwerk_testdata.h5"))
    # assert page.files.count() == 2
    #
    # with qtbot.waitSignal(page.files.model().rowsRemoved, timeout=100):
    #     qtbot.mouseClick(
    #         page.files.viewport(),
    #         QtCore.Qt.MouseButton.LeftButton,
    #         pos=page.files.visualItemRect(page.files.item(1))
    #         .marginsRemoved(QtCore.QMargins(5, 0, 0, 5))
    #         .bottomLeft(),
    #     )
    # assert page.files.count() == 1
    # assert page.isComplete()
    #
    # wiz.next()
    #
    # # TOFWERK Options page
    # page = wiz.currentPage()
    # assert isinstance(page, BatchTOFWERKWizardPage)
    #
    # assert not page.isComplete()
    # assert len(page.table.selectedIsotopes()) == 0
    #
    # with qtbot.waitSignal(page.table.isotopesChanged):
    #     qtbot.mouseClick(
    #         page.table.buttons["Ru"],
    #         QtCore.Qt.MouseButton.LeftButton,
    #         QtCore.Qt.KeyboardModifier.ShiftModifier,
    #     )
    #
    # assert len(page.table.selectedIsotopes()) == 5
    # assert page.isComplete()
    #
    # wiz.next()
    #
    # # Method page
    # page = wiz.currentPage()
    # assert isinstance(page, BatchMethodWizardPage)
    #
    # assert page.isComplete()
    #
    # assert page.isotope_table.isotope_model.rowCount() == 5
    # assert (
    #     page.limit_options.limit_method.currentText()
    #     == default_method.limit_options.limit_method.title()
    # )
    # assert (
    #     page.limit_options.compound.lognormal_sigma.value()
    #     == default_method.limit_options.compound_poisson_kws["sigma"]
    # )
    # assert page.limit_options.prominence_required == default_method.prominence_required
    # assert (
    #     page.isotope_table.isotope_model.data(
    #         page.isotope_table.isotope_model.index(0, 2)
    #     )
    #     is None
    # )
    # assert (
    #     page.isotope_table.isotope_model.data(
    #         page.isotope_table.isotope_model.index(2, 2)
    #     )
    #     == 1.0
    # )
    #
    # wiz.next()
    #
    # # Run page
    # page = wiz.currentPage()
    # assert isinstance(page, BatchRunWizardPage)
    #
    # assert page.isComplete()
    # assert page.output_files.count() == 1
    #
    # page.output_name.setText("test_batch.csv")
    # page.output_dir.setText(str(tmp_path))
    #
    # assert page.isComplete()
    #
    # wiz.accept()
    #
    # wait = 100  # up to one second
    # while wait > 0:
    #     wait -= 1
    #     if page.status.text() == "Processing complete!":
    #         break
    #     qtbot.wait(10)
    # assert wait > 0
    #
    # assert tmp_path.joinpath(page.output_name.text()).exists()
    #
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
            .marginsRemoved(QtCore.QMargins(5, 0, 0, 5))
            .bottomLeft(),
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

    assert page.isotope_table.isotope_model.rowCount() == 5
    assert (
        page.limit_options.limit_method.currentText()
        == default_method.limit_options.limit_method.title()
    )
    assert (
        page.limit_options.compound.lognormal_sigma.value()
        == default_method.limit_options.compound_poisson_kws["sigma"]
    )
    assert page.limit_options.prominence_required == default_method.prominence_required
    assert (
        page.isotope_table.isotope_model.data(
            page.isotope_table.isotope_model.index(0, 2)
        )
        is None
    )
    assert (
        page.isotope_table.isotope_model.data(
            page.isotope_table.isotope_model.index(2, 2)
        )
        == 1.0
    )

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


# def test_batch_export(tmp_path: Path, qtbot: QtBot):
#     window = SPCalWindow()
#     qtbot.add_widget(window)
#     with qtbot.wait_exposed(window):
#         window.show()
#
#     assert not window.action_open_batch.isEnabled()
#
#     path = Path(__file__).parent.parent.joinpath("data/text/tof_mix_au_ag_auag.csv")
#
#     data = read_single_particle_file(path, columns=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
#
#     with qtbot.wait_signal(window.sample.detectionsChanged):
#         window.sample.loadData(
#             data,
#             {
#                 "path": path,
#                 "columns": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#                 "first line": 0,
#                 "names": {data.dtype.names[0]: data.dtype.names[0]},
#                 "cps": False,
#                 "delimiter": ",",
#                 "importer": "text",
#                 "dwelltime": 1e-4,
#             },
#         )
#
#     assert window.action_open_batch.isEnabled()
#
#     tmp = tmp_path.joinpath("batch_export.csv")
#     dlg = window.dialogBatchProcess()
#     qtbot.add_widget(dlg)
#     dlg.files.addItems([str(path)])
#     dlg.check_summary.setChecked(True)
#
#     dlg.output_name.setText(tmp.name)
#     dlg.output_dir.setText(str(tmp_path))
#
#     assert not tmp.exists()
#
#     with qtbot.wait_signal(dlg.processingFinshed):
#         dlg.start()
#
#     assert tmp.stat().st_size > 0
#     assert dlg.summary_path is not None
#     assert dlg.summary_path.stat().st_size > 0
#
#
# def test_batch_export_no_detections(tmp_path: Path, qtbot: QtBot):
#     window = SPCalWindow()
#     qtbot.add_widget(window)
#     with qtbot.wait_exposed(window):
#         window.show()
#
#     assert not window.action_open_batch.isEnabled()
#
#     path = Path(__file__).parent.parent.joinpath("data/text/tofwerk_export_au.csv")
#     bg_path = Path(__file__).parent.parent.joinpath(
#         "data/text/tofwerk_export_au_bg.csv"
#     )
#
#     data = read_single_particle_file(path, columns=(2,))
#
#     with qtbot.wait_signal(window.sample.detectionsChanged):
#         window.sample.loadData(
#             data,
#             {
#                 "path": path,
#                 "columns": [2],
#                 "first line": 0,
#                 "names": {data.dtype.names[0]: data.dtype.names[0]},
#                 "cps": False,
#                 "delimiter": ",",
#                 "importer": "text",
#                 "dwelltime": 1e-4,
#             },
#         )
#
#     assert window.action_open_batch.isEnabled()
#
#     tmp = tmp_path.joinpath("batch_export_bg.csv")
#     dlg = window.dialogBatchProcess()
#     qtbot.add_widget(dlg)
#     dlg.files.addItems([str(bg_path)])
#
#     dlg.output_name.setText(tmp.name)
#     dlg.output_dir.setText(str(tmp_path))
#
#     assert not tmp.exists()
#
#     with qtbot.wait_signal(dlg.processingFinshed):
#         dlg.start()
#
#     assert tmp.stat().st_size > 0
#
#
# def test_batch_export_nu(tmp_path: Path, qtbot: QtBot):
#     # Todo: need to make a better (more data) Nu test data file
#     window = SPCalWindow()
#     qtbot.add_widget(window)
#     with qtbot.wait_exposed(window):
#         window.show()
#
#     assert not window.action_open_batch.isEnabled()
#
#     path = Path(__file__).parent.parent.joinpath("data/nu")
#
#     masses, signals, info = read_directory(path)
#     data = select_nu_signals(masses, signals, selected_masses={"Ar40": 39.96238})
#
#     window.options.compound_poisson.alpha.setValue(0.1)
#
#     with qtbot.wait_signal(window.sample.detectionsChanged):
#         window.sample.loadData(
#             data,
#             {
#                 "path": path,
#                 "importer": "nu",
#                 "dwelltime": 2.8e-5,
#                 "isotopes": np.array(
#                     [(18, "Ar", 40, 39.9623831237, 0.996035, 0)],
#                     dtype=[
#                         ("Number", np.uint16),
#                         ("Symbol", "U2"),
#                         ("Isotope", np.uint16),
#                         ("Mass", float),
#                         ("Composition", float),
#                         ("Preffered", np.uint8),
#                     ],
#                 ),
#                 "cycle": 1,
#                 "segment": 1,
#                 "blanking": True,
#             },
#         )
#
#     assert window.action_open_batch.isEnabled()
#
#     tmp = tmp_path.joinpath("batch_export_nu.csv")
#
#     dlg = window.dialogBatchProcess()
#     qtbot.add_widget(dlg)
#     dlg.files.addItems([str(path)])
#
#     dlg.output_name.setText(tmp.name)
#     dlg.output_dir.setText(str(tmp_path))
#
#     assert not tmp.exists()
#
#     with qtbot.wait_signal(dlg.processingFinshed):
#         dlg.start()
#
#     assert tmp.stat().st_size > 0
#
#
# def test_batch_export_tofwerk(tmp_path: Path, qtbot: QtBot):
#     window = SPCalWindow()
#     qtbot.add_widget(window)
#     with qtbot.wait_exposed(window):
#         window.show()
#
#     assert not window.action_open_batch.isEnabled()
#
#     path = Path(__file__).parent.parent.joinpath("data/tofwerk/tofwerk_au_50nm.h5")
#
#     data, info, dwell = read_tofwerk_file(path, idx=np.array([293]))
#
#     with qtbot.wait_signal(window.sample.detectionsChanged):
#         window.sample.loadData(
#             data,
#             {
#                 "path": path,
#                 "importer": "tofwerk",
#                 "dwelltime": dwell,
#                 "isotopes": np.array(
#                     [(79, "Au", 197, 196.96656879, 1.0, 1)],
#                     dtype=[
#                         ("Number", np.uint16),
#                         ("Symbol", "U2"),
#                         ("Isotope", np.uint16),
#                         ("Mass", float),
#                         ("Composition", float),
#                         ("Preffered", np.uint8),
#                     ],
#                 ),
#                 "other peaks": [],
#             },
#         )
#
#     assert window.action_open_batch.isEnabled()
#
#     tmp = tmp_path.joinpath("batch_export_tofwerk.csv")
#
#     dlg = window.dialogBatchProcess()
#     qtbot.add_widget(dlg)
#     dlg.files.addItems([str(path)])
#
#     dlg.output_name.setText(tmp.name)
#     dlg.output_dir.setText(str(tmp_path))
#
#     assert not tmp.exists()
#
#     with qtbot.wait_signal(dlg.processingFinshed):
#         dlg.start()
#
#     assert tmp.stat().st_size > 0
#
#
# def test_batch_export_images(tmp_path: Path, qtbot: QtBot):
#     window = SPCalWindow()
#     qtbot.add_widget(window)
#     with qtbot.wait_exposed(window):
#         window.show()
#
#     assert not window.action_open_batch.isEnabled()
#
#     path = Path(__file__).parent.parent.joinpath("data/text/tof_mix_au_ag_auag.csv")
#
#     data = read_single_particle_file(path, columns=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
#
#     with qtbot.wait_signal(window.sample.detectionsChanged):
#         window.sample.loadData(
#             data,
#             {
#                 "path": path,
#                 "columns": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#                 "first line": 0,
#                 "names": {data.dtype.names[0]: data.dtype.names[0]},
#                 "cps": False,
#                 "delimiter": ",",
#                 "importer": "text",
#                 "dwelltime": 1e-4,
#             },
#         )
#
#     assert window.action_open_batch.isEnabled()
#
#     tmp = tmp_path.joinpath("batch_export_images.csv")
#     dlg = window.dialogBatchProcess()
#     qtbot.add_widget(dlg)
#     dlg.files.addItems([str(path)])
#
#     dlg.check_image.setChecked(True)
#
#     dlg.output_name.setText(tmp.name)
#     dlg.output_dir.setText(str(tmp_path))
#
#     assert not tmp.exists()
#
#     with qtbot.wait_signal(dlg.processingFinshed):
#         dlg.start()
#
#     assert tmp.with_name("batch_export_images_Ag107_signal.png").exists()
#     assert tmp.with_name("batch_export_images_Ag109_signal.png").exists()
#     assert tmp.with_name("batch_export_images_Au197_signal.png").exists()
#
#
# def test_batch_export_difference(tmp_path: Path, qtbot: QtBot):
#     window = SPCalWindow()
#     qtbot.add_widget(window)
#     with qtbot.wait_exposed(window):
#         window.show()
#
#     # This was causing differences
#     window.options.check_iterative.setChecked(True)
#
#     path = Path(__file__).parent.parent.joinpath("data/text/text_batch_difference.csv")
#     data = read_single_particle_file(path, columns=(1,), first_line=4)
#
#     with qtbot.wait_signal(window.sample.detectionsChanged):
#         window.sample.loadData(
#             data,
#             {
#                 "path": path,
#                 "columns": [1],
#                 "first line": 4,
#                 "names": {"Fe56__56": "Fe56__56"},
#                 "cps": False,
#                 "delimiter": ",",
#                 "importer": "text",
#                 "dwelltime": 1e-4,
#             },
#         )
#
#     tmp_results = tmp_path.joinpath("batch_diff_results.csv")
#     tmp_batch = tmp_path.joinpath("batch_diff_batch.csv")
#
#     window.results.updateResults()
#     dlg = window.results.dialogExportResults()
#     dlg.lineedit_path.setText(str(tmp_results))
#     dlg.accept()
#
#     dlg = window.dialogBatchProcess()
#     qtbot.add_widget(dlg)
#     dlg.files.addItems([str(path)])
#     dlg.check_summary.setChecked(True)
#
#     dlg.output_name.setText(tmp_batch.name)
#     dlg.output_dir.setText(str(tmp_path))
#
#     with qtbot.wait_signal(dlg.processingFinshed):
#         dlg.start()
#
#     assert tmp_results.exists()
#     assert tmp_batch.exists()
#
#     with tmp_batch.open() as fp1, tmp_batch.open() as fp2:
#         for line1, line2 in zip(fp1.readlines(), fp2.readlines()):
#             assert line1 == line2
