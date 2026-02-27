from PySide6 import QtCore, QtGui
from pytestqt.qtbot import QtBot

from spcal.gui.docks.datafile import SPCalDataFilesDock
from spcal.gui.docks.central import SPCalCentralWidget
from spcal.gui.docks.instrumentoptions import SPCalInstrumentOptionsDock
from spcal.gui.docks.isotopeoptions import SPCalIsotopeOptionsDock
from spcal.gui.docks.limitoptions import SPCalLimitOptionsDock
from spcal.gui.docks.outputs import SPCalOutputsDock
from spcal.gui.docks.toolbar import SPCalOptionsToolBar, SPCalViewToolBar
from spcal.gui.modelviews import UnitsRole
from spcal.gui.modelviews.values import ValueWidgetDelegate
from spcal.isotope import ISOTOPE_TABLE
from spcal.processing.method import SPCalProcessingMethod
from spcal.processing.options import (
    SPCalInstrumentOptions,
    SPCalIsotopeOptions,
    SPCalLimitOptions,
)

from spcal.siunits import (
    number_concentration_units,
    mass_concentration_units,
    signal_units,
    mass_units,
    size_units,
)


def test_spcal_datfiles_dock(qtbot: QtBot, random_datafile_gen):
    dock = SPCalDataFilesDock()
    qtbot.addWidget(dock)

    with qtbot.waitExposed(dock):
        dock.show()

    assert dock.currentDataFile() is None
    assert len(dock.selectedDataFiles()) == 0

    dfs = [random_datafile_gen() for _ in range(5)]

    for df in dfs:
        with qtbot.waitSignal(dock.dataFileAdded):
            dock.addDataFile(df)

    assert len(dock.dataFiles()) == 5
    assert dock.currentDataFile() == dfs[-1]
    assert len(dock.selectedDataFiles()) == 1

    with qtbot.waitSignal(dock.dataFilesChanged):
        dock.list.selectAll()

    assert dock.currentDataFile() == dfs[-1]
    assert len(dock.selectedDataFiles()) == 5

    with qtbot.waitSignal(dock.dataFileRemoved):
        dock.model.removeRow(4)

    assert dock.currentDataFile() == dfs[-2]
    assert len(dock.dataFiles()) == 4

    dock.reset()

    assert len(dock.dataFiles()) == 0
    assert dock.currentDataFile() is None
    assert len(dock.selectedDataFiles()) == 0


def test_spcal_central_widget(
    qtbot: QtBot, default_method: SPCalProcessingMethod, random_datafile_gen
):
    widget = SPCalCentralWidget()
    qtbot.addWidget(widget)

    with qtbot.waitExposed(widget):
        widget.show()

    df = random_datafile_gen(
        size=1000,
        lam=10.0,
        isotopes=[
            ISOTOPE_TABLE[("Ag", 107)],
            ISOTOPE_TABLE[("Ag", 109)],
            ISOTOPE_TABLE[("Au", 197)],
        ],
        seed=131290,
    )

    default_method.limit_options.limit_method = "poisson"
    default_method.limit_options.poisson_kws["alpha"] = 1e-3

    results = default_method.processDataFile(df)
    default_method.filterResults(results)
    clusters = default_method.processClusters(results)

    widget.drawResultsParticle(
        list(results.values()),
        [QtGui.QColor(255, 0, 0), QtGui.QColor(0, 255, 0), QtGui.QColor(0, 0, 255)],
        ["a", "b", "c"],
        "signal",
    )
    widget.drawResultsHistogram(
        list(results.values()),
        [QtGui.QColor(255, 0, 0), QtGui.QColor(0, 255, 0), QtGui.QColor(0, 0, 255)],
        ["a", "b", "c"],
        "signal",
    )
    widget.drawResultsSpectra(df, next(iter(results.values())))
    widget.drawResultsComposition(
        list(results.values()),
        [QtGui.QColor(255, 0, 0), QtGui.QColor(0, 255, 0), QtGui.QColor(0, 0, 255)],
        "signal",
        clusters,
    )
    widget.drawResultsScatterExpr(
        list(results.values()), "107Ag + 109Ag", "197Au / 2.0", "signal", "signal"
    )

    views = ["particle", "histogram", "composition", "spectra", "scatter"]
    for view in views:
        with qtbot.waitSignal(widget.requestRedraw):
            widget.setView(view)
        assert widget.currentView() == view

    widget.clear()

    for view in views:
        with qtbot.waitSignal(widget.requestRedraw):
            widget.setView(view)
        assert widget.currentView() == view


def test_spcal_instrument_options_dock(qtbot: QtBot):
    dock = SPCalInstrumentOptionsDock(SPCalInstrumentOptions(None, 0.1))
    qtbot.addWidget(dock)
    with qtbot.waitExposed(dock):
        dock.show()

    assert dock.options_widget.uptake.baseValue() is None
    assert dock.options_widget.efficiency.value() == 0.1
    assert not dock.options_widget.action_efficiency.isEnabled()

    with qtbot.waitSignal(dock.optionsChanged):
        dock.options_widget.uptake.setBaseValue(0.3)

    assert dock.options_widget.action_efficiency.isEnabled()
    with qtbot.waitSignal(dock.efficiencyDialogRequested):
        dock.options_widget.button_efficiency.click()

    assert dock.instrumentOptions().uptake == 0.3


def test_spcal_isotope_options_dock(qtbot: QtBot):
    dock = SPCalIsotopeOptionsDock()

    qtbot.addWidget(dock)
    with qtbot.waitExposed(dock):
        dock.show()

    isotopes = [
        ISOTOPE_TABLE[("Fe", 56)],
        ISOTOPE_TABLE[("Fe", 57)],
        ISOTOPE_TABLE[("Ni", 58)],
    ]

    assert len(dock.isotopeOptions()) == 0

    dock.setIsotopes(isotopes)

    assert len(dock.isotopeOptions()) == 3

    dock.setIsotopeOption(isotopes[0], SPCalIsotopeOptions(None, None, 1.0))
    dock.setIsotopeOption(isotopes[1], SPCalIsotopeOptions(1.0, 2.0, 0.5))

    assert dock.optionForIsotope(isotopes[0]).density is None
    assert dock.optionForIsotope(isotopes[1]).density == 1.0
    assert dock.optionForIsotope(isotopes[2]).density is None

    dock.setSignificantFigures(3)

    dock.reset()
    assert len(dock.isotopeOptions()) == 3
    assert dock.optionForIsotope(isotopes[1]).density is None


def test_spcal_limit_options_dock(qtbot: QtBot):
    dock = SPCalLimitOptionsDock(
        SPCalLimitOptions(
            "poisson",
            window_size=100,
            max_iterations=100,
            gaussian_kws={"alpha": 1e-3},
            poisson_kws={"function": "currie", "alpha": 1e-4},
            compound_poisson_kws={"alpha": 1e-5, "sigma": 0.6},
        ),
    )

    qtbot.addWidget(dock)
    with qtbot.waitExposed(dock):
        dock.show()

    assert dock.options_widget.limit_method.currentText() == "Poisson"
    assert dock.options_widget.window_size.value() == 100
    assert dock.options_widget.check_window.isChecked()
    assert dock.options_widget.check_iterative.isChecked()
    assert dock.options_widget.gaussian.alpha.value() == 1e-3
    assert dock.options_widget.poisson.alpha.value() == 1e-4
    assert dock.options_widget.poisson.function == "currie"
    assert dock.options_widget.compound.alpha.value() == 1e-5
    assert dock.options_widget.compound.lognormal_sigma.value() == 0.6

    with qtbot.waitSignal(dock.optionsChanged, timeout=100):
        dock.options_widget.poisson.alpha.setValue(1e-3)

    with qtbot.waitSignal(dock.optionsChanged, timeout=100):
        dock.reset()


def test_spcal_outputs_dock(
    qtbot: QtBot, random_datafile_gen, default_method: SPCalProcessingMethod
):
    default_method.instrument_options.uptake = 1.0
    default_method.instrument_options.efficiency = 0.1
    default_method.isotope_options[ISOTOPE_TABLE[("Fe", 56)]] = SPCalIsotopeOptions(
        1.0, 1.0, 1.0
    )
    default_method.isotope_options[ISOTOPE_TABLE[("Fe", 57)]] = SPCalIsotopeOptions(
        None, None, None
    )
    default_method.isotope_options[ISOTOPE_TABLE[("Ni", 58)]] = SPCalIsotopeOptions(
        1.0, 2.0, 3.0
    )

    df = random_datafile_gen(
        isotopes=[
            ISOTOPE_TABLE[("Fe", 56)],
            ISOTOPE_TABLE[("Fe", 57)],
            ISOTOPE_TABLE[("Ni", 58)],
        ]
    )
    results = list(default_method.processDataFile(df).values())

    dock = SPCalOutputsDock()
    qtbot.addWidget(dock)
    with qtbot.waitExposed(dock):
        dock.show()

    assert dock.view.results_model.rowCount() == 0
    dock.setResults(results)
    assert dock.view.results_model.rowCount() == 3

    orientation = QtCore.Qt.Orientation.Horizontal

    assert (dock.view.results_model.headerData(0, orientation, UnitsRole)) == {}
    assert (
        dock.view.results_model.headerData(1, orientation, UnitsRole)
    ) == number_concentration_units
    for i in range(2, 7):
        assert (
            dock.view.results_model.headerData(i, orientation, UnitsRole)
        ) == signal_units

    dock.updateOutputsForKey("mass")
    assert (
        dock.view.results_model.headerData(1, orientation, UnitsRole)
    ) == mass_concentration_units
    for i in range(2, 7):
        assert (
            dock.view.results_model.headerData(i, orientation, UnitsRole)
        ) == mass_units

    dock.updateOutputsForKey("size")
    assert (
        dock.view.results_model.headerData(1, orientation, UnitsRole)
    ) == number_concentration_units
    for i in range(2, 7):
        assert (
            dock.view.results_model.headerData(i, orientation, UnitsRole)
        ) == size_units

    dock.setSignificantFigures(3)
    delegate = dock.view.itemDelegate()
    assert isinstance(delegate, ValueWidgetDelegate)
    assert delegate.sigfigs == 3

    with qtbot.waitSignal(
        dock.requestCurrentIsotope,
        check_params_cb=lambda iso: iso == ISOTOPE_TABLE[("Fe", 56)],
        timeout=100,
    ):
        dock.view.onHeaderClicked(0)

    dock.reset()
    assert dock.view.results_model.rowCount() == 0


def test_spcal_toolbar(qtbot: QtBot):
    toolbar = SPCalOptionsToolBar()
    qtbot.addWidget(toolbar)
    with qtbot.waitExposed(toolbar):
        toolbar.show()

    assert not toolbar.scatter_actions.isVisible()
    assert len(toolbar.selectedIsotopes()) == 0

    toolbar.setIsotopes(
        [
            ISOTOPE_TABLE[("Fe", 56)],
            ISOTOPE_TABLE[("Fe", 57)],
            ISOTOPE_TABLE[("Ni", 58)],
        ]
    )

    assert toolbar.combo_isotope.currentText() == "56Fe"
    assert toolbar.scatter_x.text() == "56Fe"
    assert toolbar.scatter_y.text() == "57Fe"
    assert len(toolbar.selectedIsotopes()) == 1

    with qtbot.waitSignal(toolbar.isotopeChanged, timeout=100):
        toolbar.combo_isotope.setCurrentIsotope(ISOTOPE_TABLE[("Fe", 57)])

    # test retain current
    toolbar.setIsotopes(
        [
            ISOTOPE_TABLE[("Fe", 56)],
            ISOTOPE_TABLE[("Fe", 57)],
            ISOTOPE_TABLE[("Cu", 63)],
        ]
    )
    assert toolbar.combo_isotope.currentText() == "57Fe"

    with qtbot.waitSignal(toolbar.isotopeChanged, timeout=100):
        toolbar.action_all_isotopes.trigger()
    assert len(toolbar.selectedIsotopes()) == 3

    with qtbot.waitSignal(toolbar.keyChanged, timeout=100):
        toolbar.combo_key.setCurrentIndex(1)

    toolbar.onViewChanged("scatter")
    assert toolbar.scatter_actions.isVisible()

    with qtbot.waitSignal(toolbar.scatterOptionsChanged, timeout=100):
        toolbar.scatter_x.setText("Fe56+Fe56")
        toolbar.scatter_x.editingFinished.emit()

    with qtbot.waitSignal(toolbar.requestFilterDialog, timeout=100):
        toolbar.action_filter.trigger()

    toolbar.reset()


def test_spcal_view_toolbar(qtbot: QtBot):
    toolbar = SPCalViewToolBar()
    qtbot.addWidget(toolbar)
    with qtbot.waitExposed(toolbar):
        toolbar.show()


    assert toolbar.currentView() == "particle"
    assert not toolbar.action_view_options.isEnabled()

    with qtbot.waitSignal(toolbar.viewChanged):
        toolbar.view_actions["histogram"].trigger()

    assert toolbar.currentView() == "histogram"
    assert toolbar.action_view_options.isEnabled()
