import numpy as np
from pytestqt.qtbot import QtBot

from spcal.gui.dialogs.filter import (
    BooleanItemWidget,
    FilterDialog,
    FilterItemWidget,
    ClusterFilterItemWidget,
)
from spcal.isotope import ISOTOPE_TABLE
from spcal.processing.filter import SPCalClusterFilter, SPCalValueFilter


def test_filter_dialog_empty(qtbot: QtBot):
    isotopes = [
        ISOTOPE_TABLE[("Ag", 107)],
        ISOTOPE_TABLE[("Ag", 109)],
        ISOTOPE_TABLE[("Au", 197)],
    ]
    dlg = FilterDialog(isotopes, [[]], [[]], number_clusters=10)  # type: ignore
    qtbot.addWidget(dlg)
    with qtbot.wait_exposed(dlg):
        dlg.show()

    # Will all collapse
    dlg.action_add.trigger()
    dlg.action_or.trigger()
    dlg.action_add.trigger()
    dlg.action_add.trigger()
    dlg.action_cluster_add.trigger()

    filter = dlg.list.itemWidget(dlg.list.item(0))
    assert isinstance(filter, FilterItemWidget)
    filter.value.setBaseValue(1.0)

    assert filter.isotopes.count() == 3

    filter = dlg.cluster_list.itemWidget(dlg.cluster_list.item(0))
    assert isinstance(filter, ClusterFilterItemWidget)
    filter.index.setValue(1)

    assert filter.index.maximum() == 10

    def check_filters(
        filters: list[list[SPCalValueFilter]],
        cluster_filters: list[list[SPCalClusterFilter]],
    ) -> bool:
        print(filters)
        if len(filters) != 2:
            return False
        if len(filters[0]) != 1:
            return False
        if len(filters[1]) != 2:
            return False
        if filters[0][0].value != 1.0:
            return False
        if filters[1][0].value != 0.0:
            return False
        if len(cluster_filters) != 1:
            return False
        if len(cluster_filters[0]) != 1:
            return False
        if cluster_filters[0][0].index != 1:
            return False
        return True

    with qtbot.wait_signal(
        dlg.filtersChanged, check_params_cb=check_filters, timeout=100
    ):
        dlg.accept()


def test_filter_dialog_filters(qtbot: QtBot):
    isotopes = [
        ISOTOPE_TABLE[("Ag", 107)],
        ISOTOPE_TABLE[("Ag", 109)],
        ISOTOPE_TABLE[("Au", 197)],
    ]
    filters = [
        [
            SPCalValueFilter(isotopes[0], "signal", np.greater, 100.0),
            SPCalValueFilter(isotopes[1], "size", np.less, 200.0),
        ],
        [
            SPCalValueFilter(isotopes[2], "mass", np.less_equal, 10.0),
        ],
    ]
    dlg = FilterDialog(isotopes, filters, [[SPCalClusterFilter("mass", 7)]], 10)  # type: ignore
    qtbot.addWidget(dlg)
    with qtbot.wait_exposed(dlg):
        dlg.show()

    # Check loaded correctly
    for i in range(dlg.list.count()):
        w = dlg.list.itemWidget(dlg.list.item(i))
        if i == 0:
            assert isinstance(w, FilterItemWidget)
            assert w.isotopes.currentText() == "107Ag"
            assert w.key.currentText() == "Intensity"
            assert w.operation.currentText() == ">"
            assert w.value.unit() == "cts"
            assert w.value.baseValue() == 100.0
        elif i == 1:
            assert isinstance(w, FilterItemWidget)
            assert w.isotopes.currentText() == "109Ag"
            assert w.key.currentText() == "Size"
            assert w.operation.currentText() == "<"
            assert w.value.unit() == "m"
            assert w.value.baseValue() == 200.0
        elif i == 2:
            assert isinstance(w, BooleanItemWidget)
        elif i == 3:
            assert isinstance(w, FilterItemWidget)

    w = dlg.cluster_list.itemWidget(dlg.cluster_list.item(0))
    assert isinstance(w, ClusterFilterItemWidget)
    assert w.index.value() == 7

    def check_filters(
        filters: list[list[SPCalValueFilter]],
        cluster_filters: list[list[SPCalClusterFilter]],
    ) -> bool:
        if len(filters) != 2:
            return False
        if len(filters[0]) != 2:
            return False
        if filters[0][0].value != 100.0:
            return False
        if filters[0][1].value != 200.0:
            return False
        if filters[1][0].value != 10.0:
            return False
        if len(cluster_filters) != 1 or len(cluster_filters[0]) != 1:
            return False
        if cluster_filters[0][0].key != "mass" or cluster_filters[0][0].index != 7:
            return False
        return True

    with qtbot.wait_signal(
        dlg.filtersChanged, check_params_cb=check_filters, timeout=100
    ):
        dlg.accept()
