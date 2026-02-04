import numpy as np
from pytestqt.qtbot import QtBot

from spcal.gui.dialogs.filter import BooleanItemWidget, FilterDialog
from spcal.isotope import ISOTOPE_TABLE
from spcal.processing.filter import SPCalClusterFilter, SPCalValueFilter


def test_filter_dialog_empty(qtbot: QtBot):
    isotopes = [ISOTOPE_TABLE[("Ag", 107)], ISOTOPE_TABLE[("Ag", 109)], ISOTOPE_TABLE[("Au", 197)]]
    dlg = FilterDialog(isotopes, [], [], number_clusters=10)
    qtbot.addWidget(dlg)
    with qtbot.wait_exposed(dlg):
        dlg.show()

    # Will all collapse
    dlg.action_add.trigger()
    dlg.action_add.trigger()
    dlg.action_or.trigger()
    dlg.action_add.trigger()
    dlg.action_add.trigger()
    dlg.action_cluster_add.trigger()
    dlg.list.itemWidget(dlg.list.item(0)).value.setBaseValue(1.0)
    dlg.cluster_list.itemWidget(dlg.cluster_list.item(0)).index.setValue(10)

    assert dlg.cluster_list.itemWidget(dlg.cluster_list.item(0)).index.maximum() == 10

    def check_filters(
        filters: list[list[SPCalValueFilter]], cluster_filters: list[SPCalClusterFilter]
    ) -> bool:
        if len(filters) != 1:
            return False
        if len(filters[0]) != 1:
            return False
        if filters[0][0].value != 1.0:
            return False
        if len(cluster_filters) != 1:
            return False
        if cluster_filters[0].idx != 9:
            return False
        return True

    with qtbot.wait_signal(dlg.filtersChanged, check_params_cb=check_filters):
        dlg.accept()


# def test_filter_dialog_filters(qtbot: QtBot):
#     filters = [
#         [Filter("a", "mass", ">", 1.0), Filter("b", "mass", ">", 2.0)],
#         [Filter("c", "mass", "<", 3.0)],
#     ]
#     dlg = FilterDialog(names, filters, [])
#     qtbot.addWidget(dlg)
#     with qtbot.wait_exposed(dlg):
#         dlg.show()
#
#     # Check loaded correctly
#     assert dlg.list.itemWidget(dlg.list.item(0)).names.currentText() == "a"
#     assert dlg.list.itemWidget(dlg.list.item(0)).unit.currentText() == "Mass"
#     assert dlg.list.itemWidget(dlg.list.item(0)).value.baseValue() == 1.0
#     assert dlg.list.itemWidget(dlg.list.item(1)).names.currentText() == "b"
#     assert isinstance(dlg.list.itemWidget(dlg.list.item(2)), BooleanItemWidget)
#     assert dlg.list.itemWidget(dlg.list.item(3)).names.currentText() == "c"
#     assert dlg.list.itemWidget(dlg.list.item(3)).operation.currentText() == "<"
#
#     # Remove the or
#     dlg.list.itemWidget(dlg.list.item(2)).close()
#
#     def check_filters(filters: list[list[Filter]], cluster_filters: list) -> bool:
#         if len(filters) != 1:
#             return False
#         if len(filters[0]) != 3:
#             return False
#         for filter, val in zip(filters[0], [1.0, 2.0, 3.0]):
#             if filter.value != val:
#                 return False
#         return True
#
#     with qtbot.wait_signal(dlg.filtersChanged, check_params_cb=check_filters):
#         dlg.accept()
