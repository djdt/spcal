import numpy as np
from PySide6 import QtCore, QtWidgets

from spcal.gui.util import create_action
from spcal.gui.widgets import UnitsWidget
from spcal.isotope import SPCalIsotopeBase
from spcal.processing.filter import (
    SPCalResultFilter,
    SPCalClusterFilter,
    SPCalValueFilter,
)
from spcal.siunits import mass_units, signal_units, size_units, volume_units


class FilterItemWidget(QtWidgets.QWidget):
    closeRequested = QtCore.Signal(QtWidgets.QWidget)

    KEY_LABELS = {
        "signal": "Intensity",
        "mass": "Mass",
        "size": "Size",
        # "volume": "Volume",
    }
    OPERATION_LABELS = {
        np.greater: ">",
        np.less: "<",
        np.greater_equal: ">=",
        np.less_equal: "<=",
        np.equal: "==",
    }
    OPERATION_PREFER_INVALID = {
        np.greater: False,
        np.less: True,
        np.greater_equal: False,
        np.less_equal: True,
        np.equal: True,
    }

    def __init__(
        self,
        isotopes: list[SPCalIsotopeBase],
        filter: SPCalValueFilter | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        self.isotopes = QtWidgets.QComboBox()
        self.isotopes.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContentsOnFirstShow
        )
        for isotope in isotopes:
            self.isotopes.insertItem(9999, str(isotope), isotope)

        self.key = QtWidgets.QComboBox()
        for key, label in self.KEY_LABELS.items():
            self.key.insertItem(99, label, key)
        self.key.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContentsOnFirstShow
        )
        self.key.currentTextChanged.connect(self.changeUnits)

        self.operation = QtWidgets.QComboBox()
        for op, label in self.OPERATION_LABELS.items():
            self.operation.insertItem(99, label, op)

        self.value = UnitsWidget(units=signal_units, base_value=0.0)
        self.value.combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )

        self.action_close = create_action(
            "list-remove", "Remove", "Remove the filter.", self.close
        )

        self.button_close = QtWidgets.QToolButton()
        self.button_close.setAutoRaise(True)
        self.button_close.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup
        )
        self.button_close.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        self.button_close.setDefaultAction(self.action_close)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.isotopes, 0)
        layout.addWidget(self.key, 0)
        layout.addWidget(self.operation, 0)
        layout.addWidget(self.value, 1)
        layout.addWidget(self.button_close, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        self.setLayout(layout)

        if filter is not None:
            self.setFilter(filter)

    def setFilter(self, filter: SPCalValueFilter):
        index = self.isotopes.findData(
            filter.isotope, role=QtCore.Qt.ItemDataRole.UserRole
        )
        if index == -1:
            raise KeyError(f"missing isotope {filter.isotope}")
        label = FilterItemWidget.KEY_LABELS[filter.key]
        self.isotopes.setCurrentIndex(index)
        self.key.setCurrentText(label)
        self.operation.setCurrentText(
            FilterItemWidget.OPERATION_LABELS[filter.operation]
        )
        self.value.setBaseValue(filter.value)
        self.value.setBestUnit()

    def asFilter(self) -> SPCalResultFilter:
        return SPCalValueFilter(
            self.isotopes.itemData(self.isotopes.currentIndex()),
            self.key.itemData(self.key.currentIndex()),
            self.operation.itemData(self.operation.currentIndex()),
            self.value.baseValue() or 0.0,
            prefer_invalid=FilterItemWidget.OPERATION_PREFER_INVALID[
                self.operation.itemData(self.operation.currentIndex())
            ],
        )

    def close(self) -> bool:
        self.closeRequested.emit(self)
        return super().close()

    def changeUnits(self, key: str):
        if key == "Intensity":
            units = signal_units
        elif key == "Mass":
            units = mass_units
        elif key == "Size":
            units = size_units
        elif key == "Volime":
            units = volume_units
        else:
            raise ValueError("changeUnits: unknown unit")

        self.value.setUnits(units)


class ClusterFilterItemWidget(QtWidgets.QWidget):
    closeRequested = QtCore.Signal(QtWidgets.QWidget)

    def __init__(
        self,
        filter: SPCalClusterFilter | None = None,
        maximum_index: int = 99,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        self.index = QtWidgets.QSpinBox()
        self.index.setPrefix("Cluster index:    ")
        self.index.setMinimum(1)
        self.index.setMaximum(maximum_index)

        self.key = QtWidgets.QComboBox()
        for key, label in FilterItemWidget.KEY_LABELS.items():
            self.key.insertItem(99, label, key)
        self.key.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContentsOnFirstShow
        )

        self.action_close = create_action(
            "list-remove", "Remove", "Remove the filter.", self.close
        )

        self.button_close = QtWidgets.QToolButton()
        self.button_close.setAutoRaise(True)
        self.button_close.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup
        )
        self.button_close.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        self.button_close.setDefaultAction(self.action_close)

        layout = QtWidgets.QVBoxLayout()
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.key, 0)
        hlayout.addWidget(self.index, 0)
        hlayout.addStretch(1)
        hlayout.addWidget(self.button_close, 0, QtCore.Qt.AlignmentFlag.AlignRight)

        layout.addLayout(hlayout, 1)

        frame = QtWidgets.QFrame()
        frame.setFrameShape(QtWidgets.QFrame.Shape.HLine)

        frame2 = QtWidgets.QFrame()
        frame2.setFrameShape(QtWidgets.QFrame.Shape.HLine)

        orlayout = QtWidgets.QHBoxLayout()
        orlayout.addWidget(frame, 1)
        orlayout.addWidget(
            QtWidgets.QLabel("Or"), 0, QtCore.Qt.AlignmentFlag.AlignCenter
        )
        orlayout.addWidget(frame2, 1)

        layout.addLayout(orlayout, 0)

        self.setLayout(layout)

        if filter is not None:
            self.setFilter(filter)

    def setFilter(self, filter: SPCalClusterFilter):
        label = FilterItemWidget.KEY_LABELS[filter.key]
        self.key.setCurrentText(label)
        self.index.setValue(filter.index)

    def asFilter(self) -> SPCalClusterFilter:
        return SPCalClusterFilter(
            self.key.itemData(self.key.currentIndex()), self.index.value()
        )

    def close(self) -> bool:
        self.closeRequested.emit(self)
        return super().close()


class BooleanItemWidget(QtWidgets.QWidget):
    closeRequested = QtCore.Signal(QtWidgets.QWidget)

    def __init__(self, text: str = "Or", parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        frame = QtWidgets.QFrame()
        frame.setFrameShape(QtWidgets.QFrame.Shape.HLine)

        frame2 = QtWidgets.QFrame()
        frame2.setFrameShape(QtWidgets.QFrame.Shape.HLine)

        self.action_close = create_action(
            "list-remove", "Remove", "Remove the filter.", self.close
        )

        self.button_close = QtWidgets.QToolButton()
        self.button_close.setAutoRaise(True)
        self.button_close.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup
        )
        self.button_close.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        self.button_close.setDefaultAction(self.action_close)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(frame, 1)
        layout.addWidget(QtWidgets.QLabel(text), 0, QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(frame2, 1)
        layout.addWidget(self.button_close, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        self.setLayout(layout)

    def close(self) -> bool:
        self.closeRequested.emit(self)
        return super().close()


class FilterDialog(QtWidgets.QDialog):
    filtersChanged = QtCore.Signal(list, list)

    def __init__(
        self,
        isotopes: list[SPCalIsotopeBase],
        filters: list[list[SPCalValueFilter]],
        cluster_filters: list[list[SPCalClusterFilter]],
        number_clusters: int = 0,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Particle Filtering")
        self.setMinimumSize(640, 480)

        self.isotopes = isotopes
        self.number_clusters = number_clusters

        self.list = QtWidgets.QListWidget()
        self.list.setDragEnabled(True)
        self.list.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)

        self.cluster_list = QtWidgets.QListWidget()
        self.cluster_list.setDragEnabled(True)
        self.cluster_list.setDragDropMode(
            QtWidgets.QAbstractItemView.DragDropMode.InternalMove
        )

        self.action_add = create_action(
            "list-add", "Add Filter", "Add a new filter.", lambda: self.addFilter(None)
        )
        self.action_or = create_action(
            "",
            "Or",
            "Add an or group.",
            lambda: self.addBooleanOr(),
        )

        self.action_cluster_add = create_action(
            "list-add",
            "Add Filter",
            "Add a new filter.",
            lambda: self.addClusterFilter(None),
        )
        self.button_bar = QtWidgets.QToolBar()
        self.button_bar.addAction(self.action_add)
        self.button_bar.addAction(self.action_or)

        self.cluster_bar = QtWidgets.QToolBar()
        self.cluster_bar.addAction(self.action_cluster_add)
        # self.cluster_bar.addAction(self.action_cluster_or)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        gbox_comp = QtWidgets.QGroupBox("Composition Filters")
        gbox_comp_layout = QtWidgets.QVBoxLayout()
        gbox_comp_layout.addWidget(self.button_bar, 0)
        gbox_comp_layout.addWidget(self.list, 1)
        gbox_comp.setLayout(gbox_comp_layout)

        gbox_cluster = QtWidgets.QGroupBox("Cluster Filters")
        gbox_cluster_layout = QtWidgets.QVBoxLayout()
        gbox_cluster_layout.addWidget(self.cluster_bar, 0)
        gbox_cluster_layout.addWidget(self.cluster_list, 1)
        gbox_cluster.setLayout(gbox_cluster_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.button_bar, 0)
        layout.addWidget(gbox_comp, 2)
        layout.addWidget(gbox_cluster, 1)
        layout.addWidget(self.button_box, 0)
        self.setLayout(layout)

        # add the filters
        for i in range(len(filters)):
            for filter in filters[i]:
                if filter.isotope in self.isotopes:
                    self.addFilter(filter)
            if i < len(filters) - 1:
                self.addBooleanOr()
        # add the vluster filters
        for i in range(len(cluster_filters)):
            for filter in cluster_filters[i]:
                self.addClusterFilter(filter)
            if i < len(cluster_filters) - 1:
                self.addBooleanOr()

    def isComplete(self) -> bool:
        for i in range(self.list.count()):
            widget = self.list.itemWidget(self.list.item(i))
            if isinstance(widget, FilterItemWidget):
                if widget.value.baseValue() is None:
                    return False
        return True

    def completeChanged(self):
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            self.isComplete()
        )

    def addFilter(self, filter: SPCalValueFilter | None = None):
        widget = FilterItemWidget(self.isotopes, filter=filter)
        widget.value.baseValueChanged.connect(self.completeChanged)
        self.addWidget(widget)

    def addBooleanOr(self):
        widget = BooleanItemWidget()
        self.addWidget(widget)

    def addWidget(self, widget: QtWidgets.QWidget):
        assert isinstance(
            widget, (FilterItemWidget, ClusterFilterItemWidget, BooleanItemWidget)
        )
        widget.closeRequested.connect(self.removeItem)
        item = QtWidgets.QListWidgetItem()
        self.list.insertItem(self.list.count(), item)
        self.list.setItemWidget(item, widget)
        item.setSizeHint(widget.sizeHint())

    def removeItem(self, widget: FilterItemWidget):
        for i in range(self.list.count()):
            item = self.list.item(i)
            if self.list.itemWidget(item) == widget:
                self.list.takeItem(i)
                break

    def addClusterFilter(self, filter: SPCalClusterFilter | None = None):
        widget = ClusterFilterItemWidget(
            filter=filter, maximum_index=self.number_clusters
        )
        self.addClusterWidget(widget)

    def addClusterWidget(self, widget: QtWidgets.QWidget):
        assert isinstance(widget, ClusterFilterItemWidget)
        widget.closeRequested.connect(self.removeClusterItem)
        item = QtWidgets.QListWidgetItem()
        self.cluster_list.insertItem(self.cluster_list.count(), item)
        self.cluster_list.setItemWidget(item, widget)
        item.setSizeHint(widget.sizeHint())

    def removeClusterItem(self, widget: FilterItemWidget):
        for i in range(self.cluster_list.count()):
            item = self.cluster_list.item(i)
            if self.cluster_list.itemWidget(item) == widget:
                self.cluster_list.takeItem(i)
                break

    def accept(self):
        filters = []
        group = []
        for i in range(self.list.count()):
            widget = self.list.itemWidget(self.list.item(i))
            if isinstance(widget, FilterItemWidget):
                if widget.value.baseValue() is not None:
                    group.append(widget.asFilter())
            elif isinstance(widget, BooleanItemWidget):
                if len(group) > 0:
                    filters.append(group)
                    group = []
        filters.append(group)

        cluster_filters = []
        for i in range(self.cluster_list.count()):
            widget = self.cluster_list.itemWidget(self.cluster_list.item(i))
            assert isinstance(widget, ClusterFilterItemWidget)
            cluster_filters.append([widget.asFilter()])

        self.filtersChanged.emit(filters, cluster_filters)
        super().accept()
