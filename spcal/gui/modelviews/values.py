from PySide6 import QtCore, QtWidgets

from spcal.gui.widgets.values import ValueWidget


class ValueWidgetDelegate(QtWidgets.QStyledItemDelegate):
    ErrorRole = QtCore.Qt.ItemDataRole.UserRole + 37
    def __init__(
        self,
        sigfigs: int = 6,
        min: float = 0.0,
        max: float = 1e99,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent=parent)
        self.sigfigs = sigfigs
        self.min, self.max = min, max

    def setMin(self, min: float):
        self.min = min

    def setMax(self, max: float):
        self.max = max

    def setSigFigs(self, sigfigs: int):
        self.sigfigs = sigfigs

    def createEditor(
        self,
        parent: QtWidgets.QWidget,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> QtWidgets.QWidget:
        editor = ValueWidget(
            min=self.min, max=self.max, sigfigs=self.sigfigs, parent=parent
        )
        return editor

    def setEditorData(
        self,
        editor: QtWidgets.QWidget,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ):
        assert isinstance(editor, ValueWidget)
        value = index.data(QtCore.Qt.ItemDataRole.EditRole)
        error = index.data(ValueWidgetDelegate.ErrorRole)
        editor.setValue(value)
        editor.setError(error)

    def setModelData(
        self,
        editor: QtWidgets.QWidget,
        model: QtCore.QAbstractItemModel,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ):
        assert isinstance(editor, ValueWidget)
        value = editor.value()
        model.setData(index, value, QtCore.Qt.ItemDataRole.EditRole)

    def initStyleOption(
        self,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ):
        super().initStyleOption(option, index)
        # Align text to the right
        option.displayAlignment = (  # type: ignore , works
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        # Draw the error if set
        error = index.data(ValueWidgetDelegate.ErrorRole)
        if error is not None:
            option.text = (  # type: ignore
                option.text  # type: ignore
                + " ± "
                + option.locale.toString(float(error), "g", self.sigfigs)  # type: ignore
            )
