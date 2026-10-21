from qtpy.QtWidgets import QUndoCommand, QTableWidgetItem
from qtpy.QtCore import Qt


class SetCellCommand(QUndoCommand):
    """Reversible single-cell edit on a DataTable + its widget view."""

    _MERGE_ID = 0xA1

    def __init__(self, widget, table, row: int, col: int, old_val, new_val):
        super().__init__("Edit cell")
        self._widget = widget
        self._table = table
        self._row = row
        self._col = col
        self._old = old_val
        self._new = new_val
        self._first = True  # widget already shows new_val on first push

    def _apply(self, value, skip_widget: bool = False):
        self._table.set_value(self._row, self._col, value)
        if not skip_widget:
            self._widget._updating = True
            item = self._widget.item(self._row, self._col)
            if item is None:
                item = QTableWidgetItem()
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._widget.setItem(self._row, self._col, item)
            item.setText("" if value is None else str(value))
            self._widget._updating = False

    def redo(self):
        self._apply(self._new, skip_widget=self._first)
        self._first = False

    def undo(self):
        self._apply(self._old)

    def id(self):
        return self._MERGE_ID

    def mergeWith(self, other) -> bool:
        if (
            isinstance(other, SetCellCommand)
            and other._widget is self._widget
            and other._row == self._row
            and other._col == self._col
        ):
            self._new = other._new
            return True
        return False


class AddColumnCommand(QUndoCommand):
    def __init__(self, widget, table, name: str):
        super().__init__("Add column")
        self._widget = widget
        self._table = table
        self._name = name

    def redo(self):
        self._table.add_column(self._name)
        self._widget._refresh()

    def undo(self):
        self._table.remove_column(len(self._table.columns) - 1)
        self._widget._refresh()


class RemoveColumnCommand(QUndoCommand):
    def __init__(self, widget, table, index: int):
        super().__init__("Remove column")
        self._widget = widget
        self._table = table
        self._index = index
        # keep the column object itself: values AND exclusions (and any
        # future per-column state) come back on undo
        self._column = table.columns[index]

    def redo(self):
        self._table.remove_column(self._index)
        self._widget._refresh()

    def undo(self):
        self._table.columns.insert(self._index, self._column)
        self._widget._refresh()
