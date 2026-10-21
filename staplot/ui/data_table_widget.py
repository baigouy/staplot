import csv
import io

import numpy as np
from qtpy.QtWidgets import (
    QTableWidget, QTableWidgetItem, QAbstractItemView,
    QMenu, QHeaderView, QInputDialog, QApplication, QMessageBox,
)
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QFont, QKeySequence

from ..models.project import DataTable
from .undo_commands import SetCellCommand, AddColumnCommand, RemoveColumnCommand


class DataTableWidget(QTableWidget):
    """Spreadsheet-like editor bound to a DataTable model."""

    data_changed = Signal()

    _HEADER_BG = QColor("#2d5fa6")
    _HEADER_FG = QColor("white")
    _EMPTY_ROWS = 24

    def __init__(self, parent=None):
        super().__init__(parent)
        self._table: DataTable | None = None
        self._updating = False
        self._undo_stack = None  # set by MainWindow after creation

        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.horizontalHeader().setDefaultSectionSize(110)
        self.verticalHeader().setDefaultSectionSize(22)
        self.verticalHeader().setVisible(True)

        bold_h = QFont()
        bold_h.setBold(True)
        self.horizontalHeader().setFont(bold_h)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.itemChanged.connect(self._on_item_changed)
        self.horizontalHeader().sectionDoubleClicked.connect(self._rename_column)

        self.setStyleSheet("""
            QTableWidget { gridline-color: #e0e0e0; border: none; }
            QHeaderView::section {
                background-color: #2d5fa6; color: white;
                padding: 4px; border: 1px solid #1a4a8f; font-weight: bold;
            }
        """)

    # ------------------------------------------------------------------
    def load_table(self, table: DataTable):
        self._table = table
        self._refresh()

    _EXCLUDED_BG = QColor("#ffe0e0")   # light red background for excluded cells

    def _refresh(self):
        if self._table is None:
            return
        self._updating = True
        n_cols = len(self._table.columns)
        n_rows = max(self._table.num_rows() + self._EMPTY_ROWS, self._EMPTY_ROWS)

        self.setColumnCount(n_cols)
        self.setRowCount(n_rows)
        self.setHorizontalHeaderLabels([c.name for c in self._table.columns])

        for col_idx, col in enumerate(self._table.columns):
            for row_idx, val in enumerate(col.values):
                text = "" if val is None or (isinstance(val, float) and np.isnan(val)) else str(val)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if row_idx in col.excluded:
                    item.setBackground(self._EXCLUDED_BG)
                    item.setForeground(QColor("#aa0000"))
                self.setItem(row_idx, col_idx, item)

        self._updating = False

    # ------------------------------------------------------------------
    def _on_item_changed(self, item: QTableWidgetItem):
        if self._updating or self._table is None:
            return
        row, col_idx = item.row(), item.column()
        text = item.text().strip()
        new_val: float | None
        if text == "":
            new_val = None
        else:
            try:
                new_val = float(text)
            except ValueError:
                item.setText("")
                return

        # Read old value from model before modifying
        old_val = None
        if col_idx < len(self._table.columns):
            col = self._table.columns[col_idx]
            if row < len(col.values):
                old_val = col.values[row]

        if old_val == new_val:
            return

        if self._undo_stack is not None:
            cmd = SetCellCommand(self, self._table, row, col_idx, old_val, new_val)
            self._undo_stack.push(cmd)
        else:
            self._table.set_value(row, col_idx, new_val)
        self.data_changed.emit()

    def _rename_column(self, index: int):
        if self._table is None or index >= len(self._table.columns):
            return
        old_name = self._table.columns[index].name
        new_name, ok = QInputDialog.getText(self, "Rename Column", "Column name:", text=old_name)
        if ok and new_name.strip() and new_name.strip() != old_name:
            try:
                self._table.rename_column(index, new_name.strip())
            except ValueError as exc:
                QMessageBox.warning(self, "Rename Column", str(exc))
                return
            self.setHorizontalHeaderItem(index, QTableWidgetItem(new_name.strip()))
            self.data_changed.emit()

    # ------------------------------------------------------------------
    def add_column(self):
        if self._table is None:
            return
        if self._undo_stack is not None:
            self._undo_stack.push(AddColumnCommand(self, self._table, None))
        else:
            self._table.add_column()
            self._refresh()

    def remove_selected_column(self):
        if self._table is None:
            return
        cols = sorted({i.column() for i in self.selectedItems()}, reverse=True)
        for col_idx in cols:
            if self._undo_stack is not None:
                self._undo_stack.push(RemoveColumnCommand(self, self._table, col_idx))
            else:
                self._table.remove_column(col_idx)
        if not self._undo_stack:
            self._refresh()

    # ------------------------------------------------------------------
    def _show_context_menu(self, pos):
        menu = QMenu(self)
        menu.addAction("Add Column", self.add_column)
        menu.addAction("Remove Column", self.remove_selected_column)
        menu.addSeparator()
        menu.addAction("Paste from Clipboard", self.paste_from_clipboard)
        menu.addSeparator()
        # Exclude / include selected rows
        rows = sorted({i.row() for i in self.selectedItems()})
        if rows and self._table:
            # Check if any selected rows in any column are currently excluded
            any_excluded = any(
                row in col.excluded
                for row in rows
                for col in self._table.columns
            )
            label = "Include selected rows" if any_excluded else "Exclude selected rows"
            menu.addAction(label, lambda r=rows: self._toggle_exclude_rows(r))
        menu.exec(self.viewport().mapToGlobal(pos))

    def _toggle_exclude_rows(self, rows: list[int]):
        if self._table is None:
            return
        for row in rows:
            for col in self._table.columns:
                col.toggle_excluded(row)
        self._refresh()
        self.data_changed.emit()

    # ------------------------------------------------------------------
    @staticmethod
    def _can_float(s: str) -> bool:
        """Return True if s parses as a float or is empty/whitespace."""
        s = s.strip()
        if s == "":
            return True
        try:
            float(s)
            return True
        except ValueError:
            return False

    def paste_from_clipboard(self):
        if self._table is None:
            return
        text = QApplication.clipboard().text()
        if not text.strip():
            return

        # Handle both tab-separated (Excel) and comma-separated (CSV) pasting
        first_line = text.splitlines()[0] if text.strip() else ""
        delimiter = "\t" if "\t" in first_line else ","
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        raw_rows = [r for r in reader if any(c.strip() for c in r)]
        if not raw_rows:
            return

        # ── Header detection ─────────────────────────────────────────
        # A row is treated as a header if ANY non-empty cell is non-numeric
        first_row = raw_rows[0]
        is_header = any(
            c.strip() and not self._can_float(c)
            for c in first_row
        )

        if is_header:
            headers = [h.strip() for h in first_row]
            data_rows = raw_rows[1:]
            # Map header name → column index, adding new columns as needed
            name_to_idx = {col.name: idx for idx, col in enumerate(self._table.columns)}
            col_mapping: list[int] = []
            for h in headers:
                if not h:
                    col_mapping.append(-1)   # skip blank header cells
                    continue
                if h not in name_to_idx:
                    new_idx = len(self._table.columns)
                    self._table.add_column(h)
                    name_to_idx[h] = new_idx
                col_mapping.append(name_to_idx[h])

            for r, row_data in enumerate(data_rows):
                for c_idx, cell in enumerate(row_data):
                    if c_idx >= len(col_mapping):
                        break
                    col_idx = col_mapping[c_idx]
                    if col_idx < 0:
                        continue
                    try:
                        val = float(cell.strip()) if cell.strip() else None
                    except ValueError:
                        val = None
                    self._table.set_value(r, col_idx, val)
        else:
            # No headers: paste at the current cursor/selection position
            start_row = min((i.row() for i in self.selectedItems()), default=0)
            start_col = min((i.column() for i in self.selectedItems()), default=0)
            for r, row_data in enumerate(raw_rows):
                for c, cell in enumerate(row_data):
                    col_idx = start_col + c
                    while col_idx >= len(self._table.columns):
                        self._table.add_column()
                    try:
                        val = float(cell.strip()) if cell.strip() else None
                    except ValueError:
                        val = None
                    self._table.set_value(start_row + r, col_idx, val)

        self._refresh()
        self.data_changed.emit()

    # ------------------------------------------------------------------
    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Paste):
            self.paste_from_clipboard()
        elif event.key() == Qt.Key.Key_Delete:
            for item in self.selectedItems():
                row, col_idx = item.row(), item.column()
                if self._table:
                    self._table.set_value(row, col_idx, None)
                item.setText("")
        else:
            super().keyPressEvent(event)
