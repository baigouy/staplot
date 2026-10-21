"""Contingency table (r × c) data-entry widget."""

import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QInputDialog, QHeaderView, QLabel,
)
from qtpy.QtCore import Qt, Signal

from ..models.project import ContingencyTable


class ContingencyTableWidget(QWidget):
    data_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._table: ContingencyTable | None = None
        self._loading = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Toolbar ───────────────────────────────────────────────────
        btn_bar = QHBoxLayout()
        for label, slot in [
            ("+ Row",  self._add_row),
            ("− Row",  self._remove_row),
            ("+ Col",  self._add_col),
            ("− Col",  self._remove_col),
        ]:
            btn = QPushButton(label)
            btn.setFixedWidth(64)
            btn.clicked.connect(slot)
            btn_bar.addWidget(btn)
        btn_bar.addStretch()
        layout.addLayout(btn_bar)

        # ── Table ─────────────────────────────────────────────────────
        self._tw = QTableWidget()
        self._tw.setAlternatingRowColors(True)
        self._tw.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._tw.itemChanged.connect(self._on_cell_changed)
        self._tw.setStyleSheet("""
            QTableWidget { gridline-color: #ddd; border: none; }
            QHeaderView::section {
                background-color: #2d5fa6; color: white;
                padding: 4px; border: 1px solid #1a4a8f; font-weight: bold;
            }
        """)
        layout.addWidget(self._tw)
        layout.addWidget(QLabel(
            "Enter integer counts; double-click a header to rename it."
        ))

        # Allow header renaming
        self._tw.horizontalHeader().sectionDoubleClicked.connect(self._rename_col)
        self._tw.verticalHeader().sectionDoubleClicked.connect(self._rename_row)

    # ------------------------------------------------------------------
    def load_table(self, table: ContingencyTable):
        self._table = table
        self._reload()

    def _reload(self):
        if self._table is None:
            return
        t = self._table
        t._ensure_shape()
        nrows, ncols = len(t.row_labels), len(t.col_labels)

        self._loading = True
        self._tw.blockSignals(True)
        self._tw.setRowCount(nrows)
        self._tw.setColumnCount(ncols)
        self._tw.setHorizontalHeaderLabels(t.col_labels)
        self._tw.setVerticalHeaderLabels(t.row_labels)

        for r, rl in enumerate(t.row_labels):
            for c, cl in enumerate(t.col_labels):
                val = t.get_count(rl, cl)
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._tw.setItem(r, c, item)

        self._tw.blockSignals(False)
        self._loading = False

    # ------------------------------------------------------------------
    def _on_cell_changed(self, item: QTableWidgetItem):
        if self._loading or self._table is None:
            return
        r, c = item.row(), item.column()
        t = self._table
        if r >= len(t.row_labels) or c >= len(t.col_labels):
            return
        try:
            val = int(item.text())
        except ValueError:
            val = 0
        t.set_count(t.row_labels[r], t.col_labels[c], val)
        self.data_changed.emit()

    # ------------------------------------------------------------------
    def _add_row(self):
        if self._table is None:
            return
        n = len(self._table.row_labels) + 1
        name, ok = QInputDialog.getText(self, "Add Row", "Row label:", text=f"Row {n}")
        if ok and name.strip():
            self._table.add_row(name.strip())
            self._reload()

    def _remove_row(self):
        if self._table is None:
            return
        r = self._tw.currentRow()
        if 0 <= r < len(self._table.row_labels):
            self._table.remove_row(self._table.row_labels[r])
            self._reload()

    def _add_col(self):
        if self._table is None:
            return
        n = len(self._table.col_labels) + 1
        name, ok = QInputDialog.getText(self, "Add Column", "Column label:", text=f"Col {n}")
        if ok and name.strip():
            self._table.add_col(name.strip())
            self._reload()

    def _remove_col(self):
        if self._table is None:
            return
        c = self._tw.currentColumn()
        if 0 <= c < len(self._table.col_labels):
            self._table.remove_col(self._table.col_labels[c])
            self._reload()

    def _rename_row(self, idx: int):
        if self._table is None or idx >= len(self._table.row_labels):
            return
        old = self._table.row_labels[idx]
        name, ok = QInputDialog.getText(self, "Rename Row", "Label:", text=old)
        if ok and name.strip() and name.strip() != old:
            self._table.row_labels[idx] = name.strip()
            if old in self._table.counts:
                self._table.counts[name.strip()] = self._table.counts.pop(old)
            self._reload()

    def _rename_col(self, idx: int):
        if self._table is None or idx >= len(self._table.col_labels):
            return
        old = self._table.col_labels[idx]
        name, ok = QInputDialog.getText(self, "Rename Column", "Label:", text=old)
        if ok and name.strip() and name.strip() != old:
            self._table.col_labels[idx] = name.strip()
            self._reload()
