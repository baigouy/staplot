"""
Spreadsheet widget for GroupedDataTable.

Layout mirrors Prism's Grouped format:
  - Column groups  = Factor A  (major header, spans n_reps columns each)
  - Row labels     = Factor B  (row headers)
  - Replicates     = sub-columns A, B, C… under each group
"""

import numpy as np
from qtpy.QtWidgets import (
    QTableWidget, QTableWidgetItem, QAbstractItemView,
    QMenu, QHeaderView, QInputDialog,
)
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QFont, QBrush

from ..models.project import GroupedDataTable

_GROUP_COLORS = [
    QColor("#dce8f5"), QColor("#fde9d0"), QColor("#d9f0e1"),
    QColor("#f5e0f0"), QColor("#e8e8d5"), QColor("#dde8f5"),
]


class GroupedTableWidget(QTableWidget):
    data_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._table: GroupedDataTable | None = None
        self._updating = False

        self.setAlternatingRowColors(False)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.horizontalHeader().setDefaultSectionSize(90)
        self.verticalHeader().setDefaultSectionSize(22)

        bold = QFont()
        bold.setBold(True)
        self.horizontalHeader().setFont(bold)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)
        self.itemChanged.connect(self._on_item_changed)

        self.setStyleSheet("""
            QTableWidget { gridline-color: #e0e0e0; border: none; }
        """)

    # ------------------------------------------------------------------
    def load_table(self, table: GroupedDataTable):
        self._table = table
        self._refresh()

    def _refresh(self):
        if self._table is None:
            return
        t = self._table
        self._updating = True

        n_groups = len(t.col_groups)
        n_reps = t.n_reps
        n_rows = len(t.row_labels)
        n_data_cols = n_groups * n_reps

        self.setRowCount(n_rows + 1)     # +1 for group-header pseudo-row
        self.setColumnCount(n_data_cols)

        # Build horizontal header labels: "Group A – 1", "Group A – 2", …
        h_labels = []
        for cg in t.col_groups:
            for rep in range(n_reps):
                h_labels.append(f"{cg} ({chr(65 + rep)})")
        self.setHorizontalHeaderLabels(h_labels)

        # Row headers: "Group header" + row_labels
        v_labels = ["↑ Group"] + list(t.row_labels)
        self.setVerticalHeaderLabels(v_labels)

        # Row 0: group-name pseudo-header (merged visually by color)
        for gi, cg in enumerate(t.col_groups):
            col_start = gi * n_reps
            bg = _GROUP_COLORS[gi % len(_GROUP_COLORS)]
            for rep in range(n_reps):
                ci = col_start + rep
                item = QTableWidgetItem(cg if rep == 0 else "")
                item.setBackground(bg)
                item.setFont(QFont() if rep else self._bold_font())
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setData(Qt.ItemDataRole.UserRole, ("group_header", gi, cg))
                self.setItem(0, ci, item)

        # Data rows
        for ri, rl in enumerate(t.row_labels):
            for gi, cg in enumerate(t.col_groups):
                col_start = gi * n_reps
                bg = _GROUP_COLORS[gi % len(_GROUP_COLORS)].lighter(110)
                vals = t.data.get(cg, {}).get(rl, [])
                for rep in range(n_reps):
                    ci = col_start + rep
                    val = vals[rep] if rep < len(vals) else None
                    text = "" if val is None or (isinstance(val, float) and np.isnan(val)) else str(val)
                    item = QTableWidgetItem(text)
                    item.setBackground(bg)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    self.setItem(ri + 1, ci, item)

        self._updating = False

    def _bold_font(self) -> QFont:
        f = QFont()
        f.setBold(True)
        return f

    # ------------------------------------------------------------------
    def _on_item_changed(self, item: QTableWidgetItem):
        if self._updating or self._table is None:
            return
        row, ci = item.row(), item.column()
        role_data = item.data(Qt.ItemDataRole.UserRole)

        if row == 0:
            # Rename group header
            if role_data and role_data[0] == "group_header":
                gi = role_data[1]
                new_name = item.text().strip()
                if new_name and gi < len(self._table.col_groups):
                    old = self._table.col_groups[gi]
                    if old != new_name:
                        self._table.col_groups[gi] = new_name
                        if old in self._table.data:
                            self._table.data[new_name] = self._table.data.pop(old)
                        self._refresh()
                        self.data_changed.emit()
            return

        ri = row - 1
        gi = ci // self._table.n_reps
        rep = ci % self._table.n_reps

        if ri >= len(self._table.row_labels) or gi >= len(self._table.col_groups):
            return

        rl = self._table.row_labels[ri]
        cg = self._table.col_groups[gi]
        text = item.text().strip()
        try:
            val = float(text) if text else None
        except ValueError:
            item.setText("")
            return
        self._table.set_value(cg, rl, rep, val)
        self.data_changed.emit()

    # ------------------------------------------------------------------
    def _context_menu(self, pos):
        menu = QMenu(self)
        menu.addAction("Add Column Group", self._add_col_group)
        menu.addAction("Add Row", self._add_row)
        menu.addSeparator()
        menu.addAction("Add Replicate Column", self._add_rep)
        menu.exec(self.viewport().mapToGlobal(pos))

    def _add_col_group(self):
        if self._table is None:
            return
        name, ok = QInputDialog.getText(self, "New Group", "Group name:")
        if ok and name.strip():
            self._table.add_col_group(name.strip())
            self._refresh()

    def _add_row(self):
        if self._table is None:
            return
        name, ok = QInputDialog.getText(self, "New Row", "Row label:")
        if ok and name.strip():
            self._table.add_row_label(name.strip())
            self._refresh()

    def _add_rep(self):
        if self._table is None:
            return
        self._table.n_reps += 1
        for cg in self._table.col_groups:
            for rl in self._table.row_labels:
                self._table.data.setdefault(cg, {}).setdefault(rl, []).append(None)
        self._refresh()
