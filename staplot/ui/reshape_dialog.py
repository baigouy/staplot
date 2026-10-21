"""Reshape table dialog — Wide ↔ Long format conversion."""

import numpy as np
import pandas as pd

from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QComboBox, QLineEdit,
    QPushButton, QDialogButtonBox, QLabel, QListWidget,
    QTabWidget, QWidget, QAbstractItemView, QHBoxLayout,
)
from qtpy.QtCore import Qt

from ..models.project import DataTable


def _wide_to_long(
    source: DataTable,
    value_cols: list[str],
    var_col: str = "Group",
    val_col: str = "Value",
    result_name: str | None = None,
) -> DataTable:
    """
    Stack *value_cols* into two new columns (var_col, val_col).
    All other columns are replicated as identifier variables.
    """
    df = source.to_dataframe()
    id_cols = [c for c in df.columns if c not in value_cols]
    melted  = df.melt(id_vars=id_cols, value_vars=value_cols,
                      var_name=var_col, value_name=val_col)
    melted  = melted.reset_index(drop=True)

    name = result_name or f"{source.name} (long)"
    t    = DataTable(name)
    for col in melted.columns:
        t.add_column(col)
    for r, row in melted.iterrows():
        for c_idx, col in enumerate(melted.columns):
            raw = row[col]
            if isinstance(raw, float) and np.isnan(raw):
                val = None
            elif isinstance(raw, (int, float)):
                val = float(raw)
            else:
                # text — store as is; DataTableWidget shows "" for None
                val = None  # skip non-numeric for now
            t.set_value(r, c_idx, val)
    # Re-fill text values (var_col)
    var_idx = list(melted.columns).index(var_col)
    for r, row in melted.iterrows():
        raw = row[var_col]
        t.set_value(r, var_idx, None)          # clear numeric placeholder
        # store as a special string marker — not ideal but workable
        # We'll just store the group name encoded as a dummy float
        # For the var_col we simply leave it as None (not meaningful numerically)
        # and rely on the user to see the group names separately.
        # Better: encode as indices
    # Actually map group names to indices and add a legend column
    group_map = {g: i for i, g in enumerate(value_cols)}
    for r, row in melted.iterrows():
        t.set_value(r, var_idx, float(group_map.get(row[var_col], 0)))
    return t


def _wide_to_long_clean(
    source: DataTable,
    value_cols: list[str],
    var_col: str = "Group",
    val_col: str = "Value",
    result_name: str | None = None,
) -> DataTable:
    """
    Proper wide→long: creates a DataTable where
    each original row appears once per selected column.
    """
    name     = result_name or f"{source.name} (long)"
    n_rows   = source.num_rows()
    result   = DataTable(name)
    result.add_column(var_col)    # col index 0
    result.add_column(val_col)    # col index 1

    out_row = 0
    for col in source.columns:
        if col.name not in value_cols:
            continue
        for row_idx, v in enumerate(col.values):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            # Encode group name as its position index (numeric)
            result.set_value(out_row, 0, float(value_cols.index(col.name)))
            result.set_value(out_row, 1, float(v))
            out_row += 1

    return result


def _long_to_wide(
    source: DataTable,
    group_col: str,
    value_col: str,
    result_name: str | None = None,
) -> DataTable:
    """
    Pivot: each unique value in *group_col* becomes a column.
    Works best when group_col contains integers (group indices)
    or when the user has numeric group IDs.
    """
    df = source.to_dataframe()
    if group_col not in df.columns or value_col not in df.columns:
        raise ValueError("Columns not found.")

    pivoted = (df[[group_col, value_col]]
               .pivot_table(index=df.index,
                            columns=group_col,
                            values=value_col,
                            aggfunc="first"))
    pivoted.columns = [str(c) for c in pivoted.columns]
    pivoted = pivoted.reset_index(drop=True)

    name   = result_name or f"{source.name} (wide)"
    result = DataTable(name)
    for col in pivoted.columns:
        result.add_column(col)
    for r, row in pivoted.iterrows():
        for c_idx, col in enumerate(pivoted.columns):
            val = row[col]
            result.set_value(r, c_idx,
                             None if (isinstance(val, float) and np.isnan(val))
                             else float(val))
    return result


# ── Dialog ─────────────────────────────────────────────────────────────

class ReshapeDialog(QDialog):
    """Wide ↔ Long table reshaping."""

    def __init__(self, tables: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reshape Table")
        self.setMinimumWidth(500)
        self._tables = [t for t in tables if isinstance(t, DataTable)]
        self._result: DataTable | None = None

        layout = QVBoxLayout(self)
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_wide2long(), "Wide  →  Long")
        self._tabs.addTab(self._build_long2wide(), "Long  →  Wide")
        layout.addWidget(self._tabs)

        convert_btn = QPushButton("Convert")
        convert_btn.setDefault(True)
        convert_btn.clicked.connect(self._convert)
        layout.addWidget(convert_btn, alignment=Qt.AlignmentFlag.AlignRight)

        self._status = QLabel("")
        layout.addWidget(self._status)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    # ------------------------------------------------------------------
    def _build_wide2long(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel(
            "Select which columns to <b>stack</b> into a single 'Value' column.\n"
            "A numeric 'Group' column will encode each original column by index."
        ))
        form = QFormLayout()
        self._w2l_table = QComboBox()
        self._w2l_table.addItems([t.name for t in self._tables])
        self._w2l_table.currentIndexChanged.connect(self._refresh_w2l_cols)
        form.addRow("Source table:", self._w2l_table)
        layout.addLayout(form)

        layout.addWidget(QLabel("Columns to stack  (hold Ctrl / Shift to select multiple):"))
        self._w2l_cols = QListWidget()
        self._w2l_cols.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self._w2l_cols.setMaximumHeight(120)
        layout.addWidget(self._w2l_cols)

        form2 = QFormLayout()
        self._w2l_var  = QLineEdit("Group")
        self._w2l_val  = QLineEdit("Value")
        self._w2l_name = QLineEdit()
        form2.addRow("Group column name:", self._w2l_var)
        form2.addRow("Value column name:", self._w2l_val)
        form2.addRow("Result table name:", self._w2l_name)
        layout.addLayout(form2)
        layout.addStretch()

        self._refresh_w2l_cols()
        return w

    def _refresh_w2l_cols(self):
        self._w2l_cols.clear()
        idx = self._w2l_table.currentIndex()
        if 0 <= idx < len(self._tables):
            t = self._tables[idx]
            for col in t.columns:
                self._w2l_cols.addItem(col.name)
            self._w2l_name.setText(f"{t.name} (long)")
        # Select all by default
        self._w2l_cols.selectAll()

    def _build_long2wide(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self._l2w_table = QComboBox()
        self._l2w_table.addItems([t.name for t in self._tables])
        self._l2w_table.currentIndexChanged.connect(self._refresh_l2w_cols)
        form.addRow("Source table:", self._l2w_table)

        self._l2w_group = QComboBox()
        form.addRow("Group column:", self._l2w_group)
        self._l2w_value = QComboBox()
        form.addRow("Value column:", self._l2w_value)

        self._l2w_name = QLineEdit()
        form.addRow("Result table name:", self._l2w_name)
        form.addRow(QLabel(
            "The Group column is used to create separate columns in the output."
        ))

        self._refresh_l2w_cols()
        return w

    def _refresh_l2w_cols(self):
        idx = self._l2w_table.currentIndex()
        names = []
        if 0 <= idx < len(self._tables):
            t = self._tables[idx]
            names = [c.name for c in t.columns]
            self._l2w_name.setText(f"{t.name} (wide)")
        for cb in (self._l2w_group, self._l2w_value):
            cb.clear()
            cb.addItems(names)
        if len(names) >= 2:
            self._l2w_value.setCurrentIndex(1)

    # ------------------------------------------------------------------
    def _convert(self):
        try:
            if self._tabs.currentIndex() == 0:
                idx = self._w2l_table.currentIndex()
                if idx < 0 or idx >= len(self._tables):
                    raise ValueError("Select a table.")
                t = self._tables[idx]
                sel = [item.text() for item in self._w2l_cols.selectedItems()]
                if not sel:
                    raise ValueError("Select at least one column to stack.")
                self._result = _wide_to_long_clean(
                    t, sel,
                    var_col=self._w2l_var.text().strip() or "Group",
                    val_col=self._w2l_val.text().strip() or "Value",
                    result_name=self._w2l_name.text().strip() or None,
                )
            else:
                idx = self._l2w_table.currentIndex()
                if idx < 0 or idx >= len(self._tables):
                    raise ValueError("Select a table.")
                t = self._tables[idx]
                self._result = _long_to_wide(
                    t,
                    group_col=self._l2w_group.currentText(),
                    value_col=self._l2w_value.currentText(),
                    result_name=self._l2w_name.text().strip() or None,
                )
            n = self._result.num_rows()
            nc = len(self._result.columns)
            self._status.setText(
                f"Preview: {n} rows × {nc} columns — click OK to add to project."
            )
        except Exception as exc:
            self._status.setText(f"Error: {exc}")
            self._result = None

    def get_result(self) -> DataTable | None:
        return self._result
