"""Data transform dialog — create a new table by applying a function to all columns."""

import numpy as np

from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QComboBox, QDoubleSpinBox,
    QLineEdit, QDialogButtonBox, QLabel, QCheckBox, QGroupBox,
)
from qtpy.QtCore import Qt

from ..models.project import DataTable, Project


TRANSFORMS = {
    "Log₁₀(Y)":       lambda y, _k, _c: np.log10(np.where(y > 0, y, np.nan)),
    "Ln(Y)":           lambda y, _k, _c: np.log(np.where(y > 0, y, np.nan)),
    "√Y":              lambda y, _k, _c: np.sqrt(np.where(y >= 0, y, np.nan)),
    "Y²":              lambda y, _k, _c: y ** 2,
    "1 / Y":           lambda y, _k, _c: np.where(y != 0, 1.0 / y, np.nan),
    "Y × k":           lambda y, k, _c: y * k,
    "Y + k":           lambda y, k, _c: y + k,
    "100 × Y / max(Y)":lambda y, _k, _c: 100.0 * y / np.nanmax(y) if np.nanmax(y) != 0 else y,
    "Z-score":         lambda y, _k, _c: (y - np.nanmean(y)) / np.nanstd(y, ddof=1)
                                          if np.nanstd(y, ddof=1) > 0 else y * 0,
    "% of control":    None,   # special-cased below
}


class TransformDialog(QDialog):
    def __init__(self, tables: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Transform Data")
        self.setMinimumWidth(360)
        self._tables = tables
        self._new_table: DataTable | None = None

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._table_cb = QComboBox()
        self._table_cb.addItems([t.name for t in tables if hasattr(t, "columns")])
        self._table_cb.currentIndexChanged.connect(self._refresh_controls)
        form.addRow("Source table:", self._table_cb)

        self._transform_cb = QComboBox()
        self._transform_cb.addItems(list(TRANSFORMS.keys()))
        self._transform_cb.currentTextChanged.connect(self._refresh_controls)
        form.addRow("Transform:", self._transform_cb)

        self._k_spin = QDoubleSpinBox()
        self._k_spin.setRange(-1e9, 1e9)
        self._k_spin.setValue(1.0)
        self._k_spin.setDecimals(4)
        self._k_label = QLabel("k:")
        form.addRow(self._k_label, self._k_spin)

        self._ctrl_cb = QComboBox()
        self._ctrl_label = QLabel("Control column:")
        form.addRow(self._ctrl_label, self._ctrl_cb)

        self._out_name = QLineEdit()
        form.addRow("Output table name:", self._out_name)

        layout.addLayout(form)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._apply)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self._refresh_controls()

    # ------------------------------------------------------------------
    def _src_table(self) -> DataTable | None:
        idx = self._table_cb.currentIndex()
        col_tables = [t for t in self._tables if hasattr(t, "columns")]
        return col_tables[idx] if 0 <= idx < len(col_tables) else None

    def _refresh_controls(self):
        transform = self._transform_cb.currentText()
        needs_k = transform in ("Y × k", "Y + k")
        needs_ctrl = transform == "% of control"
        self._k_label.setVisible(needs_k)
        self._k_spin.setVisible(needs_k)
        self._ctrl_label.setVisible(needs_ctrl)
        self._ctrl_cb.setVisible(needs_ctrl)

        t = self._src_table()
        if t:
            self._ctrl_cb.clear()
            self._ctrl_cb.addItems([c.name for c in t.columns])
            self._out_name.setText(f"{t.name} — {transform}")

    # ------------------------------------------------------------------
    def _apply(self):
        t = self._src_table()
        if t is None:
            return

        transform = self._transform_cb.currentText()
        k = self._k_spin.value()
        ctrl_name = self._ctrl_cb.currentText()
        out_name = self._out_name.text().strip() or f"{t.name} (transformed)"

        new_table = DataTable(out_name)
        func = TRANSFORMS.get(transform)

        ctrl_mean = 1.0
        if transform == "% of control":
            ctrl_col = next((c for c in t.columns if c.name == ctrl_name), None)
            if ctrl_col:
                ctrl_mean = float(np.nanmean(ctrl_col.valid_values())) or 1.0

        for col in t.columns:
            arr = col.as_array()
            if transform == "% of control":
                transformed = 100.0 * arr / ctrl_mean
            elif func is not None:
                try:
                    transformed = func(arr, k, None)
                except Exception:
                    transformed = arr.copy()
            else:
                transformed = arr.copy()

            new_col = new_table.add_column(col.name)
            new_col.values = [
                None if np.isnan(v) else float(v) for v in transformed
            ]

        self._new_table = new_table
        self.accept()

    def get_new_table(self) -> DataTable | None:
        return self._new_table
