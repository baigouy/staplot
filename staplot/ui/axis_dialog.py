"""Axis settings dialog — log scale, custom range, labels."""

from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QComboBox, QDoubleSpinBox,
    QLineEdit, QDialogButtonBox, QGroupBox, QCheckBox, QHBoxLayout,
    QLabel,
)
from qtpy.QtCore import Qt


class _RangeRow(QHBoxLayout):
    """A pair of optional min/max spinboxes."""

    def __init__(self, label: str, lo=None, hi=None):
        super().__init__()
        self._auto_lo = QCheckBox("auto")
        self._auto_hi = QCheckBox("auto")
        self._lo = QDoubleSpinBox()
        self._hi = QDoubleSpinBox()
        for sb in (self._lo, self._hi):
            sb.setRange(-1e9, 1e9)
            sb.setDecimals(4)
            sb.setFixedWidth(90)
        self._auto_lo.setChecked(lo is None)
        self._auto_hi.setChecked(hi is None)
        if lo is not None:
            self._lo.setValue(lo)
        if hi is not None:
            self._hi.setValue(hi)
        self._auto_lo.toggled.connect(lambda v: self._lo.setDisabled(v))
        self._auto_hi.toggled.connect(lambda v: self._hi.setDisabled(v))
        self._lo.setDisabled(lo is None)
        self._hi.setDisabled(hi is None)
        self.addWidget(QLabel("min")); self.addWidget(self._auto_lo); self.addWidget(self._lo)
        self.addSpacing(10)
        self.addWidget(QLabel("max")); self.addWidget(self._auto_hi); self.addWidget(self._hi)

    def get(self) -> tuple:
        lo = None if self._auto_lo.isChecked() else self._lo.value()
        hi = None if self._auto_hi.isChecked() else self._hi.value()
        return lo, hi


class AxisDialog(QDialog):
    """Configure axis scales and ranges for a graph."""

    def __init__(self, current: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Axis Settings")
        self.setMinimumWidth(420)
        opts = current or {}

        layout = QVBoxLayout(self)
        form = QFormLayout()

        # X axis
        x_grp = QGroupBox("X axis")
        x_layout = QFormLayout(x_grp)
        self._x_scale = QComboBox(); self._x_scale.addItems(["linear", "log"])
        self._x_scale.setCurrentText(opts.get("x_scale", "linear"))
        x_layout.addRow("Scale:", self._x_scale)
        self._x_range = _RangeRow("X", opts.get("x_min"), opts.get("x_max"))
        x_layout.addRow("Range:", self._x_range)
        self._x_label = QLineEdit(opts.get("x_label", ""))
        x_layout.addRow("Label:", self._x_label)
        layout.addWidget(x_grp)

        # Y axis
        y_grp = QGroupBox("Y axis")
        y_layout = QFormLayout(y_grp)
        self._y_scale = QComboBox(); self._y_scale.addItems(["linear", "log"])
        self._y_scale.setCurrentText(opts.get("y_scale", "linear"))
        y_layout.addRow("Scale:", self._y_scale)
        self._y_range = _RangeRow("Y", opts.get("y_min"), opts.get("y_max"))
        y_layout.addRow("Range:", self._y_range)
        self._y_label = QLineEdit(opts.get("y_label", ""))
        y_layout.addRow("Label:", self._y_label)
        # Y-start at zero
        self._y_zero = QCheckBox("Start Y axis at zero")
        self._y_zero.setChecked(opts.get("y_zero", False))
        y_layout.addRow("", self._y_zero)
        layout.addWidget(y_grp)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_options(self) -> dict:
        x_min, x_max = self._x_range.get()
        y_min, y_max = self._y_range.get()
        return {
            "x_scale": self._x_scale.currentText(),
            "y_scale": self._y_scale.currentText(),
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "x_label": self._x_label.text(),
            "y_label": self._y_label.text(),
            "y_zero": self._y_zero.isChecked(),
        }
