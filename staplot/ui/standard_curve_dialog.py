"""Standard curve dialog — interpolate unknown X values from measured Y values."""

import numpy as np
from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QComboBox, QTextEdit,
    QPushButton, QDialogButtonBox, QLabel, QLineEdit,
)
from qtpy.QtGui import QFont


class StandardCurveDialog(QDialog):
    """
    Given a previously computed curve-fit result, interpolate unknown
    X values from user-supplied Y values by numerically inverting the model.
    """

    def __init__(self, analyses: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Standard Curve — Interpolate Unknowns")
        self.setMinimumWidth(460)

        # Only analyses that carry a fitted model with numeric params
        self._fits = [
            a for a in analyses
            if "r2" in a.details and "params" in a.details and "model" in a.details
        ]

        layout = QVBoxLayout(self)

        form = QFormLayout()

        self._fit_cb = QComboBox()
        self._fit_cb.addItems([a.name for a in self._fits])
        self._fit_cb.currentIndexChanged.connect(self._update_info)
        form.addRow("Fitted model:", self._fit_cb)
        layout.addLayout(form)

        self._info_lbl = QLabel()
        self._info_lbl.setStyleSheet("color: #444; font-size: 10px;")
        layout.addWidget(self._info_lbl)

        form2 = QFormLayout()
        self._y_edit = QLineEdit()
        self._y_edit.setPlaceholderText("e.g.  0.45, 0.78, 1.23")
        form2.addRow("Unknown Y values:", self._y_edit)
        layout.addLayout(form2)

        hint = QLabel(
            "Enter measured response (Y) values separated by commas.\n"
            "The corresponding X values (e.g. concentrations) will be interpolated."
        )
        hint.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(hint)

        calc_btn = QPushButton("Interpolate X values")
        calc_btn.setDefault(True)
        calc_btn.clicked.connect(self._interpolate)
        layout.addWidget(calc_btn)

        self._result_box = QTextEdit()
        self._result_box.setReadOnly(True)
        mono = QFont("Courier New", 9)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self._result_box.setFont(mono)
        self._result_box.setMinimumHeight(160)
        layout.addWidget(self._result_box)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        if self._fits:
            self._update_info()
        else:
            self._info_lbl.setText(
                "No curve fits available.  Run  Statistics → Curve Fitting  first."
            )

    # ------------------------------------------------------------------
    def _current_fit(self):
        idx = self._fit_cb.currentIndex()
        return self._fits[idx] if 0 <= idx < len(self._fits) else None

    def _update_info(self):
        fit = self._current_fit()
        if fit is None:
            self._info_lbl.setText("")
            return
        d = fit.details
        r2    = d.get("r2", float("nan"))
        model = d.get("model", "?")
        params = d.get("params", {})
        p_str = "  ".join(f"{k}={v:.4g}" for k, v in params.items())
        self._info_lbl.setText(f"Model: {model}   R² = {r2:.4f}\nParams: {p_str}")

    # ------------------------------------------------------------------
    def _interpolate(self):
        fit = self._current_fit()
        if fit is None:
            self._result_box.setPlainText("No fit selected.")
            return

        text = self._y_edit.text().strip()
        if not text:
            self._result_box.setPlainText("Enter Y values above.")
            return
        try:
            y_vals = [float(s.strip()) for s in text.split(",") if s.strip()]
        except ValueError:
            self._result_box.setPlainText("Invalid input — enter numbers separated by commas.")
            return

        d = fit.details
        model_name = d.get("model", "")
        params_dict = d.get("params", {})   # dict {name: value} from curve_fit.py

        try:
            from ..stats.curve_fit import MODELS
            if model_name not in MODELS:
                raise ValueError(f"Model '{model_name}' not registered.")

            func, param_names = MODELS[model_name]
            params = [params_dict[n] for n in param_names]

            from scipy.optimize import brentq

            lines = [f"{'Y value':>14s}  →  {'X (interpolated)':>16s}"]
            lines.append("─" * 36)
            for y_target in y_vals:
                try:
                    # Try to bracket the solution
                    x_lo, x_hi = 1e-12, 1e9
                    # Scan a few orders of magnitude to find a sign change
                    xs = np.logspace(-6, 6, 500)
                    ys = func(xs, *params) - y_target
                    sign_changes = np.where(np.diff(np.sign(ys)))[0]
                    if len(sign_changes) == 0:
                        lines.append(f"{y_target:>14.6g}  →  no solution found")
                        continue
                    i = sign_changes[0]
                    x_lo, x_hi = float(xs[i]), float(xs[i + 1])
                    x_val = brentq(lambda x: func(x, *params) - y_target,
                                   x_lo, x_hi, xtol=1e-10, maxiter=500)
                    lines.append(f"{y_target:>14.6g}  →  {x_val:>16.6g}")
                except Exception:
                    lines.append(f"{y_target:>14.6g}  →  no solution in search range")
            self._result_box.setPlainText("\n".join(lines))

        except Exception as exc:
            self._result_box.setPlainText(f"Error: {exc}")
