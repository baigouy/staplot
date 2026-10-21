"""Power analysis / sample-size calculator dialog."""

from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QComboBox, QDoubleSpinBox,
    QSpinBox, QDialogButtonBox, QPushButton, QTextEdit, QLabel,
)
from qtpy.QtGui import QFont


class PowerDialog(QDialog):
    """Standalone power analysis dialog (no table needed)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Power Analysis")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        form = QFormLayout()

        self._test_cb = QComboBox()
        self._test_cb.addItems([
            "Two-sample t-test (unpaired)",
            "Paired t-test",
            "One-Way ANOVA",
        ])
        form.addRow("Test type:", self._test_cb)

        self._effect_sb = QDoubleSpinBox()
        self._effect_sb.setRange(0.01, 5.0)
        self._effect_sb.setSingleStep(0.05)
        self._effect_sb.setValue(0.5)
        self._effect_sb.setDecimals(3)
        form.addRow("Effect size (Cohen's d / f):", self._effect_sb)

        self._alpha_sb = QDoubleSpinBox()
        self._alpha_sb.setRange(0.001, 0.2)
        self._alpha_sb.setSingleStep(0.005)
        self._alpha_sb.setValue(0.05)
        self._alpha_sb.setDecimals(3)
        form.addRow("α (significance level):", self._alpha_sb)

        self._power_sb = QDoubleSpinBox()
        self._power_sb.setRange(0.5, 0.999)
        self._power_sb.setSingleStep(0.05)
        self._power_sb.setValue(0.80)
        self._power_sb.setDecimals(3)
        form.addRow("Target power (1−β):", self._power_sb)

        self._n_sb = QSpinBox()
        self._n_sb.setRange(0, 100_000)
        self._n_sb.setValue(0)
        self._n_sb.setSpecialValueText("Auto (solve for n)")
        form.addRow("N per group (0 = solve):", self._n_sb)

        self._k_sb = QSpinBox()
        self._k_sb.setRange(2, 50)
        self._k_sb.setValue(3)
        form.addRow("Groups k  (ANOVA only):", self._k_sb)

        layout.addLayout(form)

        hint = QLabel(
            "Cohen's d: small = 0.20 | medium = 0.50 | large = 0.80\n"
            "Cohen's f: small = 0.10 | medium = 0.25 | large = 0.40"
        )
        hint.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(hint)

        calc_btn = QPushButton("Calculate")
        calc_btn.setDefault(True)
        calc_btn.clicked.connect(self._calculate)
        layout.addWidget(calc_btn)

        self._result_box = QTextEdit()
        self._result_box.setReadOnly(True)
        mono = QFont("Courier New", 9)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self._result_box.setFont(mono)
        self._result_box.setMinimumHeight(130)
        layout.addWidget(self._result_box)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    # ------------------------------------------------------------------
    def _calculate(self):
        test  = self._test_cb.currentText()
        effect = self._effect_sb.value()
        alpha  = self._alpha_sb.value()
        power  = self._power_sb.value()
        n_val  = self._n_sb.value()
        k      = self._k_sb.value()
        n = None if n_val == 0 else n_val

        try:
            if "ANOVA" in test:
                from ..stats.power import run_power_anova
                result = run_power_anova(effect, alpha, power, k, n)
            else:
                from ..stats.power import run_power_ttest
                paired = "Paired" in test
                result = run_power_ttest(effect, alpha, power, n, paired)
            self._result_box.setPlainText(result.summary_text)
        except Exception as exc:
            self._result_box.setPlainText(f"Error: {exc}")
