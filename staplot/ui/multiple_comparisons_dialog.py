"""Multiple-comparisons correction dialog."""

from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QDoubleSpinBox, QPushButton, QTextEdit,
    QTableWidget, QTableWidgetItem, QDialogButtonBox,
    QLabel, QHeaderView, QAbstractItemView, QMessageBox,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont


_METHODS = [
    ("fdr_bh",    "Benjamini-Hochberg FDR  (recommended)"),
    ("fdr_by",    "Benjamini-Yekutieli FDR"),
    ("bonferroni","Bonferroni"),
    ("holm",      "Holm (Holm-Bonferroni)"),
    ("holm-sidak", "Holm-Sidak"),
    ("sidak",     "Sidak"),
]


class MultipleComparisonsDialog(QDialog):
    """
    Enter (or auto-import) a list of p-values, choose a correction method,
    and see adjusted p-values.
    """

    def __init__(self, analyses: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Multiple Comparisons Correction")
        self.setMinimumSize(580, 540)
        self._analyses = analyses
        self._result = None

        layout = QVBoxLayout(self)

        # ── Method / alpha ────────────────────────────────────────────
        form = QFormLayout()
        self._method_cb = QComboBox()
        for key, label in _METHODS:
            self._method_cb.addItem(label, key)
        form.addRow("Correction method:", self._method_cb)

        self._alpha_sb = QDoubleSpinBox()
        self._alpha_sb.setRange(0.001, 0.2)
        self._alpha_sb.setSingleStep(0.005)
        self._alpha_sb.setValue(0.05)
        self._alpha_sb.setDecimals(3)
        form.addRow("Family-wise alpha:", self._alpha_sb)
        layout.addLayout(form)

        # ── Table of comparisons ──────────────────────────────────────
        layout.addWidget(QLabel(
            "Enter comparison names and raw p-values, or click "
            "<b>Import from analyses</b> to load pairwise results automatically."
        ))

        self._tbl = QTableWidget(0, 2)
        self._tbl.setHorizontalHeaderLabels(["Comparison name", "Raw p-value"])
        self._tbl.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self._tbl.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents)
        self._tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(self._tbl)

        # Row control buttons
        row_btns = QHBoxLayout()
        add_btn = QPushButton("+ Row")
        add_btn.clicked.connect(self._add_row)
        del_btn = QPushButton("− Row")
        del_btn.clicked.connect(self._del_row)
        imp_btn = QPushButton("Import from analyses")
        imp_btn.clicked.connect(self._import_from_analyses)
        clr_btn = QPushButton("Clear all")
        clr_btn.clicked.connect(lambda: self._tbl.setRowCount(0))
        for b in (add_btn, del_btn, imp_btn, clr_btn):
            row_btns.addWidget(b)
        row_btns.addStretch()
        layout.addLayout(row_btns)

        # ── Run button ────────────────────────────────────────────────
        run_btn = QPushButton("Apply correction")
        run_btn.setDefault(True)
        run_btn.clicked.connect(self._run)
        layout.addWidget(run_btn, alignment=Qt.AlignmentFlag.AlignRight)

        # ── Result box ────────────────────────────────────────────────
        self._result_box = QTextEdit()
        self._result_box.setReadOnly(True)
        mono = QFont("Courier New", 8)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self._result_box.setFont(mono)
        self._result_box.setMinimumHeight(140)
        layout.addWidget(self._result_box)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        # Pre-populate from analyses
        self._import_from_analyses(silent=True)

    # ------------------------------------------------------------------
    def _add_row(self):
        r = self._tbl.rowCount()
        self._tbl.insertRow(r)
        self._tbl.setItem(r, 0, QTableWidgetItem(f"Comparison {r + 1}"))
        self._tbl.setItem(r, 1, QTableWidgetItem("1.000"))

    def _del_row(self):
        rows = sorted({i.row() for i in self._tbl.selectedItems()}, reverse=True)
        for r in rows:
            self._tbl.removeRow(r)

    def _import_from_analyses(self, silent: bool = False):
        from ..stats.multiple_comparisons import extract_pairwise_p
        names, pvals = extract_pairwise_p(self._analyses)
        if not names:
            if not silent:
                QMessageBox.information(
                    self, "Nothing to import",
                    "No pairwise comparisons found in existing analyses.\n"
                    "Run t-tests or post-hoc ANOVA comparisons first."
                )
            return
        self._tbl.setRowCount(0)
        for name, pv in zip(names, pvals):
            r = self._tbl.rowCount()
            self._tbl.insertRow(r)
            self._tbl.setItem(r, 0, QTableWidgetItem(name))
            self._tbl.setItem(r, 1, QTableWidgetItem(f"{pv:.6g}"))

    def _collect(self) -> tuple[list[str], list[float]]:
        names, pvals = [], []
        for r in range(self._tbl.rowCount()):
            n_item = self._tbl.item(r, 0)
            p_item = self._tbl.item(r, 1)
            name = n_item.text().strip() if n_item else f"Row {r + 1}"
            try:
                pv = float(p_item.text().strip()) if p_item else 1.0
            except ValueError:
                pv = 1.0
            names.append(name)
            pvals.append(pv)
        return names, pvals

    def _run(self):
        names, pvals = self._collect()
        if not pvals:
            self._result_box.setPlainText("Add at least one p-value.")
            return
        method = self._method_cb.currentData()
        alpha  = self._alpha_sb.value()
        try:
            from ..stats.multiple_comparisons import run_multiple_comparisons
            self._result = run_multiple_comparisons(pvals, names, method, alpha)
            self._result_box.setPlainText(self._result.summary_text)
        except Exception as exc:
            self._result_box.setPlainText(f"Error: {exc}")

    def get_result(self):
        return self._result
