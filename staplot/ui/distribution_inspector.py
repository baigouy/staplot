"""Distribution Inspector — unified normality, shape, fitting, and test-advisor dialog."""

from __future__ import annotations

import io

import matplotlib
matplotlib.use("QtAgg")

from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QComboBox, QLabel, QPushButton, QGroupBox, QFormLayout,
    QSplitter, QTextEdit, QApplication,
)
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QFont, QImage

from ..models.project import DataTable
from ..plots import styles
from .plot_canvas import StyledCanvas


# ── colour helpers ──────────────────────────────────────────────────

_GREEN  = QColor("#d4edda")
_RED    = QColor("#f8d7da")
_YELLOW = QColor("#fff3cd")
_GRAY   = QColor("#f0f0f0")


def _verdict_color(is_normal: bool) -> QColor:
    return _GREEN if is_normal else _RED


# ── DistributionInspector ───────────────────────────────────────────

class DistributionInspector(QDialog):
    """
    All-in-one dialog for:
      • Normality tests + shape stats per column      (Summary tab)
      • Histogram + KDE + Q-Q panel per column        (Visual tab)
      • Distribution fitting ranked by AIC             (in Visual tab)
      • Automatic test recommendation + one-click run  (Summary tab)
    """

    # Emitted when the user clicks "Run recommended test" for a column
    run_test_requested = Signal(int)   # analysis-dialog tab index

    def __init__(self, tables: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Distribution Inspector")
        self.setMinimumSize(860, 620)
        self._tables   = [t for t in tables if isinstance(t, DataTable) and t.columns]
        self._result   = None      # AnalysisResult from last run
        self._dist_fig = None      # current detail figure

        layout = QVBoxLayout(self)

        # ── Table selector ────────────────────────────────────────────
        top = QHBoxLayout()
        top.addWidget(QLabel("Table:"))
        self._table_cb = QComboBox()
        self._table_cb.addItems([t.name for t in self._tables])
        self._table_cb.currentIndexChanged.connect(self._clear_results)
        top.addWidget(self._table_cb)
        top.addStretch()
        run_btn = QPushButton("Run Distribution Check")
        run_btn.setDefault(True)
        run_btn.clicked.connect(self._run)
        top.addWidget(run_btn)
        layout.addLayout(top)

        # ── Tabs ──────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        # Tab 1 — Summary table
        self._tab_summary = QWidget()
        self._tabs.addTab(self._tab_summary, "Summary & Advisor")
        self._build_summary_tab()

        # Tab 2 — Per-column visual
        self._tab_visual = QWidget()
        self._tabs.addTab(self._tab_visual, "Visual Details")
        self._build_visual_tab()

        # Tab 3 — Raw text report
        self._tab_text = QWidget()
        self._tabs.addTab(self._tab_text, "Text Report")
        self._build_text_tab()

        # ── Footer buttons ─────────────────────────────────────────────
        foot = QHBoxLayout()
        copy_btn = QPushButton("Copy Report")
        copy_btn.clicked.connect(self._copy_report)
        foot.addWidget(copy_btn)
        copy_img_btn = QPushButton("Copy Plot")
        copy_img_btn.clicked.connect(self._copy_plot)
        foot.addWidget(copy_img_btn)
        foot.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        foot.addWidget(close_btn)
        layout.addLayout(foot)

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------
    def _build_summary_tab(self):
        layout = QVBoxLayout(self._tab_summary)

        # Instructions label
        info = QLabel(
            "Run a distribution check to see normality results and test recommendations per column."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Summary table
        self._summary_tbl = QTableWidget(0, 7)
        self._summary_tbl.setHorizontalHeaderLabels([
            "Column", "n", "Shapiro-Wilk p", "Skewness",
            "Kurtosis", "Verdict", "Recommended test (2 groups)",
        ])
        self._summary_tbl.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._summary_tbl.horizontalHeader().setSectionResizeMode(
            6, QHeaderView.ResizeMode.Stretch
        )
        self._summary_tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._summary_tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._summary_tbl.verticalHeader().setVisible(False)
        layout.addWidget(self._summary_tbl)

        # Advisor panel (shows detail for selected row)
        grp = QGroupBox("Test Advisor")
        adv_layout = QVBoxLayout(grp)
        self._advisor_label = QLabel("Select a column in the table above.")
        self._advisor_label.setWordWrap(True)
        bold = QFont(); bold.setBold(True)
        self._advisor_label.setFont(bold)
        adv_layout.addWidget(self._advisor_label)

        btn_row = QHBoxLayout()
        self._run_rec_btn = QPushButton("Run recommended test…")
        self._run_rec_btn.setEnabled(False)
        self._run_rec_btn.clicked.connect(self._open_recommended_test)
        btn_row.addWidget(self._run_rec_btn)
        btn_row.addStretch()
        adv_layout.addLayout(btn_row)
        layout.addWidget(grp)

        self._summary_tbl.itemSelectionChanged.connect(self._on_summary_select)

    def _build_visual_tab(self):
        layout = QVBoxLayout(self._tab_visual)

        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Column:"))
        self._col_cb = QComboBox()
        self._col_cb.currentTextChanged.connect(self._update_visual)
        sel_row.addWidget(self._col_cb)
        sel_row.addStretch()
        layout.addLayout(sel_row)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # Matplotlib canvas (histogram + QQ)
        self._detail_canvas_widget = QWidget()
        self._detail_canvas_layout = QVBoxLayout(self._detail_canvas_widget)
        self._detail_canvas_layout.setContentsMargins(0, 0, 0, 0)
        self._placeholder_label = QLabel(
            "Run Distribution Check to see plots.", alignment=Qt.AlignmentFlag.AlignCenter
        )
        self._detail_canvas_layout.addWidget(self._placeholder_label)
        self._mpl_canvas = None
        splitter.addWidget(self._detail_canvas_widget)

        # Fitting table
        fit_grp = QGroupBox("Distribution fitting (ranked by AIC — lower is better)")
        fit_v   = QVBoxLayout(fit_grp)
        self._fit_tbl = QTableWidget(0, 5)
        self._fit_tbl.setHorizontalHeaderLabels(
            ["Distribution", "AIC", "BIC", "KS p-value", "Fit OK?"]
        )
        self._fit_tbl.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._fit_tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._fit_tbl.setMaximumHeight(160)
        fit_v.addWidget(self._fit_tbl)
        splitter.addWidget(fit_grp)
        splitter.setSizes([400, 160])

        layout.addWidget(splitter)

    def _build_text_tab(self):
        layout = QVBoxLayout(self._tab_text)
        self._text_box = QTextEdit()
        self._text_box.setReadOnly(True)
        mono = QFont("Courier New", 8)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self._text_box.setFont(mono)
        layout.addWidget(self._text_box)

    # ------------------------------------------------------------------
    # Run analysis
    # ------------------------------------------------------------------
    def _current_table(self) -> DataTable | None:
        idx = self._table_cb.currentIndex()
        return self._tables[idx] if 0 <= idx < len(self._tables) else None

    def _clear_results(self):
        self._result = None
        self._summary_tbl.setRowCount(0)
        self._fit_tbl.setRowCount(0)
        self._text_box.clear()
        self._advisor_label.setText("Select a column in the table above.")
        self._run_rec_btn.setEnabled(False)
        self._col_cb.clear()

    def _run(self):
        t = self._current_table()
        if t is None:
            return
        from ..stats.distribution import run_distribution_check
        from ..plots.distribution_plot import plot_distribution_grid

        self._result = run_distribution_check(t)
        self._populate_summary()
        self._populate_col_selector()
        self._text_box.setPlainText(self._result.summary_text)
        self._tabs.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Summary tab population
    # ------------------------------------------------------------------
    def _populate_summary(self):
        if self._result is None:
            return
        details = self._result.details
        self._summary_tbl.setRowCount(0)

        for col_name, info in details.items():
            row = self._summary_tbl.rowCount()
            self._summary_tbl.insertRow(row)

            n          = info.get("n", 0)
            sw_p       = info.get("shapiro_p", float("nan"))
            skew       = info.get("skewness",  float("nan"))
            kurt       = info.get("kurtosis",  float("nan"))
            is_normal  = info.get("is_normal", True)
            rec        = info.get("rec_2groups", "—")
            error      = info.get("error")

            items = [
                col_name,
                str(n),
                f"{sw_p:.4f}" if sw_p == sw_p else "—",
                f"{skew:+.3f}" if skew == skew else "—",
                f"{kurt:+.3f}" if kurt == kurt else "—",
                "Normal" if is_normal else "Non-normal",
                "—" if error else rec,
            ]
            for c_idx, text in enumerate(items):
                itm = QTableWidgetItem(text)
                itm.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                itm.setFlags(itm.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._summary_tbl.setItem(row, c_idx, itm)

            # Colour rows
            bg = _verdict_color(is_normal) if not error else _GRAY
            for c_idx in range(self._summary_tbl.columnCount()):
                self._summary_tbl.item(row, c_idx).setBackground(bg)

    # ------------------------------------------------------------------
    # Visual tab helpers
    # ------------------------------------------------------------------
    def _populate_col_selector(self):
        self._col_cb.blockSignals(True)
        self._col_cb.clear()
        if self._result:
            self._col_cb.addItems(list(self._result.details.keys()))
        self._col_cb.blockSignals(False)
        if self._col_cb.count():
            self._update_visual(self._col_cb.currentText())

    def _update_visual(self, col_name: str):
        if not self._result or not col_name:
            return
        t = self._current_table()
        if t is None:
            return

        info = self._result.details.get(col_name, {})
        if info.get("error"):
            return

        # ── Histogram + QQ plot ───────────────────────────────────────
        from ..stats.distribution import fit_distributions
        from ..plots.distribution_plot import plot_distribution

        # Reconstruct fits list with actual scipy params
        arr  = next(c.valid_values() for c in t.columns if c.name == col_name)
        fits = fit_distributions(arr)

        fig = plot_distribution(t, col_name, fits=fits)
        self._dist_fig = fig
        self._set_canvas(fig)

        # ── Fitting table ─────────────────────────────────────────────
        self._fit_tbl.setRowCount(0)
        for fit in fits[:5]:
            r = self._fit_tbl.rowCount()
            self._fit_tbl.insertRow(r)
            ok_str = "Yes" if fit["ks_p"] >= 0.05 else "No"
            for c_idx, txt in enumerate([
                fit["distribution"],
                f"{fit['aic']:.1f}",
                f"{fit['bic']:.1f}",
                f"{fit['ks_p']:.4f}",
                ok_str,
            ]):
                itm = QTableWidgetItem(txt)
                itm.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._fit_tbl.setItem(r, c_idx, itm)
                if c_idx == 4:
                    itm.setBackground(_GREEN if ok_str == "Yes" else _RED)
            if r == 0:  # highlight best row
                for c_idx in range(self._fit_tbl.columnCount()):
                    self._fit_tbl.item(r, c_idx).setFont(
                        QFont("", -1, QFont.Weight.Bold)
                    )

    def _set_canvas(self, fig):
        """Replace the current matplotlib canvas with *fig*."""
        # Remove old canvas
        if self._mpl_canvas is not None:
            self._detail_canvas_layout.removeWidget(self._mpl_canvas)
            self._mpl_canvas.setParent(None)
            self._mpl_canvas = None
        # Remove placeholder
        if self._placeholder_label is not None:
            self._detail_canvas_layout.removeWidget(self._placeholder_label)
            self._placeholder_label.setParent(None)
            self._placeholder_label = None

        canvas = StyledCanvas(fig)
        canvas.setMinimumHeight(260)
        self._detail_canvas_layout.addWidget(canvas)
        self._mpl_canvas = canvas
        canvas.draw()

    # ------------------------------------------------------------------
    # Advisor panel
    # ------------------------------------------------------------------
    def _on_summary_select(self):
        rows = self._summary_tbl.selectedItems()
        if not rows:
            return
        row   = self._summary_tbl.row(rows[0])
        col_n = self._summary_tbl.item(row, 0).text()
        if self._result is None:
            return
        info  = self._result.details.get(col_n, {})
        if info.get("error"):
            self._advisor_label.setText(f"{col_n}: insufficient data.")
            self._run_rec_btn.setEnabled(False)
            return

        is_normal = info.get("is_normal", True)
        rec2      = info.get("rec_2groups", "—")
        rec3      = info.get("rec_3groups", "—")
        rec_corr  = info.get("rec_correlation", "—")
        best_fit  = info.get("best_fit", "—")
        sw_p      = info.get("shapiro_p", float("nan"))
        verdict   = "normally distributed" if is_normal else "NOT normally distributed"

        msg = (
            f"{col_n}: {verdict} (Shapiro-Wilk p = {sw_p:.4f})\n\n"
            f"  2-group comparison : {rec2}\n"
            f"  3+ groups          : {rec3}\n"
            f"  Best-fit distribution: {best_fit}"
        )
        self._advisor_label.setText(msg)
        self._run_rec_btn.setEnabled(True)
        # Store the recommended analysis-dialog tab index
        self._run_rec_btn._tab_idx = 0 if is_normal else 0  # always t-test/MW tab
        self._run_rec_btn._nonpar  = not is_normal

        # Also update the visual tab to show this column
        self._col_cb.setCurrentText(col_n)
        if self._tabs.currentIndex() != 0:
            self._update_visual(col_n)

    def _open_recommended_test(self):
        btn    = self._run_rec_btn
        nonpar = getattr(btn, "_nonpar", False)
        # Emit signal with tab 0 (t-test tab); MainWindow sets non-parametric checkbox
        self.run_test_requested.emit(0)
        # Pass the non-parametric preference via a custom attribute the parent can read
        self._nonpar_recommendation = nonpar
        self.accept()

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    def _copy_report(self):
        QApplication.clipboard().setText(self._text_box.toPlainText())

    @styles.styled
    def _copy_plot(self):
        if self._dist_fig is None:
            return
        buf = io.BytesIO()
        self._dist_fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        img = QImage()
        img.loadFromData(buf.read())
        QApplication.clipboard().setImage(img)

    # ------------------------------------------------------------------
    # Public getters (for MainWindow integration)
    # ------------------------------------------------------------------
    def get_result(self):
        return self._result

    def nonpar_recommended(self) -> bool:
        return getattr(self, "_nonpar_recommendation", False)
