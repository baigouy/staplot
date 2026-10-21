from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QCheckBox, QPushButton, QTextEdit, QLabel,
    QDialogButtonBox, QTabWidget, QWidget, QDoubleSpinBox, QSpinBox,
    QStackedWidget, QListWidget, QAbstractItemView,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont

from ..models.project import DataTable, GroupedDataTable, ContingencyTable, AnalysisResult
from .. import stats as stat_lib


class AnalysisDialog(QDialog):
    """Unified dialog for running all supported statistical tests."""

    _TABS = [
        "t-test / Wilcoxon",      # 0
        "ANOVA / Kruskal-Wallis", # 1
        "Two-Way ANOVA",          # 2
        "Correlation",            # 3
        "Normality",              # 4
        "Curve Fitting",          # 5
        "Descriptive",            # 6
        "Chi-square / Fisher",    # 7
        "RM ANOVA / Friedman",    # 8
        "Outliers (Grubbs)",      # 9
        "ROC Curve",              # 10
        "Multiple Regression",    # 11
    ]

    def __init__(self, tables: list, parent=None, initial_tab: int = 0):
        super().__init__(parent)
        self.setWindowTitle("Statistical Analysis")
        self.setMinimumSize(600, 600)
        self._tables = tables
        self._result: AnalysisResult | None = None

        layout = QVBoxLayout(self)

        # Table selector (for non-contingency tests)
        top = QFormLayout()
        self._table_cb = QComboBox()
        col_tables = [t for t in tables if hasattr(t, "columns")]
        self._table_cb.addItems([t.name for t in col_tables])
        self._col_tables = col_tables
        self._table_cb.currentIndexChanged.connect(self._refresh_groups)
        top.addRow("Table:", self._table_cb)
        layout.addLayout(top)

        # Test tabs
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_ttest_tab(),       self._TABS[0])
        self._tabs.addTab(self._build_anova_tab(),       self._TABS[1])
        self._tabs.addTab(self._build_two_way_tab(),     self._TABS[2])
        self._tabs.addTab(self._build_corr_tab(),        self._TABS[3])
        self._tabs.addTab(self._build_normality_tab(),   self._TABS[4])
        self._tabs.addTab(self._build_curvefit_tab(),    self._TABS[5])
        self._tabs.addTab(self._build_descriptive_tab(), self._TABS[6])
        self._tabs.addTab(self._build_chisq_tab(),       self._TABS[7])
        self._tabs.addTab(self._build_rm_tab(),          self._TABS[8])
        self._tabs.addTab(self._build_grubbs_tab(),      self._TABS[9])
        self._tabs.addTab(self._build_roc_tab(),         self._TABS[10])
        self._tabs.addTab(self._build_mlr_tab(),         self._TABS[11])
        self._tabs.setCurrentIndex(min(initial_tab, len(self._TABS) - 1))
        layout.addWidget(self._tabs)

        # Run button
        run_btn = QPushButton("Run Analysis")
        run_btn.setDefault(True)
        run_btn.clicked.connect(self._run)
        layout.addWidget(run_btn, alignment=Qt.AlignmentFlag.AlignRight)

        # Results
        self._result_box = QTextEdit()
        self._result_box.setReadOnly(True)
        mono = QFont("Courier New", 8)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self._result_box.setFont(mono)
        self._result_box.setMinimumHeight(140)
        layout.addWidget(self._result_box)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self._refresh_groups()

    # ------------------------------------------------------------------
    def _current_table(self):
        idx = self._table_cb.currentIndex()
        if idx < 0 or idx >= len(self._col_tables):
            return None
        return self._col_tables[idx]

    def _group_names(self) -> list[str]:
        t = self._current_table()
        return [c.name for c in t.columns] if t and hasattr(t, "columns") else []

    def _refresh_groups(self):
        names = self._group_names()
        for cb in [self._tt_ga, self._tt_gb, self._cf_x, self._cf_y,
                   self._corr_a, self._corr_b, self._roc_label, self._roc_score,
                   self._mlr_y]:
            cb.clear()
            cb.addItems(names)
        # Refresh MLR X list
        self._mlr_x.clear()
        self._mlr_x.addItems(names)
        if len(names) >= 2:
            self._tt_gb.setCurrentIndex(1)
            self._corr_b.setCurrentIndex(1)
            self._cf_y.setCurrentIndex(min(1, len(names) - 1))
            self._roc_score.setCurrentIndex(min(1, len(names) - 1))

    # ------------------------------------------------------------------
    # Tab builders — original 7
    # ------------------------------------------------------------------
    def _build_ttest_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self._tt_ga = QComboBox()
        self._tt_gb = QComboBox()
        self._tt_paired = QCheckBox("Paired")
        self._tt_alt = QComboBox()
        self._tt_alt.addItems(["two-sided", "greater", "less"])
        self._tt_nonpar = QCheckBox("Non-parametric (Mann-Whitney / Wilcoxon)")
        form.addRow("Group A:", self._tt_ga)
        form.addRow("Group B:", self._tt_gb)
        form.addRow("", self._tt_paired)
        form.addRow("Alternative:", self._tt_alt)
        form.addRow("", self._tt_nonpar)
        return w

    def _build_anova_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self._anova_nonpar = QCheckBox("Non-parametric (Kruskal-Wallis)")
        self._anova_posthoc = QComboBox()
        self._anova_posthoc.addItems(["tukey", "none"])
        form.addRow("", self._anova_nonpar)
        form.addRow("Post-hoc:", self._anova_posthoc)
        form.addRow(QLabel("(Uses all columns in the selected table)"))
        return w

    def _build_two_way_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel(
            "Requires a Grouped data table (Factor A = columns, Factor B = rows).\n"
            "Create one via  Data → Add Grouped Table."
        ))
        return w

    def _build_corr_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self._corr_a = QComboBox()
        self._corr_b = QComboBox()
        self._corr_method = QComboBox()
        self._corr_method.addItems(["pearson", "spearman", "kendall"])
        form.addRow("Column A:", self._corr_a)
        form.addRow("Column B:", self._corr_b)
        form.addRow("Method:", self._corr_method)
        return w

    def _build_normality_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel(
            "Runs Shapiro-Wilk, D'Agostino-Pearson, and KS tests\n"
            "on every column in the selected table.\n\n"
            "p < 0.05  → data deviate from normality → consider non-parametric tests."
        ))
        return w

    def _build_curvefit_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self._cf_x = QComboBox()
        self._cf_y = QComboBox()
        self._cf_model = QComboBox()
        self._cf_model.addItems(list(stat_lib.MODELS.keys()))
        form.addRow("X column:", self._cf_x)
        form.addRow("Y column:", self._cf_y)
        form.addRow("Model:", self._cf_model)
        return w

    def _build_descriptive_tab(self) -> QWidget:
        w = QWidget()
        QVBoxLayout(w).addWidget(
            QLabel("Computes N, Mean, SD, SEM, 95% CI, Median for all columns.")
        )
        return w

    # ------------------------------------------------------------------
    # Tab builders — new (v0.4-0.6)
    # ------------------------------------------------------------------
    def _build_chisq_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        self._chisq_table_cb = QComboBox()
        cont_tables = [t for t in self._tables if isinstance(t, ContingencyTable)]
        self._cont_tables = cont_tables
        self._chisq_table_cb.addItems([t.name for t in cont_tables])

        form = QFormLayout()
        form.addRow("Contingency table:", self._chisq_table_cb)
        layout.addLayout(form)

        self._chisq_fisher = QCheckBox("Fisher's exact test (2×2 only)")
        layout.addWidget(self._chisq_fisher)

        layout.addWidget(QLabel(
            "\nCreate a contingency table via  Data → Add Contingency Table.\n"
            "Enter integer counts, then run the test here."
        ))
        layout.addStretch()
        return w

    def _build_rm_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel(
            "Repeated-measures ANOVA / Friedman test.\n\n"
            "Table format: each column = one treatment/time-point;\n"
            "each row = one subject (same subject across all columns).\n"
        ))
        self._rm_nonpar = QCheckBox("Non-parametric (Friedman)")
        layout.addWidget(self._rm_nonpar)
        layout.addStretch()
        return w

    def _build_grubbs_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self._grubbs_alpha = QDoubleSpinBox()
        self._grubbs_alpha.setRange(0.001, 0.2)
        self._grubbs_alpha.setSingleStep(0.005)
        self._grubbs_alpha.setValue(0.05)
        self._grubbs_alpha.setDecimals(3)
        form.addRow("α (significance level):", self._grubbs_alpha)
        form.addRow(QLabel(
            "\nTests each column for a single extreme outlier using Grubbs' test.\n"
            "Runs on all columns of the selected table."
        ))
        return w

    def _build_roc_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self._roc_label = QComboBox()
        self._roc_score = QComboBox()
        form.addRow("Label column (0/1):", self._roc_label)
        form.addRow("Score column:", self._roc_score)
        form.addRow(QLabel(
            "\nLabel column must contain binary values (0 = negative, 1 = positive).\n"
            "Score column contains the continuous classifier output."
        ))
        return w

    def _build_mlr_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        form = QFormLayout()
        self._mlr_y = QComboBox()
        form.addRow("Dependent (Y):", self._mlr_y)
        layout.addLayout(form)
        layout.addWidget(QLabel(
            "Predictors (X) — hold Ctrl / Shift to select multiple:"
        ))
        self._mlr_x = QListWidget()
        self._mlr_x.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self._mlr_x.setMaximumHeight(120)
        layout.addWidget(self._mlr_x)
        layout.addWidget(QLabel(
            "OLS regression with intercept. Residual plot available via "
            "Graphs menu after running."
        ))
        layout.addStretch()
        return w

    # ------------------------------------------------------------------
    def _run(self):
        t = self._current_table()
        tab = self._tabs.currentIndex()
        try:
            # ── Tabs that don't need _current_table ──────────────────
            if tab == 7:   # Chi-square / Fisher
                idx = self._chisq_table_cb.currentIndex()
                if idx < 0 or idx >= len(self._cont_tables):
                    raise ValueError("Select a contingency table.")
                ct = self._cont_tables[idx]
                if self._chisq_fisher.isChecked():
                    result = stat_lib.run_fisher_exact(ct)
                else:
                    result = stat_lib.run_chi_square(ct)
                result.run_params = {}

            elif tab == 8:  # RM ANOVA / Friedman
                if t is None:
                    raise ValueError("Select a data table.")
                friedman = self._rm_nonpar.isChecked()
                result = (stat_lib.run_friedman(t) if friedman else stat_lib.run_rm_anova(t))
                result.run_params = {"non_parametric": friedman}

            elif tab == 9:  # Grubbs
                if t is None:
                    raise ValueError("Select a data table.")
                alpha = self._grubbs_alpha.value()
                result = stat_lib.run_grubbs(t, alpha=alpha)
                result.run_params = {"alpha": alpha}

            elif tab == 10:  # ROC
                if t is None:
                    raise ValueError("Select a data table.")
                lbl, score = self._roc_label.currentText(), self._roc_score.currentText()
                result = stat_lib.run_roc(t, lbl, score)
                result.run_params = {"label_col": lbl, "score_col": score}

            # ── Original tabs ─────────────────────────────────────────
            elif tab == 0:
                if t is None:
                    raise ValueError("Select a data table.")
                ga, gb = self._tt_ga.currentText(), self._tt_gb.currentText()
                alt = self._tt_alt.currentText()
                paired = self._tt_paired.isChecked()
                nonpar = self._tt_nonpar.isChecked()
                if nonpar:
                    result = (stat_lib.run_wilcoxon(t, ga, gb, alt) if paired
                              else stat_lib.run_mann_whitney(t, ga, gb, alt))
                else:
                    result = stat_lib.run_ttest(t, ga, gb, paired, alt)
                result.run_params = {
                    "group_a": ga, "group_b": gb,
                    "paired": paired, "alternative": alt, "non_parametric": nonpar,
                }

            elif tab == 1:
                if t is None:
                    raise ValueError("Select a data table.")
                nonpar = self._anova_nonpar.isChecked()
                post_hoc = self._anova_posthoc.currentText()
                result = (stat_lib.run_kruskal_wallis(t) if nonpar
                          else stat_lib.run_one_way_anova(t, post_hoc))
                result.run_params = {"non_parametric": nonpar, "post_hoc": post_hoc}

            elif tab == 2:
                if not isinstance(t, GroupedDataTable):
                    raise ValueError("Two-Way ANOVA requires a Grouped table.")
                result = stat_lib.run_two_way_anova(t)
                result.run_params = {}

            elif tab == 3:
                if t is None:
                    raise ValueError("Select a data table.")
                ca, cb = self._corr_a.currentText(), self._corr_b.currentText()
                meth = self._corr_method.currentText()
                result = stat_lib.run_correlation(t, ca, cb, meth)
                result.run_params = {"col_a": ca, "col_b": cb, "method": meth}

            elif tab == 4:
                if t is None:
                    raise ValueError("Select a data table.")
                result = stat_lib.run_normality(t)
                result.run_params = {}

            elif tab == 5:
                if t is None:
                    raise ValueError("Select a data table.")
                x_col = self._cf_x.currentText()
                y_col = self._cf_y.currentText()
                model = self._cf_model.currentText()
                result = stat_lib.run_curve_fit(t, x_col, y_col, model)
                result.run_params = {"x_col": x_col, "y_col": y_col, "model": model}

            elif tab == 11:  # Multiple Regression
                if t is None:
                    raise ValueError("Select a data table.")
                y_col  = self._mlr_y.currentText()
                x_cols = [item.text() for item in self._mlr_x.selectedItems()]
                if not x_cols:
                    raise ValueError("Select at least one predictor (X column).")
                result = stat_lib.run_multiple_regression(t, y_col, x_cols)
                result.run_params = {"y_col": y_col, "x_cols": x_cols}

            else:  # Descriptive
                if t is None:
                    raise ValueError("Select a data table.")
                from ..stats.tests import run_descriptive
                result = run_descriptive(t)
                result.run_params = {}

            self._result = result
            self._result_box.setPlainText(result.summary_text)

        except Exception as exc:
            self._result_box.setPlainText(f"Error: {exc}")

    def get_result(self) -> AnalysisResult | None:
        return self._result
