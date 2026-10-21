"""
'What test should I use?' wizard.
Walks the user through: data shape → normality check → recommendation → run.
"""

from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QStackedWidget,
    QComboBox, QRadioButton, QButtonGroup, QPushButton, QLabel,
    QTextEdit, QGroupBox, QWidget, QFrame,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont

from ..models.project import DataTable
from .. import stats as stat_lib
from .analysis_dialog import AnalysisDialog


class TestWizard(QDialog):
    """Step-by-step guide: data type → normality → recommended test → run."""

    def __init__(self, tables: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("What test should I use?")
        self.setMinimumSize(520, 460)
        self._tables = [t for t in tables if hasattr(t, "columns")]
        self._analysis_result = None

        outer = QVBoxLayout(self)

        # Step header
        self._header = QLabel()
        hdr_font = QFont()
        hdr_font.setBold(True)
        hdr_font.setPointSize(11)
        self._header.setFont(hdr_font)
        outer.addWidget(self._header)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        outer.addWidget(sep)

        # Pages
        self._stack = QStackedWidget()
        outer.addWidget(self._stack)

        self._stack.addWidget(self._page_data())       # 0
        self._stack.addWidget(self._page_normality())  # 1
        self._stack.addWidget(self._page_result())     # 2

        # Nav buttons
        nav = QHBoxLayout()
        self._back_btn = QPushButton("← Back")
        self._next_btn = QPushButton("Next →")
        self._run_btn  = QPushButton("Run Test")
        self._run_btn.setVisible(False)
        nav.addWidget(self._back_btn)
        nav.addStretch()
        nav.addWidget(self._run_btn)
        nav.addWidget(self._next_btn)
        outer.addLayout(nav)

        self._back_btn.clicked.connect(self._prev)
        self._next_btn.clicked.connect(self._next)
        self._run_btn.clicked.connect(self._run_recommended)

        self._go_to(0)

    # ------------------------------------------------------------------
    # Page builders
    # ------------------------------------------------------------------
    def _page_data(self) -> QWidget:
        w = QWidget()
        layout = QFormLayout(w)

        self._tbl_cb = QComboBox()
        self._tbl_cb.addItems([t.name for t in self._tables])
        layout.addRow("Table:", self._tbl_cb)

        grp_box = QGroupBox("Number of groups to compare")
        grp_lay = QVBoxLayout(grp_box)
        self._grp2 = QRadioButton("2 groups")
        self._grp2.setChecked(True)
        self._grpN = QRadioButton("3 or more groups")
        grp_lay.addWidget(self._grp2)
        grp_lay.addWidget(self._grpN)
        layout.addRow(grp_box)

        paired_box = QGroupBox("Are the observations paired / matched?")
        paired_lay = QVBoxLayout(paired_box)
        self._paired_yes = QRadioButton("Yes — same subjects measured twice (or matched pairs)")
        self._paired_no  = QRadioButton("No — independent groups")
        self._paired_no.setChecked(True)
        paired_lay.addWidget(self._paired_yes)
        paired_lay.addWidget(self._paired_no)
        layout.addRow(paired_box)

        return w

    def _page_normality(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("Running Shapiro-Wilk normality test on each group…"))
        self._norm_result = QTextEdit()
        self._norm_result.setReadOnly(True)
        mono = QFont("Courier New", 8)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self._norm_result.setFont(mono)
        layout.addWidget(self._norm_result)
        return w

    def _page_result(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        self._rec_label = QLabel()
        rec_font = QFont()
        rec_font.setPointSize(11)
        self._rec_label.setFont(rec_font)
        self._rec_label.setWordWrap(True)
        layout.addWidget(self._rec_label)

        self._reason_label = QLabel()
        self._reason_label.setWordWrap(True)
        self._reason_label.setStyleSheet("color: #555555;")
        layout.addWidget(self._reason_label)

        layout.addStretch()
        return w

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def _go_to(self, idx: int):
        self._stack.setCurrentIndex(idx)
        headers = [
            "Step 1 of 3 — Describe your data",
            "Step 2 of 3 — Normality check",
            "Step 3 of 3 — Recommended test",
        ]
        self._header.setText(headers[idx])
        self._back_btn.setEnabled(idx > 0)
        last = idx == 2
        self._next_btn.setVisible(not last)
        self._run_btn.setVisible(last)

    def _prev(self):
        self._go_to(max(0, self._stack.currentIndex() - 1))

    def _next(self):
        idx = self._stack.currentIndex()
        if idx == 0:
            self._run_normality()
            self._go_to(1)
        elif idx == 1:
            self._build_recommendation()
            self._go_to(2)

    # ------------------------------------------------------------------
    # Logic
    # ------------------------------------------------------------------
    def _current_table(self) -> DataTable | None:
        i = self._tbl_cb.currentIndex()
        return self._tables[i] if 0 <= i < len(self._tables) else None

    def _run_normality(self):
        t = self._current_table()
        if t is None:
            self._norm_result.setPlainText("No table selected.")
            return
        result = stat_lib.run_normality(t)
        self._norm_result.setPlainText(result.summary_text)
        self._norm_details = result.details

    def _all_normal(self) -> bool:
        if not hasattr(self, "_norm_details"):
            return True
        return all(
            v.get("passed", True)
            for v in self._norm_details.values()
            if isinstance(v, dict)
        )

    def _build_recommendation(self):
        two_groups = self._grp2.isChecked()
        paired     = self._paired_yes.isChecked()
        normal     = self._all_normal()

        if two_groups:
            if paired:
                if normal:
                    test   = "Paired t-test"
                    reason = "Two groups, paired data, both pass normality → parametric paired test."
                    self._recommended = ("ttest", True, "two-sided")
                else:
                    test   = "Wilcoxon signed-rank test"
                    reason = "Two groups, paired data, but normality failed → non-parametric paired test."
                    self._recommended = ("wilcoxon", True, "two-sided")
            else:
                if normal:
                    test   = "Unpaired t-test (Welch)"
                    reason = "Two independent groups, both pass normality → Welch's t-test (doesn't assume equal variance)."
                    self._recommended = ("ttest", False, "two-sided")
                else:
                    test   = "Mann-Whitney U test"
                    reason = "Two independent groups, normality failed → non-parametric alternative to t-test."
                    self._recommended = ("mann-whitney", False, "two-sided")
        else:
            if normal:
                test   = "One-way ANOVA + Tukey post-hoc"
                reason = "3+ groups, all pass normality → parametric omnibus test with pairwise comparisons."
                self._recommended = ("anova",)
            else:
                test   = "Kruskal-Wallis + Dunn post-hoc"
                reason = "3+ groups, normality failed → non-parametric omnibus test."
                self._recommended = ("kruskal",)

        self._rec_label.setText(f"✔  Recommended test:  {test}")
        self._reason_label.setText(reason)

    def _run_recommended(self):
        t = self._current_table()
        if t is None or not hasattr(self, "_recommended"):
            return
        rec = self._recommended
        groups = [c.name for c in t.columns]

        try:
            if rec[0] == "ttest":
                if len(groups) < 2:
                    raise ValueError("Need at least 2 columns.")
                r = stat_lib.run_ttest(t, groups[0], groups[1], paired=rec[1], alternative=rec[2])
            elif rec[0] == "wilcoxon":
                if len(groups) < 2:
                    raise ValueError("Need at least 2 columns.")
                r = stat_lib.run_wilcoxon(t, groups[0], groups[1], alternative=rec[2])
            elif rec[0] == "mann-whitney":
                if len(groups) < 2:
                    raise ValueError("Need at least 2 columns.")
                r = stat_lib.run_mann_whitney(t, groups[0], groups[1], alternative=rec[2])
            elif rec[0] == "anova":
                r = stat_lib.run_one_way_anova(t)
            else:
                r = stat_lib.run_kruskal_wallis(t)

            self._analysis_result = r
            self.accept()
        except Exception as exc:
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", str(exc))

    def get_result(self):
        return self._analysis_result
