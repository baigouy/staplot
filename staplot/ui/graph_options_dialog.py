from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QComboBox, QCheckBox,
    QLineEdit, QDialogButtonBox, QLabel, QPushButton, QHBoxLayout,
    QScrollArea, QWidget, QGroupBox, QTabWidget,
)
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QColorDialog
from qtpy.QtCore import Qt

from ..models.project import DataTable, GroupedDataTable
from ..plots.styles import PALETTE, JOURNAL_PRESETS

def _try_float(text: str, default: float) -> float:
    try:
        return float(text.strip())
    except (ValueError, AttributeError):
        return default


def _try_int(text: str, default: int) -> int:
    try:
        return int(text.strip())
    except (ValueError, AttributeError):
        return default


PLOT_TYPES = [
    "bar", "before_after", "scatter", "box", "violin",
    "histogram", "survival", "qq", "heatmap", "roc", "bland_altman",
    "grouped_bar", "volcano",
]

# Plot types that use per-group colour pickers
_COLORED_TYPES = {"bar", "before_after", "scatter", "box", "violin",
                  "histogram", "qq", "grouped_bar"}


class _ColorButton(QPushButton):
    def __init__(self, color: str, parent=None):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(28, 22)
        self._apply()
        self.clicked.connect(self._pick)

    def _apply(self):
        self.setStyleSheet(f"background-color: {self._color}; border: 1px solid #666;")

    def _pick(self):
        c = QColorDialog.getColor(QColor(self._color), self)
        if c.isValid():
            self._color = c.name()
            self._apply()

    def color(self) -> str:
        return self._color


class GraphOptionsDialog(QDialog):
    def __init__(self, tables: list, initial_type: str = "bar", parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Graph")
        self.setMinimumWidth(430)
        # Include all table types (grouped bar needs GroupedDataTable)
        self._tables = [t for t in tables
                        if hasattr(t, "columns") or isinstance(t, GroupedDataTable)]
        self._color_btns: list[_ColorButton] = []

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # ── Tab 1: Graph ──────────────────────────────────────────────
        graph_tab = QWidget()
        form1 = QFormLayout(graph_tab)

        # Journal preset
        self._preset_cb = QComboBox()
        self._preset_cb.addItems(list(JOURNAL_PRESETS.keys()))
        form1.addRow("Journal preset:", self._preset_cb)

        self._table_cb = QComboBox()
        self._table_cb.addItems([t.name for t in self._tables])
        self._table_cb.currentIndexChanged.connect(self._rebuild_colors)
        self._table_cb.currentIndexChanged.connect(self._rebuild_col_selectors)
        form1.addRow("Table:", self._table_cb)

        self._type_cb = QComboBox()
        self._type_cb.addItems(PLOT_TYPES)
        if initial_type in PLOT_TYPES:
            self._type_cb.setCurrentText(initial_type)
        form1.addRow("Graph type:", self._type_cb)

        self._title_edit  = QLineEdit()
        form1.addRow("Title:", self._title_edit)
        self._xlabel_edit = QLineEdit()
        form1.addRow("X label:", self._xlabel_edit)
        self._ylabel_edit = QLineEdit()
        form1.addRow("Y label:", self._ylabel_edit)

        self._error_cb = QComboBox()
        self._error_cb.addItems(["sd", "sem", "ci95", "none"])     # SD first: journals prefer it
        form1.addRow("Error bars:", self._error_cb)

        self._points_chk = QCheckBox("Show individual data points")
        self._points_chk.setChecked(True)
        form1.addRow("", self._points_chk)

        self._regression_chk = QCheckBox("Show regression line (scatter)")
        self._regression_chk.setChecked(True)
        form1.addRow("", self._regression_chk)

        self._sig_chk = QCheckBox("Auto-add significance brackets from analyses")
        self._sig_chk.setChecked(True)
        form1.addRow("", self._sig_chk)

        tabs.addTab(graph_tab, "Graph")

        # ── Tab 2: Colors ─────────────────────────────────────────────
        self._color_tab = QWidget()
        self._color_layout = QVBoxLayout(self._color_tab)
        self._color_layout.addWidget(QLabel("Click a swatch to change group colour:"))
        self._color_scroll = QScrollArea()
        self._color_scroll.setWidgetResizable(True)
        self._color_inner = QWidget()
        self._color_inner_layout = QVBoxLayout(self._color_inner)
        self._color_scroll.setWidget(self._color_inner)
        self._color_layout.addWidget(self._color_scroll)
        tabs.addTab(self._color_tab, "Colors")

        # ── Tab 3: Histogram ──────────────────────────────────────────
        hist_tab = QWidget()
        form3 = QFormLayout(hist_tab)
        self._bins_edit = QLineEdit("auto")
        form3.addRow("Bins:", self._bins_edit)
        self._density_chk = QCheckBox("Density (normalise area to 1)")
        form3.addRow("", self._density_chk)
        self._kde_chk = QCheckBox("KDE overlay")
        self._kde_chk.setChecked(True)
        form3.addRow("", self._kde_chk)
        self._cumulative_chk = QCheckBox("Cumulative")
        form3.addRow("", self._cumulative_chk)
        tabs.addTab(hist_tab, "Histogram")

        # ── Tab 4: Bland-Altman ───────────────────────────────────────
        ba_tab = QWidget()
        form4 = QFormLayout(ba_tab)
        self._ba_col_a = QComboBox()
        self._ba_col_b = QComboBox()
        self._ba_pct = QCheckBox("Plot % difference instead of absolute")
        form4.addRow("Method A column:", self._ba_col_a)
        form4.addRow("Method B column:", self._ba_col_b)
        form4.addRow("", self._ba_pct)
        form4.addRow(QLabel(
            "Select 'bland_altman' as graph type in the Graph tab."
        ))
        tabs.addTab(ba_tab, "Bland-Altman")

        # ── Tab 5: ROC ────────────────────────────────────────────────
        roc_tab = QWidget()
        form5 = QFormLayout(roc_tab)
        self._roc_label_cb = QComboBox()
        self._roc_score_cb = QComboBox()
        form5.addRow("Label column (0/1):", self._roc_label_cb)
        form5.addRow("Score column:", self._roc_score_cb)
        form5.addRow(QLabel(
            "Label column: 0 = negative, 1 = positive.\n"
            "Select 'roc' as graph type in the Graph tab."
        ))
        tabs.addTab(roc_tab, "ROC")

        # ── Tab 6: Heatmap ────────────────────────────────────────────
        hm_tab = QWidget()
        form6 = QFormLayout(hm_tab)
        self._hm_mode = QComboBox()
        self._hm_mode.addItems(["values", "correlation"])
        form6.addRow("Mode:", self._hm_mode)
        self._hm_cmap = QComboBox()
        self._hm_cmap.addItems(["RdYlBu_r", "RdBu_r", "viridis", "plasma", "coolwarm", "Blues"])
        form6.addRow("Colour map:", self._hm_cmap)
        self._hm_annotate = QCheckBox("Annotate cells with values")
        self._hm_annotate.setChecked(True)
        form6.addRow("", self._hm_annotate)
        form6.addRow(QLabel("Select 'heatmap' as graph type in the Graph tab."))
        tabs.addTab(hm_tab, "Heatmap")

        # ── Tab 7: Volcano ───────────────────────────────────────────
        vol_tab = QWidget()
        form7 = QFormLayout(vol_tab)
        self._vol_fc   = QComboBox()
        self._vol_p    = QComboBox()
        self._vol_lbl  = QComboBox()   # optional label column
        self._vol_fc_thr = QLineEdit("1.0")
        self._vol_p_thr  = QLineEdit("0.05")
        self._vol_top    = QLineEdit("10")
        form7.addRow("log2 FC column:", self._vol_fc)
        form7.addRow("p-value column:", self._vol_p)
        form7.addRow("Label column (optional):", self._vol_lbl)
        form7.addRow("|log2FC| threshold:", self._vol_fc_thr)
        form7.addRow("p threshold:", self._vol_p_thr)
        form7.addRow("Label top N hits:", self._vol_top)
        form7.addRow(QLabel("Select 'volcano' as graph type in the Graph tab."))
        tabs.addTab(vol_tab, "Volcano")

        # ── Tab 8: Grouped Bar ────────────────────────────────────────
        grp_tab = QWidget()
        form8 = QFormLayout(grp_tab)
        form8.addRow(QLabel(
            "For a Grouped Bar chart, select a Grouped table in the\n"
            "Graph tab and choose 'grouped_bar' as the graph type.\n\n"
            "Error bars use the Error Bars setting from the Graph tab.\n"
            "Colours are set in the Colors tab."
        ))
        tabs.addTab(grp_tab, "Grouped Bar")

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self._rebuild_colors()
        self._rebuild_col_selectors()

    # ------------------------------------------------------------------
    def _current_table(self):
        idx = self._table_cb.currentIndex()
        return self._tables[idx] if 0 <= idx < len(self._tables) else None

    def _rebuild_colors(self):
        while self._color_inner_layout.count():
            item = self._color_inner_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._color_btns.clear()

        t = self._current_table()
        if t is None:
            return
        # GroupedDataTable uses col_groups; DataTable uses columns
        if isinstance(t, GroupedDataTable):
            entries = t.col_groups
        else:
            entries = [c.name for c in t.columns]

        for i, name in enumerate(entries):
            row_w = QHBoxLayout()
            lbl = QLabel(name)
            lbl.setFixedWidth(130)
            btn = _ColorButton(PALETTE[i % len(PALETTE)])
            self._color_btns.append(btn)
            row_w.addWidget(lbl)
            row_w.addWidget(btn)
            row_w.addStretch()
            container = QWidget()
            container.setLayout(row_w)
            self._color_inner_layout.addWidget(container)
        self._color_inner_layout.addStretch()

    def _rebuild_col_selectors(self):
        t = self._current_table()
        if isinstance(t, GroupedDataTable):
            names = []   # grouped tables don't expose columns for BA/ROC/volcano
        else:
            names = [c.name for c in t.columns] if t else []
        for cb in [self._ba_col_a, self._ba_col_b,
                   self._roc_label_cb, self._roc_score_cb,
                   self._vol_fc, self._vol_p]:
            cb.clear()
            cb.addItems(names)
        # Volcano label column: include a "(none)" option
        self._vol_lbl.clear()
        self._vol_lbl.addItem("(none)")
        self._vol_lbl.addItems(names)
        if len(names) >= 2:
            self._ba_col_b.setCurrentIndex(1)
            self._roc_score_cb.setCurrentIndex(1)
            self._vol_p.setCurrentIndex(1)

    # ------------------------------------------------------------------
    def get_options(self) -> dict:
        bins_txt = self._bins_edit.text().strip()
        try:
            bins = int(bins_txt)
        except ValueError:
            bins = "auto"

        return {
            "table_name":      self._table_cb.currentText(),
            "plot_type":       self._type_cb.currentText(),
            "title":           self._title_edit.text(),
            "xlabel":          self._xlabel_edit.text(),
            "ylabel":          self._ylabel_edit.text(),
            "error_type":      self._error_cb.currentText(),
            "show_points":     self._points_chk.isChecked(),
            "show_regression": self._regression_chk.isChecked(),
            "sig_brackets":    self._sig_chk.isChecked(),
            "colors":          [btn.color() for btn in self._color_btns],
            "journal_preset":  self._preset_cb.currentText(),
            # Histogram
            "hist_bins":        bins,
            "hist_density":     self._density_chk.isChecked(),
            "hist_kde":         self._kde_chk.isChecked(),
            "hist_cumulative":  self._cumulative_chk.isChecked(),
            # Bland-Altman
            "ba_col_a":        self._ba_col_a.currentText(),
            "ba_col_b":        self._ba_col_b.currentText(),
            "ba_percentage":   self._ba_pct.isChecked(),
            # ROC
            "roc_label":       self._roc_label_cb.currentText(),
            "roc_score":       self._roc_score_cb.currentText(),
            # Heatmap
            "heatmap_mode":      self._hm_mode.currentText(),
            "heatmap_cmap":      self._hm_cmap.currentText(),
            "heatmap_annotate":  self._hm_annotate.isChecked(),
            # Volcano
            "volcano_fc":        self._vol_fc.currentText(),
            "volcano_p":         self._vol_p.currentText(),
            "volcano_label":     (self._vol_lbl.currentText()
                                  if self._vol_lbl.currentText() != "(none)" else None),
            "volcano_fc_thresh": _try_float(self._vol_fc_thr.text(), 1.0),
            "volcano_p_thresh":  _try_float(self._vol_p_thr.text(), 0.05),
            "volcano_top":       _try_int(self._vol_top.text(), 10),
        }
