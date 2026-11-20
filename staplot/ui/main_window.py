import os

from qtpy.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QSplitter, QStackedWidget,
    QAction, QFileDialog, QMessageBox, QInputDialog, QLabel, QMenu,
)
from qtpy.QtCore import Qt, QSettings, QTimer
from qtpy.QtGui import QKeySequence
# QUndoStack lives in QtWidgets (PyQt5) or QtGui (PyQt6) — try both
try:
    from qtpy.QtWidgets import QUndoStack
except ImportError:
    from qtpy.QtGui import QUndoStack  # type: ignore[no-redef]

from ..models.project import (
    Project, GraphConfig, GroupedDataTable, ContingencyTable, DataTable,
)
from .navigator import NavigatorPanel
from .data_table_widget import DataTableWidget
from .grouped_table_widget import GroupedTableWidget
from .contingency_table_widget import ContingencyTableWidget
from .plot_canvas import PlotCanvas, apply_axis_settings
from .results_widget import ResultsWidget
from .analysis_dialog import AnalysisDialog
from .graph_options_dialog import GraphOptionsDialog
from .. import plots as plot_lib
from ..plots import styles as plot_styles


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Staplot")
        self.resize(1100, 700)

        self.project = Project()
        self._undo_stack = QUndoStack(self)
        self._active_gc: GraphConfig | None = None
        self._dirty = False

        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._new_default_table()
        self._start_autosave()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(self.splitter)

        self.navigator = NavigatorPanel(self)
        self.navigator.item_activated.connect(self._on_nav_item)
        self.splitter.addWidget(self.navigator)

        self.stack = QStackedWidget()
        self.splitter.addWidget(self.stack)
        self.splitter.setSizes([200, 900])
        self.splitter.setStretchFactor(1, 1)

        self.table_widget = DataTableWidget(self)
        self.table_widget._undo_stack = self._undo_stack
        self.table_widget.data_changed.connect(self._on_data_changed)

        self.grouped_widget = GroupedTableWidget(self)
        self.grouped_widget.data_changed.connect(self._on_data_changed)

        self.contingency_widget = ContingencyTableWidget(self)
        self.contingency_widget.data_changed.connect(self._on_data_changed)

        self.plot_canvas = PlotCanvas(self)
        self.plot_canvas.axes_changed.connect(self._on_axes_changed)
        self.plot_canvas.point_excluded.connect(self._on_point_excluded)

        self.results_widget = ResultsWidget(self)

        self.stack.addWidget(self.table_widget)         # 0
        self.stack.addWidget(self.plot_canvas)           # 1
        self.stack.addWidget(self.results_widget)        # 2
        self.stack.addWidget(self.grouped_widget)        # 3
        self.stack.addWidget(self.contingency_widget)    # 4

        self._status_label = QLabel("Ready")
        self.statusBar().addPermanentWidget(self._status_label)

    # ------------------------------------------------------------------
    def _setup_menu(self):
        mb = self.menuBar()

        # ── File ──────────────────────────────────────────────────────
        file_m = mb.addMenu("&File")
        self._add_action(file_m, "&New Project",       self.new_project,     "Ctrl+N")
        self._add_action(file_m, "&Open…",             self.open_project,    "Ctrl+O")
        self._add_action(file_m, "&Save",              self.save_project,    "Ctrl+S")
        self._add_action(file_m, "Save &As…",          self.save_project_as)
        file_m.addSeparator()

        self._recent_menu = file_m.addMenu("Recent Files")
        self._update_recent_menu()

        file_m.addSeparator()
        self._add_action(file_m, "Import CSV…",        self.import_csv)
        self._add_action(file_m, "Import Excel…",      self.import_excel)
        self._add_action(file_m, "Export Graph…",      self.export_graph)
        file_m.addSeparator()
        self._add_action(file_m, "&Quit",              self.close,           "Ctrl+Q")

        # ── Edit ──────────────────────────────────────────────────────
        edit_m = mb.addMenu("&Edit")
        undo_act = self._undo_stack.createUndoAction(self, "&Undo")
        undo_act.setShortcut(QKeySequence("Ctrl+Z"))
        edit_m.addAction(undo_act)
        redo_act = self._undo_stack.createRedoAction(self, "&Redo")
        redo_act.setShortcut(QKeySequence("Ctrl+Y"))
        edit_m.addAction(redo_act)
        edit_m.addSeparator()
        self._add_action(edit_m, "Copy Graph to Clipboard",
                         lambda: self.plot_canvas.copy_to_clipboard(), "Ctrl+Shift+C")

        # ── Data ──────────────────────────────────────────────────────
        data_m = mb.addMenu("&Data")
        self._add_action(data_m, "Add Table",               self.add_table)
        self._add_action(data_m, "Add Grouped Table…",      self.add_grouped_table)
        self._add_action(data_m, "Add Contingency Table…",  self.add_contingency_table)
        self._add_action(data_m, "Rename Table…",           self.rename_table)
        data_m.addSeparator()
        self._add_action(data_m, "Add Column",        self.table_widget.add_column)
        self._add_action(data_m, "Remove Column",     self.table_widget.remove_selected_column)
        data_m.addSeparator()
        self._add_action(data_m, "Transform Data…",         self.transform_data)
        self._add_action(data_m, "Row Statistics…",         self.compute_row_stats)
        self._add_action(data_m, "Paste from Clipboard",
                         self.table_widget.paste_from_clipboard, "Ctrl+Shift+V")

        # ── Statistics ────────────────────────────────────────────────
        stat_m = mb.addMenu("&Statistics")
        self._add_action(stat_m, "What test should I use?…",  self.open_test_wizard)
        stat_m.addSeparator()
        self._add_action(stat_m, "Descriptive Statistics",    self._run_descriptive)
        self._add_action(stat_m, "Distribution Inspector…",   self.open_distribution_inspector)
        self._add_action(stat_m, "Normality Tests",           lambda: self._open_analysis(4))
        self._add_action(stat_m, "Outlier Test (Grubbs)…",   lambda: self._open_analysis(9))
        stat_m.addSeparator()
        self._add_action(stat_m, "t-test…",                   lambda: self._open_analysis(0))
        self._add_action(stat_m, "One-Way ANOVA…",            lambda: self._open_analysis(1))
        self._add_action(stat_m, "Two-Way ANOVA…",            lambda: self._open_analysis(2))
        self._add_action(stat_m, "RM ANOVA / Friedman…",      lambda: self._open_analysis(8))
        self._add_action(stat_m, "Correlation…",              lambda: self._open_analysis(3))
        self._add_action(stat_m, "Chi-square / Fisher…",      lambda: self._open_analysis(7))
        stat_m.addSeparator()
        self._add_action(stat_m, "Curve Fitting…",            lambda: self._open_analysis(5))
        self._add_action(stat_m, "Multiple Regression…",      lambda: self._open_analysis(11))
        self._add_action(stat_m, "ROC Curve…",                lambda: self._open_analysis(10))
        stat_m.addSeparator()
        self._add_action(stat_m, "Multiple Comparisons…",     self.open_multiple_comparisons)
        self._add_action(stat_m, "Power Analysis…",           self.open_power_dialog)

        # ── Graphs ────────────────────────────────────────────────────
        graph_m = mb.addMenu("&Graphs")
        for label, ptype in [
            ("Bar Chart",       "bar"),
            ("Before/After",    "before_after"),
            ("Scatter Plot",    "scatter"),
            ("Box Plot",        "box"),
            ("Violin Plot",     "violin"),
            ("Histogram",       "histogram"),
            ("Survival Curve",  "survival"),
        ]:
            self._add_action(graph_m, label, lambda pt=ptype: self.new_graph(pt))
        graph_m.addSeparator()
        for label, ptype in [
            ("Q-Q Plot",        "qq"),
            ("Heatmap",         "heatmap"),
            ("ROC Curve Plot",  "roc"),
            ("Bland-Altman",    "bland_altman"),
        ]:
            self._add_action(graph_m, label, lambda pt=ptype: self.new_graph(pt))
        for label, ptype in [
            ("Grouped Bar",     "grouped_bar"),
            ("Volcano Plot",    "volcano"),
        ]:
            self._add_action(graph_m, label, lambda pt=ptype: self.new_graph(pt))
        graph_m.addSeparator()
        self._add_action(graph_m, "Multi-Panel Figure…",      self.open_figure_layout)
        self._add_action(graph_m, "Graph Gallery…",            self.open_graph_gallery)

        # ── Tools ─────────────────────────────────────────────────────
        tools_m = mb.addMenu("&Tools")
        self._add_action(tools_m, "Standard Curve…",      self.open_standard_curve)
        self._add_action(tools_m, "Generate Report…",     self.open_report)
        tools_m.addSeparator()
        self._add_action(tools_m, "Computed Column…",     self.open_computed_column)
        self._add_action(tools_m, "Reshape Table…",       self.open_reshape)
        self._add_action(tools_m, "Export Table as CSV…", self.export_table_csv)

        # ── Help ──────────────────────────────────────────────────────
        help_m = mb.addMenu("&Help")
        self._add_action(help_m, "About", self._about)

    # ------------------------------------------------------------------
    def _setup_toolbar(self):
        from qtpy.QtCore import QSize
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QSize(18, 18))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self._add_action(tb, "New Table", self.add_table, icon_name="fa5s.table")
        self._add_action(tb, "Wizard",    self.open_test_wizard, icon_name="fa5s.magic")
        self._add_action(tb, "Edit a plot", self.open_plot_editor, icon_name="fa5s.paint-brush")
        tb.addSeparator()
        for label, pt, ic in [
            ("Bar",     "bar",          "fa5s.chart-bar"),
            ("B/A",     "before_after", "fa5s.exchange-alt"),
            ("Scatter", "scatter",      "mdi6.chart-scatter-plot"),
            ("Box",     "box",          "mdi6.chart-box-outline"),
            ("Violin",  "violin",       "mdi6.chart-bell-curve"),
            ("Hist",    "histogram",    "mdi6.chart-histogram"),
            ("Q-Q",     "qq",           "mdi6.chart-line-variant"),
            ("Heatmap", "heatmap",      "mdi6.grid"),
            ("Grp Bar", "grouped_bar",  "mdi6.chart-bar-stacked"),
            ("Volcano", "volcano",      "mdi6.chart-scatter-plot-hexbin"),
        ]:
            self._add_action(tb, label, lambda p=pt: self.new_graph(p), icon_name=ic)

    def open_plot_editor(self, path: str | None = None):
        """A table (CSV, Excel, SQLite) beside its plot; click any part of the plot to
        change it (plot_editor.PlotEditor, as PyFigures opens a data plot)."""
        from .plot_editor import PlotEditor
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Table to plot", "",
                                                  "Tables (*.csv *.tsv *.txt *.xlsx *.xls *.db *.sqlite);;All files (*)")
        if not path:
            return None
        ed = PlotEditor(path, "bar", parent=self)
        ed.exec()
        return ed

    def _add_action(self, parent, label, slot, shortcut=None, icon_name="") -> QAction:
        act = QAction(label, self)
        if icon_name:
            from .plot_editor import icon
            act.setIcon(icon(icon_name))
        act.triggered.connect(slot)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        parent.addAction(act)
        return act

    # ------------------------------------------------------------------
    # Auto-save
    # ------------------------------------------------------------------
    def _start_autosave(self):
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(5 * 60 * 1000)   # 5 minutes
        self._autosave_timer.timeout.connect(self._auto_save)
        self._autosave_timer.start()

    def _mark_modified(self, message: str = "Modified"):
        """The project has unsaved changes (dedicated flag: status-bar text is
        overwritten by informational messages, so it can't carry this state)."""
        self._dirty = True
        self._status_label.setText(message)

    def _auto_save(self):
        if self.project.filepath and self._dirty:
            try:
                self.project.save(self.project.filepath)
            except Exception as exc:
                # Project.save is atomic: the previous file is intact
                self._status_label.setText(f"Auto-save FAILED: {exc}")
                return
            self._dirty = False
            self._status_label.setText("Auto-saved")

    # ------------------------------------------------------------------
    # Recent files
    # ------------------------------------------------------------------
    def _get_recent_files(self) -> list[str]:
        settings = QSettings("Staplot", "Staplot")
        val = settings.value("recent_files", [])
        return val if isinstance(val, list) else []

    def _add_to_recent(self, path: str):
        recent = self._get_recent_files()
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        recent = recent[:8]
        settings = QSettings("Staplot", "Staplot")
        settings.setValue("recent_files", recent)
        self._update_recent_menu()

    def _update_recent_menu(self):
        self._recent_menu.clear()
        recent = self._get_recent_files()
        for path in recent:
            name = os.path.basename(path)
            act = QAction(name, self)
            act.setData(path)
            act.setStatusTip(path)
            act.triggered.connect(lambda checked=False, p=path: self._open_recent(p))
            self._recent_menu.addAction(act)
        if not recent:
            placeholder = QAction("(none)", self)
            placeholder.setEnabled(False)
            self._recent_menu.addAction(placeholder)

    def _open_recent(self, path: str):
        if not os.path.exists(path):
            QMessageBox.warning(self, "File not found",
                                f"'{path}' no longer exists.")
            return
        if not self._confirm_discard():
            return
        try:
            self.project = Project.load(path)
            self._dirty = False
            self._active_gc = None
            self.navigator.populate(self.project)
            self._load_first_table()
            self.setWindowTitle(f"Staplot — {self.project.name}")
            self._add_to_recent(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    def _new_default_table(self):
        t = self.project.add_table("Data 1")
        for name in ["Group A", "Group B", "Group C"]:
            t.add_column(name)
        self.navigator.add_table(t.name)
        self.table_widget.load_table(t)
        self.stack.setCurrentIndex(0)

    def add_table(self):
        name, ok = QInputDialog.getText(self, "New Table", "Table name:")
        if not ok or not name.strip():
            return
        t = self.project.add_table(name.strip())
        t.add_column("Group A"); t.add_column("Group B")
        self.navigator.add_table(t.name)
        self.table_widget.load_table(t)
        self.stack.setCurrentIndex(0)

    def add_grouped_table(self):
        name, ok = QInputDialog.getText(self, "New Grouped Table", "Table name:")
        if not ok or not name.strip():
            return
        reps, ok2 = QInputDialog.getInt(self, "Replicates", "Replicates per cell:", 3, 1, 20)
        if not ok2:
            return
        t = self.project.add_grouped_table(name.strip(), n_reps=reps)
        for cg in ["Group A", "Group B"]:
            t.add_col_group(cg)
        for rl in ["Row 1", "Row 2", "Row 3"]:
            t.add_row_label(rl)
        self.navigator.add_table(t.name)
        self.grouped_widget.load_table(t)
        self.stack.setCurrentIndex(3)

    def add_contingency_table(self):
        name, ok = QInputDialog.getText(self, "New Contingency Table", "Table name:")
        if not ok or not name.strip():
            return
        t = self.project.add_contingency_table(name.strip())
        for r in ["Row 1", "Row 2"]:
            t.add_row(r)
        for c in ["Col 1", "Col 2"]:
            t.add_col(c)
        self.navigator.add_table(t.name)
        self.contingency_widget.load_table(t)
        self.stack.setCurrentIndex(4)

    def rename_table(self):
        t = self._active_table()
        if t is None:
            return
        new_name, ok = QInputDialog.getText(self, "Rename Table", "New name:", text=t.name)
        if ok and new_name.strip():
            old, new = t.name, new_name.strip()
            if new == old:
                return
            if self.project.get_table(new) is not None:
                QMessageBox.warning(self, "Rename Table",
                                    f"A table named '{new}' already exists.")
                return
            t.name = new
            # graphs and analyses refer to their table by name
            for gc in self.project.graphs:
                if gc.source_table == old:
                    gc.source_table = new
            for a in self.project.analyses:
                if a.source_table == old:
                    a.source_table = new
            self.navigator.rename_last_table(old, t.name)
            self._mark_modified()

    def _active_table(self):
        idx = self.stack.currentIndex()
        if idx == 0:
            return self.table_widget._table
        if idx == 3:
            return self.grouped_widget._table
        if idx == 4:
            return self.contingency_widget._table
        return self.project.tables[0] if self.project.tables else None

    # ------------------------------------------------------------------
    # Row statistics
    # ------------------------------------------------------------------
    def compute_row_stats(self):
        t = self._active_table()
        if t is None or not isinstance(t, DataTable) or not t.columns:
            QMessageBox.warning(self, "No table",
                                "Open a standard data table first.")
            return
        import numpy as np
        n_rows = t.num_rows()
        new_t = DataTable(f"{t.name} (Row Stats)")
        for col_name in ["Mean", "SD", "SEM", "N"]:
            new_t.add_column(col_name)

        df = t.to_dataframe()
        for row_idx in range(n_rows):
            vals = df.iloc[row_idx].dropna().values.astype(float)
            if len(vals) == 0:
                continue
            mean = float(np.mean(vals))
            sd   = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            sem  = sd / float(np.sqrt(len(vals))) if len(vals) > 1 else float("nan")
            n    = float(len(vals))
            for col_idx, v in enumerate([mean, sd, sem, n]):
                new_t.set_value(row_idx, col_idx, v)

        new_t.name = self.project._table_name(new_t.name)
        self.project.tables.append(new_t)
        self.navigator.add_table(new_t.name)
        self.table_widget.load_table(new_t)
        self.stack.setCurrentIndex(0)
        self._mark_modified(f"Created: {new_t.name}")

    # ------------------------------------------------------------------
    # Live graph update
    # ------------------------------------------------------------------
    def _on_data_changed(self):
        self._mark_modified()
        # Live graph refresh
        if self.stack.currentIndex() == 1 and self._active_gc is not None:
            self._render_graph(self._active_gc, live_refresh=True)
        # Live re-analysis for the currently-edited table
        t = self._active_table()
        if t is not None:
            self._rerun_analyses_for_table(t.name)

    # ------------------------------------------------------------------
    # Interactive point exclusion
    # ------------------------------------------------------------------
    def _on_point_excluded(self, col_name: str, row_idx: int, exclude_all: bool):
        """Toggle exclusion of a clicked data point and refresh everything."""
        if self._active_gc is None:
            return
        t = self.project.get_table(self._active_gc.source_table)
        if t is None or not isinstance(t, DataTable):
            return
        if exclude_all:
            for col in t.columns:
                col.toggle_excluded(row_idx)
        else:
            for col in t.columns:
                if col.name == col_name:
                    col.toggle_excluded(row_idx)
                    break
        # Sync table widget (only if it is displaying this table)
        if self.table_widget._table is t:
            self.table_widget._refresh()
        self._on_data_changed()

    # ------------------------------------------------------------------
    # Live re-analysis dispatcher
    # ------------------------------------------------------------------
    def _rerun_analyses_for_table(self, table_name: str):
        """Re-run all analyses whose source_table matches *table_name*."""
        from .. import stats as stat_lib

        t = self.project.get_table(table_name)
        if t is None:
            return

        refreshed_names = []
        for result in self.project.analyses:
            if result.source_table != table_name or not result.run_params:
                continue
            new_r = self._dispatch_analysis(result, t, stat_lib)
            if new_r is not None:
                result.summary_text = new_r.summary_text
                result.details      = new_r.details
                refreshed_names.append(result.name)

        if refreshed_names and self.stack.currentIndex() == 2:
            # Refresh results widget if it is showing one of the updated analyses
            shown = getattr(self.results_widget, "_current_result", None)
            if shown is not None and shown.name in refreshed_names:
                self.results_widget.show_result(shown)

    @staticmethod
    def _dispatch_analysis(result, table, stat_lib):
        """Re-run a single analysis from its stored method + run_params."""
        m = result.method
        p = result.run_params
        try:
            if m == "t-test":
                if p.get("non_parametric"):
                    if p.get("paired"):
                        return stat_lib.run_wilcoxon(table, p["group_a"], p["group_b"], p.get("alternative", "two-sided"))
                    return stat_lib.run_mann_whitney(table, p["group_a"], p["group_b"], p.get("alternative", "two-sided"))
                return stat_lib.run_ttest(table, p["group_a"], p["group_b"], p.get("paired", False), p.get("alternative", "two-sided"))
            if m in ("one-way-anova", "kruskal-wallis"):
                if p.get("non_parametric"):
                    return stat_lib.run_kruskal_wallis(table)
                return stat_lib.run_one_way_anova(table, p.get("post_hoc", "tukey"))
            if m == "two-way-anova":
                return stat_lib.run_two_way_anova(table)
            if m.startswith("correlation"):
                return stat_lib.run_correlation(table, p["col_a"], p["col_b"], p.get("method", "pearson"))
            if m == "normality":
                return stat_lib.run_normality(table)
            if m.startswith("curve-fit"):
                return stat_lib.run_curve_fit(table, p["x_col"], p["y_col"], p.get("model", "linear"))
            if m == "grubbs":
                return stat_lib.run_grubbs(table, alpha=p.get("alpha", 0.05))
            if m == "roc":
                return stat_lib.run_roc(table, p["label_col"], p["score_col"])
            if m == "rm-anova":
                return stat_lib.run_rm_anova(table)
            if m == "friedman":
                return stat_lib.run_friedman(table)
            if m == "multiple-regression":
                return stat_lib.run_multiple_regression(table, p["y_col"], p["x_cols"])
            if m == "descriptive":
                from ..stats.tests import run_descriptive
                return run_descriptive(table)
            if m in ("mann-whitney", "wilcoxon"):
                alt = p.get("alternative", "two-sided")
                if m == "wilcoxon":
                    return stat_lib.run_wilcoxon(table, p["group_a"], p["group_b"], alt)
                return stat_lib.run_mann_whitney(table, p["group_a"], p["group_b"], alt)
        except Exception:
            pass
        return None

    def _on_axes_changed(self, opts: dict):
        if self._active_gc is not None:
            self._active_gc.options["axis"] = opts

    # ------------------------------------------------------------------
    # Navigator
    # ------------------------------------------------------------------
    def _on_nav_item(self, section: str, name: str, _kind: str):
        if section == "table":
            t = self.project.get_table(name)
            if t:
                if isinstance(t, GroupedDataTable):
                    self.grouped_widget.load_table(t)
                    self.stack.setCurrentIndex(3)
                elif isinstance(t, ContingencyTable):
                    self.contingency_widget.load_table(t)
                    self.stack.setCurrentIndex(4)
                else:
                    self.table_widget.load_table(t)
                    self.stack.setCurrentIndex(0)
        elif section == "analysis":
            result = next((a for a in self.project.analyses if a.name == name), None)
            if result:
                self.results_widget.show_result(result)
                self.stack.setCurrentIndex(2)
        elif section == "graph":
            gc = next((g for g in self.project.graphs if g.name == name), None)
            if gc:
                self._render_graph(gc)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def _run_descriptive(self):
        t = self._active_table()
        if t is None:
            QMessageBox.warning(self, "No table", "Open a data table first.")
            return
        from ..stats.tests import run_descriptive
        self._save_and_show_result(run_descriptive(t))

    def _open_analysis(self, tab: int = 0):
        if not self.project.tables:
            QMessageBox.warning(self, "No data", "Create a data table first.")
            return
        dlg = AnalysisDialog(self.project.tables, parent=self, initial_tab=tab)
        if dlg.exec() and dlg.get_result():
            self._save_and_show_result(dlg.get_result())

    def _save_and_show_result(self, result):
        self.project.analyses.append(result)
        self.navigator.add_analysis(result.name)
        self.results_widget.show_result(result)
        self.stack.setCurrentIndex(2)
        self._mark_modified(f"Analysis: {result.name}")

    def open_test_wizard(self):
        if not self.project.tables:
            QMessageBox.warning(self, "No data", "Create a data table first.")
            return
        from .test_wizard import TestWizard
        dlg = TestWizard(self.project.tables, parent=self)
        if dlg.exec() and dlg.get_result():
            self._save_and_show_result(dlg.get_result())

    def open_power_dialog(self):
        from .power_dialog import PowerDialog
        dlg = PowerDialog(parent=self)
        dlg.exec()

    # ------------------------------------------------------------------
    # Distribution Inspector
    # ------------------------------------------------------------------
    def open_distribution_inspector(self):
        col_tables = [t for t in self.project.tables if isinstance(t, DataTable)]
        if not col_tables:
            QMessageBox.warning(self, "No data", "Create a data table first.")
            return
        from .distribution_inspector import DistributionInspector
        dlg = DistributionInspector(col_tables, parent=self)
        dlg.run_test_requested.connect(self._on_dist_test_requested)
        if dlg.exec():
            # Save distribution-check result to the project
            result = dlg.get_result()
            if result:
                self._save_and_show_result(result)
            # If the inspector recommended a non-parametric test, open the
            # analysis dialog on the t-test tab with the checkbox pre-ticked
            if dlg.nonpar_recommended():
                self._open_analysis_nonpar()

    def _on_dist_test_requested(self, tab_idx: int):
        """Called when user clicks 'Run recommended test' inside the inspector."""
        pass  # handled via dialog accept + nonpar_recommended()

    def _open_analysis_nonpar(self):
        """Open the analysis dialog on tab 0 with non-parametric pre-selected."""
        from .analysis_dialog import AnalysisDialog
        dlg = AnalysisDialog(self.project.tables, parent=self, initial_tab=0)
        dlg._tt_nonpar.setChecked(True)
        if dlg.exec() and dlg.get_result():
            self._save_and_show_result(dlg.get_result())

    # ------------------------------------------------------------------
    # Graphs
    # ------------------------------------------------------------------
    def new_graph(self, plot_type: str = "bar"):
        if not self.project.tables:
            QMessageBox.warning(self, "No data", "Create a data table first.")
            return
        dlg = GraphOptionsDialog(self.project.tables, initial_type=plot_type, parent=self)
        if not dlg.exec():
            return
        opts = dlg.get_options()
        # Apply journal preset
        plot_styles.set_preset(opts.get("journal_preset", "Default (Staplot)"))
        name = f"{opts['plot_type'].capitalize()} — {opts['table_name']}"
        gc = self.project.add_graph(name, opts["table_name"], opts["plot_type"], opts)
        self.navigator.add_graph(gc.name)
        self._dirty = True
        self._render_graph(gc)

    def _sig_pairs_for_table(self, table_name: str) -> list[tuple]:
        pairs = []
        for a in self.project.analyses:
            if a.source_table != table_name:
                continue
            d = a.details
            if "p" in d and "group_a" in d and "group_b" in d:
                pairs.append((d["group_a"], d["group_b"], float(d["p"])))
        return pairs

    def _plot_gc_to_fig(self, gc: GraphConfig):
        """Render a GraphConfig to a Figure with no side effects (for gallery/layout)."""
        t = self.project.get_table(gc.source_table)
        if t is None:
            return None
        opts = gc.options
        colors = opts.get("colors") or None
        sig_pairs = self._sig_pairs_for_table(gc.source_table) if opts.get("sig_brackets", True) else []
        # Select the stored preset if present (plots apply it while they are built)
        plot_styles.apply(opts.get("journal_preset"))
        try:
            return self._make_fig(gc.plot_type, t, opts, colors, sig_pairs)
        except Exception:
            return None

    def _make_fig(self, pt: str, t, opts: dict, colors, sig_pairs):
        """Pure function: build and return a Figure for the given plot type."""
        import matplotlib.pyplot as plt
        if pt in ("box", "violin"):
            return plot_lib.plot_box(t, plot_type=pt,
                                     show_points=opts.get("show_points", True),
                                     title=opts.get("title", ""),
                                     ylabel=opts.get("ylabel", ""),
                                     colors=colors,
                                     sig_pairs=sig_pairs or None)
        elif pt == "scatter":
            return plot_lib.plot_scatter(t,
                                         show_regression=opts.get("show_regression", True),
                                         title=opts.get("title", ""),
                                         xlabel=opts.get("xlabel", ""),
                                         ylabel=opts.get("ylabel", ""),
                                         colors=colors)
        elif pt == "before_after":
            return plot_lib.plot_before_after(t,
                                              show_mean=True,
                                              error_type=opts.get("error_type", "sd"),
                                              title=opts.get("title", ""),
                                              ylabel=opts.get("ylabel", ""),
                                              colors=colors)
        elif pt == "histogram":
            return plot_lib.plot_histogram(t,
                                           bins=opts.get("hist_bins", "auto"),
                                           density=opts.get("hist_density", False),
                                           show_kde=opts.get("hist_kde", True),
                                           cumulative=opts.get("hist_cumulative", False),
                                           title=opts.get("title", ""),
                                           xlabel=opts.get("xlabel", "Value"),
                                           ylabel=opts.get("ylabel", ""),
                                           colors=colors)
        elif pt == "survival":
            return plot_lib.plot_survival(t,
                                          title=opts.get("title", ""),
                                          xlabel=opts.get("xlabel", "Time"),
                                          ylabel=opts.get("ylabel", "Survival probability"))
        elif pt == "qq":
            return plot_lib.plot_qq(t,
                                    title=opts.get("title", ""),
                                    colors=colors)
        elif pt == "heatmap":
            return plot_lib.plot_heatmap(t,
                                         mode=opts.get("heatmap_mode", "values"),
                                         cmap=opts.get("heatmap_cmap", "RdYlBu_r"),
                                         annotate=opts.get("heatmap_annotate", True),
                                         title=opts.get("title", ""))
        elif pt == "roc":
            return plot_lib.plot_roc(t,
                                     label_col=opts.get("roc_label", ""),
                                     score_col=opts.get("roc_score", ""),
                                     title=opts.get("title", ""),
                                     color=colors[0] if colors else None)
        elif pt == "bland_altman":
            return plot_lib.plot_bland_altman(t,
                                              col_a=opts.get("ba_col_a", ""),
                                              col_b=opts.get("ba_col_b", ""),
                                              percentage=opts.get("ba_percentage", False),
                                              title=opts.get("title", ""),
                                              color=colors[0] if colors else None)
        elif pt == "grouped_bar":
            return plot_lib.plot_grouped_bar(t,
                                             error_type=opts.get("error_type", "sd"),
                                             show_points=opts.get("show_points", True),
                                             title=opts.get("title", ""),
                                             ylabel=opts.get("ylabel", ""),
                                             colors=colors,
                                             sig_pairs=sig_pairs or None)
        elif pt == "volcano":
            return plot_lib.plot_volcano(t,
                                         fc_col=opts.get("volcano_fc", ""),
                                         p_col=opts.get("volcano_p", ""),
                                         fc_thresh=opts.get("volcano_fc_thresh", 1.0),
                                         p_thresh=opts.get("volcano_p_thresh", 0.05),
                                         title=opts.get("title", ""),
                                         label_col=opts.get("volcano_label"),
                                         label_top=opts.get("volcano_top", 10),
                                         color_up=colors[0] if colors else None,
                                         color_down=colors[1] if colors and len(colors) > 1 else None)
        else:  # bar (default)
            return plot_lib.plot_bar(t,
                                     error_type=opts.get("error_type", "sd"),
                                     show_points=opts.get("show_points", True),
                                     title=opts.get("title", ""),
                                     ylabel=opts.get("ylabel", ""),
                                     colors=colors,
                                     sig_pairs=sig_pairs or None)

    def _render_graph(self, gc: GraphConfig, live_refresh: bool = False):
        t = self.project.get_table(gc.source_table)
        if t is None:
            return
        opts = gc.options
        colors = opts.get("colors") or None
        sig_pairs = (self._sig_pairs_for_table(gc.source_table)
                     if opts.get("sig_brackets", True) else [])
        plot_styles.apply(opts.get("journal_preset"))
        try:
            fig = self._make_fig(gc.plot_type, t, opts, colors, sig_pairs)
            axis_opts = opts.get("axis", {})
            self._active_gc = gc
            self.plot_canvas.show_figure(fig, axis_opts if axis_opts else None)
            if not live_refresh:
                self.stack.setCurrentIndex(1)
        except Exception as exc:
            if not live_refresh:
                QMessageBox.critical(self, "Plot error", str(exc))
            else:
                self._status_label.setText(f"Plot error: {exc}")

    # ------------------------------------------------------------------
    # Multi-panel / gallery
    # ------------------------------------------------------------------
    def _render_for_layout(self, gc: GraphConfig):
        """Used by FigureLayoutDialog — renders without touching the canvas."""
        return self._plot_gc_to_fig(gc)

    def open_figure_layout(self):
        if not self.project.graphs:
            QMessageBox.information(self, "No graphs", "Create some graphs first.")
            return
        from .layout_dialog import LayoutDialog
        dlg = LayoutDialog(self.project, self._plot_gc_to_fig, parent=self)
        dlg.exec()

    def open_graph_gallery(self):
        if not self.project.graphs:
            QMessageBox.information(self, "No graphs", "Create some graphs first.")
            return
        from .graph_gallery import GraphGallery
        dlg = GraphGallery(self.project, self._plot_gc_to_fig, parent=self)
        dlg.graph_selected.connect(self._on_gallery_select)
        dlg.exec()

    def _on_gallery_select(self, name: str):
        gc = next((g for g in self.project.graphs if g.name == name), None)
        if gc:
            self._render_graph(gc)

    def export_graph(self):
        self.stack.setCurrentIndex(1)
        self.plot_canvas.export_dialog()

    # ------------------------------------------------------------------
    # Standard curve
    # ------------------------------------------------------------------
    def open_standard_curve(self):
        fits = [a for a in self.project.analyses
                if "r2" in a.details and "params" in a.details]
        if not fits:
            QMessageBox.information(self, "No fits",
                                    "Run Statistics → Curve Fitting first.")
            return
        from .standard_curve_dialog import StandardCurveDialog
        dlg = StandardCurveDialog(self.project.analyses, parent=self)
        dlg.exec()

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def open_report(self):
        from .report_dialog import ReportDialog
        dlg = ReportDialog(self.project, self._plot_gc_to_fig, parent=self)
        dlg.exec()

    # ------------------------------------------------------------------
    # Multiple comparisons
    # ------------------------------------------------------------------
    def open_multiple_comparisons(self):
        from .multiple_comparisons_dialog import MultipleComparisonsDialog
        dlg = MultipleComparisonsDialog(self.project.analyses, parent=self)
        if dlg.exec() and dlg.get_result():
            self._save_and_show_result(dlg.get_result())

    # ------------------------------------------------------------------
    # Computed column
    # ------------------------------------------------------------------
    def open_computed_column(self):
        col_tables = [t for t in self.project.tables
                      if isinstance(t, DataTable)]
        if not col_tables:
            QMessageBox.warning(self, "No table", "Create a data table first.")
            return
        from .computed_column_dialog import ComputedColumnDialog
        dlg = ComputedColumnDialog(self.project.tables, parent=self)
        if dlg.exec():
            t = dlg.get_table()
            if t is not None:
                self.table_widget.load_table(t)
                self.stack.setCurrentIndex(0)
                self._mark_modified(f"Computed column added to: {t.name}")
                self._on_data_changed()

    # ------------------------------------------------------------------
    # Reshape
    # ------------------------------------------------------------------
    def open_reshape(self):
        col_tables = [t for t in self.project.tables
                      if isinstance(t, DataTable)]
        if not col_tables:
            QMessageBox.warning(self, "No table", "Create a data table first.")
            return
        from .reshape_dialog import ReshapeDialog
        dlg = ReshapeDialog(self.project.tables, parent=self)
        if dlg.exec():
            new_t = dlg.get_result()
            if new_t:
                new_t.name = self.project._table_name(new_t.name)
                self.project.tables.append(new_t)
                self.navigator.add_table(new_t.name)
                self.table_widget.load_table(new_t)
                self.stack.setCurrentIndex(0)
                self._mark_modified(f"Created: {new_t.name}")

    # ------------------------------------------------------------------
    # Export table as CSV
    # ------------------------------------------------------------------
    def export_table_csv(self):
        t = self._active_table()
        if t is None or not isinstance(t, DataTable):
            QMessageBox.warning(self, "No table",
                                "Open a standard data table first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Table as CSV", t.name + ".csv",
            "CSV (*.csv);;TSV (*.tsv);;All files (*)")
        if not path:
            return
        sep = "\t" if path.lower().endswith(".tsv") else ","
        try:
            df = t.to_dataframe()
            df.to_csv(path, sep=sep, index=False, encoding="utf-8-sig")
            self._status_label.setText(f"Exported: {os.path.basename(path)}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    # ------------------------------------------------------------------
    # Data transforms
    # ------------------------------------------------------------------
    def transform_data(self):
        if not self.project.tables:
            QMessageBox.warning(self, "No data", "Create a data table first.")
            return
        from .transform_dialog import TransformDialog
        dlg = TransformDialog(self.project.tables, parent=self)
        if dlg.exec():
            new_t = dlg.get_new_table()
            if new_t:
                new_t.name = self.project._table_name(new_t.name)
                self.project.tables.append(new_t)
                self.navigator.add_table(new_t.name)
                self.table_widget.load_table(new_t)
                self.stack.setCurrentIndex(0)
                self._mark_modified(f"Created: {new_t.name}")

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------
    def new_project(self):
        if not self._confirm_discard():
            return
        self._dirty = False
        self.project = Project()
        self._undo_stack.clear()
        self._active_gc = None
        self.navigator.clear_all()
        self._new_default_table()
        self.setWindowTitle("Staplot")
        self._status_label.setText("Ready")

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "",
            "Staplot Project (*.staplot);;JSON (*.json)")
        if not path or not self._confirm_discard():
            return
        try:
            self.project = Project.load(path)
            self._dirty = False
            self._active_gc = None
            self.navigator.populate(self.project)
            self._load_first_table()
            self.setWindowTitle(f"Staplot — {self.project.name}")
            self._add_to_recent(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))

    def _load_first_table(self):
        if not self.project.tables:
            return
        t = self.project.tables[0]
        if isinstance(t, GroupedDataTable):
            self.grouped_widget.load_table(t)
            self.stack.setCurrentIndex(3)
        elif isinstance(t, ContingencyTable):
            self.contingency_widget.load_table(t)
            self.stack.setCurrentIndex(4)
        else:
            self.table_widget.load_table(t)
            self.stack.setCurrentIndex(0)

    def _save_to(self, path) -> bool:
        try:
            self.project.save(path)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed",
                                 f"{exc}\n\nThe previous file (if any) was left unchanged.")
            return False
        self._dirty = False
        self._status_label.setText("Saved")
        return True

    def save_project(self) -> bool:
        if self.project.filepath:
            return self._save_to(self.project.filepath)
        return self.save_project_as()

    def save_project_as(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "",
            "Staplot Project (*.staplot);;JSON (*.json)")
        if not path or not self._save_to(path):
            return False
        self.setWindowTitle(f"Staplot — {self.project.name}")
        self._add_to_recent(path)
        return True

    def import_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import CSV", "",
            "CSV files (*.csv);;TSV (*.tsv *.txt);;All files (*)")
        if not path:
            return
        import csv as csv_lib
        sep = "\t" if path.lower().endswith((".tsv", ".txt")) else ","
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                rows = list(csv_lib.reader(f, delimiter=sep))
            if not rows:
                return
            headers, data_rows = rows[0], rows[1:]
            name = os.path.splitext(os.path.basename(path))[0]
            t = self.project.add_table(name)
            for h in headers:
                t.add_column(h or f"Col{len(t.columns) + 1}")
            for r_idx, row in enumerate(data_rows):
                for c_idx, cell in enumerate(row):
                    if c_idx >= len(t.columns):
                        t.add_column()
                    try:
                        val = float(cell.strip()) if cell.strip() else None
                    except ValueError:
                        val = None
                    t.set_value(r_idx, c_idx, val)
            self.navigator.add_table(t.name)
            self.table_widget.load_table(t)
            self.stack.setCurrentIndex(0)
            self._mark_modified(f"Imported: {name}")
        except Exception as exc:
            QMessageBox.critical(self, "Import failed", str(exc))

    def import_excel(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Excel", "",
            "Excel files (*.xlsx *.xls *.xlsm);;All files (*)")
        if not path:
            return
        try:
            # openpyxl alone could not read an old .xls: read_table picks the reader
            from staplot.api import read_table
            df = read_table(path)
            if df.empty:
                return
            headers = [str(h) for h in df.columns]
            data_rows = [[None if (isinstance(v, float) and v != v) else v for v in row]
                         for row in df.itertuples(index=False)]
            name = os.path.splitext(os.path.basename(path))[0]
            t = self.project.add_table(name)
            for h in headers:
                t.add_column(h)
            for r_idx, row in enumerate(data_rows):
                for c_idx, cell in enumerate(row or []):
                    if c_idx >= len(t.columns):
                        t.add_column()
                    try:
                        val = float(cell) if cell is not None else None
                    except (ValueError, TypeError):
                        val = None
                    t.set_value(r_idx, c_idx, val)
            self.navigator.add_table(t.name)
            self.table_widget.load_table(t)
            self.stack.setCurrentIndex(0)
            self._mark_modified(f"Imported: {name}")
        except ImportError as exc:
            QMessageBox.critical(self, "Missing dependency", str(exc))
        except Exception as exc:
            QMessageBox.critical(self, "Import failed", str(exc))

    # ------------------------------------------------------------------
    def _about(self):
        QMessageBox.about(
            self, "About Staplot",
            "<b>Staplot 0.6</b><br>"
            "Statistical analysis and publication-quality plotting.<br>"
            "A GraphPad Prism replacement built in Python.<br><br>"
            "Features: descriptive stats, t-tests, ANOVA, RM-ANOVA, "
            "correlation, curve fitting, chi-square, Fisher, Grubbs, "
            "power analysis, ROC, Bland-Altman, Q-Q, heatmap, survival, "
            "significance brackets, journal presets, and more.",
        )

    def _confirm_discard(self) -> bool:
        """True when it is fine to drop the current project (saved, or the
        user chose Save / Discard)."""
        if not self._dirty:
            return True
        ans = QMessageBox.question(
            self, "Unsaved changes", "Save changes to the current project?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel)
        if ans == QMessageBox.StandardButton.Save:
            return self.save_project()
        return ans == QMessageBox.StandardButton.Discard

    def closeEvent(self, event):
        if self._confirm_discard():
            event.accept()
        else:
            event.ignore()
