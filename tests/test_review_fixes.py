"""Fixes beyond ChatGPT's regression list: every row-paired analysis/plot keeps
row identity, unique names, safe formulas, rename keeps references, dirty flag."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from staplot.models.project import DataTable, Project, unique_name


def table(**columns):
    t = DataTable("t")
    for name, values in columns.items():
        t.add_column(name).values = list(values)
    return t


def test_paired_arrays_intersects_rows_and_exclusions():
    t = table(a=[1, None, 3, 4, 5], b=[10, 20, None, 40, 50])
    t.columns[0].excluded.add(4)
    a, b = t.paired_arrays("a", "b")
    assert a.tolist() == [1, 4] and b.tolist() == [10, 40]


def test_wilcoxon_pairs_by_row():
    from staplot.stats.tests import run_wilcoxon
    t = table(a=[1, None, 4, 9, 12, 3, 7], b=[2, 5, None, 11, 16, 5, 6])
    exp = stats.wilcoxon([1, 9, 12, 3, 7], [2, 11, 16, 5, 6])
    assert run_wilcoxon(t, "a", "b").details["p"] == pytest.approx(exp.pvalue)


def test_rm_anova_and_friedman_use_complete_subjects():
    from staplot.stats.repeated_measures import run_rm_anova, run_friedman
    t = table(a=[1, 2, None, 4, 5], b=[2, 3, 9, 5, 7], c=[3, 5, 4, None, 8])
    # complete subjects: rows 0, 1, 4
    exp = stats.friedmanchisquare([1, 2, 5], [2, 3, 7], [3, 5, 8])
    assert run_friedman(t).details["chi2"] == pytest.approx(exp.statistic)
    assert "Subjects: 3" in run_rm_anova(t).summary_text


def test_curve_fit_pairs_by_row():
    from staplot.stats.curve_fit import run_curve_fit
    t = table(x=[1, 2, 3, 4, 5], y=[2, None, 6, 8, 10])
    r = run_curve_fit(t, "x", "y", "linear")
    assert "N = 4" in r.summary_text and "R² = 1.0" in r.summary_text


def test_scatter_plot_points_stay_paired():
    from staplot.plots.scatter import plot_scatter
    fig = plot_scatter(table(x=[1, 2, 3, 4, 5], y=[2, None, 6, 8, 10]), show_regression=False)
    pts = fig.axes[0].collections[0].get_offsets()
    assert np.asarray(pts).tolist() == [[1, 2], [3, 6], [4, 8], [5, 10]]


def test_bland_altman_and_roc_plot_pair_by_row():
    from staplot.plots.bland_altman import plot_bland_altman
    from staplot.plots.roc_plot import plot_roc
    fig = plot_bland_altman(table(a=[1, None, 3, 4], b=[1, 9, 3, 5]), "a", "b")
    diffs = np.asarray(fig.axes[0].collections[0].get_offsets())[:, 1]
    assert diffs.tolist() == [0, 0, -1]
    fig = plot_roc(table(label=[1, None, 0, 0], score=[4, 3, 2, 1]), "label", "score")
    assert "AUC = 1.000" in [t.get_text() for t in fig.axes[0].get_legend().get_texts()][0]


def test_volcano_labels_follow_rows():
    from staplot.plots.volcano import plot_volcano
    t = table(fc=[3, None, -3, 0.1], p=[0.001, 0.001, 0.002, 0.5])
    t.add_column("name").values = ["geneA", "geneB", "geneC", "geneD"]
    fig = plot_volcano(t, "fc", "p", label_col="name", label_top=5)
    texts = sorted(a.get_text() for a in fig.axes[0].texts)
    assert texts == ["geneA", "geneC"]


def test_logrank_matches_statsmodels_with_censoring():
    from statsmodels.duration.survfunc import survdiff
    from staplot.stats.roc import run_log_rank
    ta, ea = [2, 3, 3, 6, 7, 10], [1, 0, 1, 1, 0, 1]
    tb, eb = [1, 1, 4, 5, 5, 8], [1, 1, 1, 0, 1, 1]
    t = table(ta=ta, ea=ea, tb=tb, eb=eb)
    exp_stat, exp_p = survdiff(np.array(ta + tb, float), np.array(ea + eb, float),
                               np.array([0] * 6 + [1] * 6))
    r = run_log_rank(t, "ta", "ea", "tb", "eb")
    assert r.details["chi2"] == pytest.approx(exp_stat)
    assert r.details["p"] == pytest.approx(exp_p)


def test_roc_auc_matches_pairwise_ranking():
    from staplot.stats.roc import compute_roc
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 60)
    scores = np.round(rng.normal(size=60) + labels, 1)        # many ties
    pos, neg = scores[labels == 1], scores[labels == 0]
    exp = ((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean()
    assert compute_roc(labels, scores)[2] == pytest.approx(exp)


def test_unique_names():
    t = table(a=[1])
    assert t.add_column("a").name == "a (2)"
    with pytest.raises(ValueError):
        t.rename_column(1, "a")
    p = Project()
    p.add_table("Data")
    assert p.add_table("Data").name == "Data (2)"
    assert unique_name("x", ["x", "x (2)"]) == "x (3)"


def test_formula_allow_list():
    from staplot.ui.computed_column_dialog import eval_formula
    data = {"A": np.array([1.0, 4.0]), "B": np.array([2.0])}
    assert eval_formula("log2(A) + 1", data).tolist() == [1.0, 3.0]
    assert np.isnan(eval_formula("A / B", data)[1])             # ragged → NaN
    assert eval_formula("2", data).tolist() == [2.0, 2.0]
    for bad in ("A.__class__", "__import__('os')", "[A][0]", "A if A else B", "open"):
        with pytest.raises(ValueError):
            eval_formula(bad, data)
    with pytest.raises(ValueError, match="both become"):
        eval_formula("A_B", {"A B": np.ones(1), "A-B": np.ones(1)})
    with pytest.raises(ValueError, match="shadows"):
        eval_formula("log", {"log": np.ones(1)})


def test_rename_table_keeps_graph_and_analysis_refs(qtbot, monkeypatch):
    from qtpy.QtWidgets import QInputDialog
    from staplot.ui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    t = w._active_table()
    g = w.project.add_graph("g", t.name, "bar")
    from staplot.models.project import AnalysisResult
    w.project.analyses.append(AnalysisResult("a", "m", t.name, ""))
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Renamed", True)))
    w.rename_table()
    assert t.name == "Renamed" and g.source_table == "Renamed"
    assert w.project.analyses[0].source_table == "Renamed"
    assert w._dirty
    w._dirty = False


def test_autosave_uses_dirty_flag_and_reports_failure(qtbot, tmp_path, monkeypatch):
    from staplot.ui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    target = tmp_path / "p.staplot"
    w.project.save(str(target))
    w._mark_modified("Created: something")       # informational text, still dirty
    w._auto_save()
    assert not w._dirty and w._status_label.text() == "Auto-saved"

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(w.project, "save", boom)
    w._mark_modified()
    w._auto_save()
    assert w._dirty and "FAILED" in w._status_label.text()
    w._dirty = False
