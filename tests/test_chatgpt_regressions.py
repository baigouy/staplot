"""Regression tests from ChatGPT's 2026-10-02 review
(suggested_chatGPT_fixes/code_review_findings.md), integrated into the normal
suite — self-contained, so they keep running if that folder is removed.
All 10 findings are fixed; these lock the behaviour in."""
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from staplot.models.project import DataTable, Project
from staplot.stats.correlation import run_correlation
from staplot.stats.tests import run_ttest
from staplot.stats.roc import compute_roc, run_roc, run_log_rank
from staplot.stats.regression import run_multiple_regression
from staplot.stats.multiple_comparisons import run_multiple_comparisons


def table(**columns):
    result = DataTable('isolated review data')
    for name, values in columns.items():
        result.add_column(name).values = list(values)
    return result


# finding 1: paired analyses keep subject / row identity
@pytest.mark.parametrize('excluded', [False, True])
def test_correlation_keeps_original_row_pairs(excluded):
    t = table(x=[1, 2, 3, 4, 5], y=[2, 100, 6, 8, 10])
    if excluded:
        t.columns[1].excluded.add(1)
    else:
        t.columns[1].values[1] = None
    r = run_correlation(t, 'x', 'y')
    assert r.details['n'] == 4
    assert r.details['r'] == pytest.approx(1.0)


def test_paired_ttest_keeps_original_row_pairs():
    t = table(a=[1, None, 4, 9, 12], b=[2, 5, None, 11, 16])
    expected = stats.ttest_rel([1, 9, 12], [2, 11, 16])
    actual = run_ttest(t, 'a', 'b', paired=True)
    assert actual.details['t'] == pytest.approx(expected.statistic)
    assert actual.details['p'] == pytest.approx(expected.pvalue)


# finding 2: ROC ties get half credit whatever their order
def test_roc_ties_are_half_credit_and_order_invariant():
    scores = np.array([1., 1., 1., 1.])
    for labels in [np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])]:
        assert compute_roc(labels, scores)[2] == pytest.approx(0.5)


# finding 6: the Youden threshold is read at the same curve point
def test_roc_threshold_matches_curve_point():
    result = run_roc(table(label=[1, 1, 0, 0], score=[4, 3, 2, 1]), 'label', 'score')
    assert 'threshold (Youden index) ≈ 3.0000' in result.summary_text


# finding 7: fractional labels are rejected before any cast
def test_roc_rejects_fractional_labels_before_cast():
    with pytest.raises(ValueError):
        run_roc(table(label=[0, 0.5, 1, 1], score=[1, 2, 3, 4]), 'label', 'score')


# finding 8: ROC needs both classes
def test_roc_requires_both_classes():
    with pytest.raises(ValueError):
        run_roc(table(label=[1, 1, 1], score=[1, 2, 3]), 'label', 'score')


# finding 3: log-rank uses the risk-set variance (matches statsmodels)
def test_logrank_uses_risk_set_variance():
    from statsmodels.duration.survfunc import survdiff
    t = table(ta=[1, 2, 3, 4], ea=[1, 1, 1, 1], tb=[5, 6, 7, 8], eb=[1, 1, 1, 1])
    expected_stat, expected_p = survdiff(np.arange(1., 9.), np.ones(8), np.array([0] * 4 + [1] * 4))
    result = run_log_rank(t, 'ta', 'ea', 'tb', 'eb')
    assert result.details['chi2'] == pytest.approx(expected_stat)
    assert result.details['p'] == pytest.approx(expected_p)


# finding 4: regression ignores excluded rows
def test_regression_respects_excluded_rows():
    t = table(x=[1, 2, 3, 4, 5], y=[2, 4, 1000, 8, 10])
    t.columns[1].excluded.add(2)
    result = run_multiple_regression(t, 'y', ['x'])
    assert result.details['n'] == 4
    assert result.details['coefs']['x'] == pytest.approx(2.0)


# finding 9: undoing a column removal restores its exclusions
def test_remove_column_undo_restores_exclusions():
    from staplot.ui.undo_commands import RemoveColumnCommand
    t = table(a=[1, 2, 3])
    t.columns[0].excluded.add(1)
    widget = Mock()
    command = RemoveColumnCommand(widget, t, 0)
    command.redo()
    command.undo()
    assert t.columns[0].values == [1, 2, 3]
    assert t.columns[0].excluded == {1}


# finding 5: a failed save leaves the previous project file intact
def test_failed_save_preserves_existing_project(tmp_path):
    target = tmp_path / 'project.json'
    p = Project('original')
    p.save(str(target))
    original = target.read_bytes()

    def interrupted_dump(data, stream, **kwargs):
        stream.write('{partial')
        raise OSError('simulated full disk')
    with patch('staplot.models.project.json.dump', side_effect=interrupted_dump):
        with pytest.raises(OSError):
            p.save(str(target))
    assert target.read_bytes() == original
    assert [f.name for f in tmp_path.iterdir()] == ['project.json']   # no temp left


# finding 10: 'holm' is labelled Holm, not Holm-Sidak
def test_holm_correction_is_labeled_correctly():
    result = run_multiple_comparisons([0.01, 0.03, 0.04], ['a', 'b', 'c'], method='holm')
    assert 'Holm-Sidak' not in result.summary_text


# controls (were already correct)
def test_control_unpaired_welch_matches_scipy():
    t = table(a=[1, 3, 5, 9], b=[2, 4, 8, 16])
    expected = stats.ttest_ind(t.columns[0].as_array(), t.columns[1].as_array(), equal_var=False)
    result = run_ttest(t, 'a', 'b')
    assert result.details['p'] == pytest.approx(expected.pvalue)


def test_control_project_roundtrip_preserves_exclusions(tmp_path):
    p = Project()
    t = p.add_table()
    c = t.add_column('a')
    c.values = [1, None, 3]
    c.excluded.add(2)
    target = tmp_path / 'project.json'
    p.save(str(target))
    loaded = Project.load(str(target))
    assert loaded.tables[0].columns[0].excluded == {2}
    assert loaded.tables[0].columns[0].values == [1, None, 3]
