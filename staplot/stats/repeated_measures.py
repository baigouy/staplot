"""Repeated-measures ANOVA (parametric) and Friedman test (non-parametric)."""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
import itertools

from ..models.project import DataTable, AnalysisResult
from .descriptive import format_p, significance_stars


def _complete_subjects(table: DataTable):
    """(names, {name: array}) restricted to subjects (rows) measured under
    EVERY treatment, so index i is the same subject in each array."""
    names = [c.name for c in table.columns if len(c.valid_values()) > 0]
    if len(names) < 2:
        raise ValueError("Need at least 2 groups (columns).")
    arrays = table.paired_arrays(*names)
    return names, dict(zip(names, arrays))


def run_rm_anova(table: DataTable) -> AnalysisResult:
    """One-way repeated-measures ANOVA. Columns = treatments, rows = subjects."""
    names, groups = _complete_subjects(table)
    min_n = len(groups[names[0]])
    if min_n < 2:
        raise ValueError("Need at least 2 subjects (rows) measured under every treatment.")

    # Build long-format dataframe
    rows = []
    for subj_idx in range(min_n):
        for grp_name, arr in groups.items():
            rows.append({"subject": subj_idx, "treatment": grp_name, "value": arr[subj_idx]})
    df = pd.DataFrame(rows)

    model = AnovaRM(df, "value", "subject", within=["treatment"])
    result = model.fit()
    anova_tbl = result.anova_table

    F = float(anova_tbl["F Value"].iloc[0])
    p = float(anova_tbl["Pr > F"].iloc[0])
    df_num = float(anova_tbl["Num DF"].iloc[0])
    df_den = float(anova_tbl["Den DF"].iloc[0])

    # Pairwise post-hoc (paired t-test + Bonferroni)
    pairs = list(itertools.combinations(names, 2))
    raw_p = []
    for a, b in pairs:
        _, pij = stats.ttest_rel(groups[a], groups[b])
        raw_p.append(pij)
    _, adj_p, _, _ = multipletests(raw_p, method="bonferroni")

    ph_lines = ["", "Post-hoc (paired t-test, Bonferroni):"]
    ph_lines.append(f"{'Comparison':35s} {'adj. p':>10s} {'sig':>5s}")
    for (a, b), ap in zip(pairs, adj_p):
        ph_lines.append(f"{a} vs {b:25s} {ap:>10.4f} {significance_stars(ap):>5s}")

    lines = [
        "=== One-Way Repeated-Measures ANOVA ===",
        f"Subjects: {min_n}  |  Treatments: {len(names)}",
        "",
        f"F({df_num:.0f}, {df_den:.0f}) = {F:.4f}",
        f"{format_p(p)}  {significance_stars(p)}",
        *ph_lines,
    ]
    return AnalysisResult(
        name=f"RM ANOVA: {table.name}",
        method="rm-anova",
        source_table=table.name,
        summary_text="\n".join(lines),
        details={"F": F, "p": p, "groups": names},
    )


def run_friedman(table: DataTable) -> AnalysisResult:
    """Friedman test (non-parametric RM ANOVA). Columns = treatments, rows = blocks."""
    names, groups = _complete_subjects(table)
    min_n = len(groups[names[0]])
    if min_n < 2:
        raise ValueError("Need at least 2 subjects (rows) measured under every treatment.")
    arrays = [groups[n] for n in names]

    stat, p = stats.friedmanchisquare(*arrays)

    # Pairwise Wilcoxon post-hoc + Bonferroni
    pairs = list(itertools.combinations(range(len(names)), 2))
    raw_p = []
    for i, j in pairs:
        _, pij = stats.wilcoxon(arrays[i], arrays[j])
        raw_p.append(pij)
    _, adj_p, _, _ = multipletests(raw_p, method="bonferroni")

    ph_lines = ["", "Post-hoc (Wilcoxon, Bonferroni):"]
    ph_lines.append(f"{'Comparison':35s} {'adj. p':>10s} {'sig':>5s}")
    for (i, j), ap in zip(pairs, adj_p):
        ph_lines.append(f"{names[i]} vs {names[j]:25s} {ap:>10.4f} {significance_stars(ap):>5s}")

    lines = [
        "=== Friedman Test ===",
        f"Subjects: {min_n}  |  Treatments: {len(names)}",
        "",
        f"χ²({len(names)-1}) = {stat:.4f}",
        f"{format_p(float(p))}  {significance_stars(float(p))}",
        *ph_lines,
    ]
    return AnalysisResult(
        name=f"Friedman: {table.name}",
        method="friedman",
        source_table=table.name,
        summary_text="\n".join(lines),
        details={"chi2": float(stat), "p": float(p), "groups": names},
    )
