"""Chi-square test of independence and Fisher's exact test for contingency tables."""

import numpy as np
from scipy import stats

from ..models.project import ContingencyTable, AnalysisResult
from .descriptive import format_p, significance_stars


def run_chi_square(table: ContingencyTable) -> AnalysisResult:
    counts = table.get_matrix().astype(float)
    if counts.shape[0] < 2 or counts.shape[1] < 2:
        raise ValueError("Need at least a 2×2 contingency table.")

    chi2, p, dof, expected = stats.chi2_contingency(counts, correction=False)
    chi2_yates, p_yates, _, _ = stats.chi2_contingency(counts, correction=True)

    # Cramér's V
    n = counts.sum()
    v = np.sqrt(chi2 / (n * (min(counts.shape) - 1))) if n > 0 else np.nan

    # Row / column totals
    row_totals = counts.sum(axis=1)
    col_totals = counts.sum(axis=0)

    obs_header = f"{'':20s}" + "".join(f"{c:>10s}" for c in table.col_labels) + f"{'Total':>10s}"
    obs_rows = []
    for i, rl in enumerate(table.row_labels):
        row_str = f"{rl:20s}" + "".join(f"{counts[i,j]:>10.0f}" for j in range(len(table.col_labels)))
        row_str += f"{row_totals[i]:>10.0f}"
        obs_rows.append(row_str)
    total_row = f"{'Total':20s}" + "".join(f"{c:>10.0f}" for c in col_totals) + f"{n:>10.0f}"

    exp_header = f"\nExpected counts:"
    exp_rows = []
    for i, rl in enumerate(table.row_labels):
        row_str = f"{rl:20s}" + "".join(f"{expected[i,j]:>10.2f}" for j in range(len(table.col_labels)))
        exp_rows.append(row_str)

    lines = [
        "=== Chi-Square Test of Independence ===",
        "",
        "Observed counts:",
        obs_header,
        "-" * len(obs_header),
        *obs_rows,
        total_row,
        exp_header,
        *exp_rows,
        "",
        f"χ²          = {chi2:.4f}  (df={dof})",
        f"χ² (Yates)  = {chi2_yates:.4f}",
        f"Cramér's V  = {v:.4f}",
        "",
        f"  Without correction:  {format_p(p)}  {significance_stars(p)}",
        f"  Yates' correction:   {format_p(p_yates)}  {significance_stars(p_yates)}",
    ]
    return AnalysisResult(
        name=f"Chi-square: {table.name}",
        method="chi-square",
        source_table=table.name,
        summary_text="\n".join(lines),
        details={"chi2": float(chi2), "p": float(p), "dof": int(dof), "cramers_v": float(v)},
    )


def run_fisher_exact(table: ContingencyTable) -> AnalysisResult:
    counts = table.get_matrix()
    if counts.shape != (2, 2):
        raise ValueError("Fisher's exact test requires exactly a 2×2 table.")

    odds_ratio, p = stats.fisher_exact(counts)
    lines = [
        "=== Fisher's Exact Test (2×2) ===",
        "",
        f"Odds Ratio  = {odds_ratio:.4f}",
        f"{format_p(float(p))}  {significance_stars(float(p))}",
    ]
    return AnalysisResult(
        name=f"Fisher's exact: {table.name}",
        method="fisher-exact",
        source_table=table.name,
        summary_text="\n".join(lines),
        details={"odds_ratio": float(odds_ratio), "p": float(p)},
    )
