"""Pearson and Spearman correlation."""

import numpy as np
from scipy import stats

from ..models.project import DataTable, AnalysisResult
from .descriptive import format_p, significance_stars


def run_correlation(
    table: DataTable,
    col_a: str,
    col_b: str,
    method: str = "pearson",
) -> AnalysisResult:
    """method: 'pearson' | 'spearman' | 'kendall'"""
    # x/y pairs come from the same row: drop a row if either value is missing
    a, b = table.paired_arrays(col_a, col_b)

    if len(a) < 3:
        raise ValueError("Need at least 3 paired values for correlation.")

    method = method.lower()
    if method == "pearson":
        r, p = stats.pearsonr(a, b)
        label = "Pearson r"
    elif method == "spearman":
        r, p = stats.spearmanr(a, b)
        label = "Spearman ρ"
    elif method == "kendall":
        r, p = stats.kendalltau(a, b)
        label = "Kendall τ"
    else:
        raise ValueError(f"Unknown method '{method}'.")

    r2 = r ** 2
    # 95% CI for Pearson r via Fisher Z
    ci_lo = ci_hi = np.nan
    if method == "pearson" and len(a) > 3:
        z = np.arctanh(np.clip(r, -1 + 1e-15, 1 - 1e-15))   # |r| = 1 → CI [±1, ±1]
        se = 1.0 / np.sqrt(len(a) - 3)
        z_lo = z - 1.96 * se
        z_hi = z + 1.96 * se
        ci_lo, ci_hi = float(np.tanh(z_lo)), float(np.tanh(z_hi))

    lines = [
        f"=== {label.split()[0]} Correlation ===",
        f"{col_a}  vs  {col_b}",
        f"N = {len(a)}",
        "",
        f"{label:12s} = {r:>8.4f}",
        f"R²          = {r2:>8.4f}",
        f"{format_p(float(p)):20s}  {significance_stars(float(p))}",
    ]
    if not np.isnan(ci_lo):
        lines.append(f"95% CI       [{ci_lo:.4f},  {ci_hi:.4f}]")

    return AnalysisResult(
        name=f"{label}: {col_a} vs {col_b}",
        method=f"correlation-{method}",
        source_table=table.name,
        summary_text="\n".join(lines),
        details={
            "r": float(r),
            "r2": float(r2),
            "p": float(p),
            "n": int(len(a)),
            "col_a": col_a,
            "col_b": col_b,
        },
    )
