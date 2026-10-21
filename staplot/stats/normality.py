"""Normality tests: Shapiro-Wilk, D'Agostino-Pearson, KS-test."""

import numpy as np
from scipy import stats

from ..models.project import DataTable, AnalysisResult
from .descriptive import format_p, significance_stars


def run_normality(table: DataTable) -> AnalysisResult:
    lines = ["=== Normality Tests ===", ""]
    all_details: dict = {}

    for col in table.columns:
        arr = col.valid_values()
        n = len(arr)
        col_detail: dict = {"n": n}
        lines.append(f"--- {col.name}  (n={n}) ---")

        if n < 3:
            lines.append("  Insufficient data (need n ≥ 3).")
            lines.append("")
            continue

        # Shapiro-Wilk  (best for n ≤ 50; also valid for larger n)
        w, p_sw = stats.shapiro(arr)
        col_detail["shapiro_W"] = float(w)
        col_detail["shapiro_p"] = float(p_sw)
        lines.append(f"  Shapiro-Wilk       W = {w:.4f}   {format_p(p_sw):15s} {significance_stars(p_sw)}")

        # D'Agostino-Pearson (needs n ≥ 8)
        if n >= 8:
            k2, p_dp = stats.normaltest(arr)
            col_detail["dagostino_K2"] = float(k2)
            col_detail["dagostino_p"] = float(p_dp)
            lines.append(f"  D'Agostino-Pearson K² = {k2:.4f}  {format_p(p_dp):15s} {significance_stars(p_dp)}")
        else:
            p_dp = np.nan
            lines.append(f"  D'Agostino-Pearson — requires n ≥ 8")

        # Kolmogorov-Smirnov against a normal with same mean/sd
        mu, sigma = np.mean(arr), np.std(arr, ddof=1)
        if sigma > 0:
            ks, p_ks = stats.kstest(arr, "norm", args=(mu, sigma))
            col_detail["ks_stat"] = float(ks)
            col_detail["ks_p"] = float(p_ks)
            lines.append(f"  KS test            D = {ks:.4f}   {format_p(p_ks):15s} {significance_stars(p_ks)}")

        # Verdict: pass if Shapiro-Wilk p > 0.05
        normal = p_sw > 0.05
        verdict = (
            "PASS — data appear normally distributed. Parametric tests are appropriate."
            if normal
            else "FAIL — data deviate from normality. Consider non-parametric alternatives."
        )
        lines.append(f"  → {verdict}")
        lines.append("")
        col_detail["passed"] = normal
        all_details[col.name] = col_detail

    return AnalysisResult(
        name=f"Normality tests: {table.name}",
        method="normality",
        source_table=table.name,
        summary_text="\n".join(lines),
        details=all_details,
    )
