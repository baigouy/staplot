"""Multiple-comparison correction (Bonferroni, FDR, Holm, …)."""

import numpy as np
from statsmodels.stats.multitest import multipletests

from ..models.project import AnalysisResult
from .descriptive import format_p, significance_stars

_METHOD_LABELS = {
    "bonferroni": "Bonferroni",
    "fdr_bh":     "Benjamini-Hochberg (FDR)",
    "fdr_by":     "Benjamini-Yekutieli (FDR)",
    "holm":       "Holm (Holm-Bonferroni)",
    "holm-sidak": "Holm-Sidak",
    "sidak":      "Sidak",
}


def run_multiple_comparisons(
    p_values: list[float],
    names: list[str],
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> AnalysisResult:
    """
    Apply a multiple-testing correction to a list of p-values.

    Parameters
    ----------
    p_values : raw p-values (one per comparison)
    names    : labels for each comparison
    method   : statsmodels multipletests method string
    alpha    : family-wise significance threshold
    """
    if len(p_values) == 0:
        raise ValueError("No p-values provided.")
    if len(p_values) != len(names):
        raise ValueError("p_values and names must have the same length.")

    p_arr                          = np.array(p_values, dtype=float)
    reject, p_corr, alpha_s, alpha_b = multipletests(p_arr, alpha=alpha, method=method)

    method_label = _METHOD_LABELS.get(method, method)
    n_sig        = int(np.sum(reject))

    col_w = max(30, max(len(n) for n in names) + 2)
    hdr   = f"{'Comparison':{col_w}s} {'Raw p':>10s} {'Adj. p':>12s} {'Sig':>6s}"
    sep   = "─" * len(hdr)
    rows  = []
    for nm, pr, pa, rej in zip(names, p_values, p_corr, reject):
        stars = significance_stars(float(pa)) if rej else ""
        rows.append(
            f"{nm:{col_w}s} {format_p(float(pr)):>10s}"
            f" {format_p(float(pa)):>12s} {'YES' + stars:>6s}"
            if rej else
            f"{nm:{col_w}s} {format_p(float(pr)):>10s}"
            f" {format_p(float(pa)):>12s} {'—':>6s}"
        )

    lines = [
        f"=== Multiple Comparisons Correction ({method_label}) ===",
        f"Comparisons : {len(p_values)}",
        f"alpha       : {alpha:.3f}",
        "",
        hdr,
        sep,
        *rows,
        sep,
        f"Significant after correction: {n_sig} / {len(p_values)}",
        "",
        "Note: 'Adj. p' is the corrected p-value (or Bonferroni-adjusted threshold).",
    ]

    return AnalysisResult(
        name=f"Multiple comparisons ({method_label})",
        method=f"mc-{method}",
        source_table="",
        summary_text="\n".join(lines),
        details={
            "method":      method,
            "alpha":       alpha,
            "names":       names,
            "p_raw":       list(p_values),
            "p_corrected": p_corr.tolist(),
            "reject":      reject.tolist(),
            "n_sig":       n_sig,
        },
    )


def extract_pairwise_p(analyses: list) -> tuple[list[str], list[float]]:
    """
    Scan a list of AnalysisResult objects and collect pairwise p-values
    (those stored with 'group_a', 'group_b', 'p' keys in details).
    Returns (names, p_values).
    """
    names, pvals = [], []
    for a in analyses:
        d = a.details
        if "group_a" in d and "group_b" in d and "p" in d:
            label = f"{d['group_a']} vs {d['group_b']}"
            names.append(label)
            pvals.append(float(d["p"]))
    return names, pvals
