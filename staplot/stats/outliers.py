"""Grubbs outlier test (single outlier) applied column-wise."""

import numpy as np
from scipy import stats

from ..models.project import DataTable, AnalysisResult
from .descriptive import format_p


def _grubbs_critical(n: int, alpha: float = 0.05) -> float:
    t_crit = stats.t.ppf(alpha / (2 * n), df=n - 2)
    return ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit ** 2 / (n - 2 + t_crit ** 2))


def grubbs_test_column(arr: np.ndarray, alpha: float = 0.05) -> dict:
    clean = arr[~np.isnan(arr)]
    n = len(clean)
    if n < 3:
        return {"n": n, "error": "Need n ≥ 3"}
    mean, std = np.mean(clean), np.std(clean, ddof=1)
    if std == 0:
        return {"n": n, "G": 0.0, "G_crit": np.nan, "outlier": False}
    G = np.max(np.abs(clean - mean)) / std
    G_crit = _grubbs_critical(n, alpha)
    idx = int(np.argmax(np.abs(clean - mean)))
    return {
        "n": n,
        "G": float(G),
        "G_crit": float(G_crit),
        "outlier": bool(G > G_crit),
        "suspect_value": float(clean[idx]),
        "suspect_index": idx,
    }


def run_grubbs(table: DataTable, alpha: float = 0.05) -> AnalysisResult:
    lines = [f"=== Grubbs Outlier Test (alpha = {alpha}) ===", ""]
    all_details: dict = {}

    for col in table.columns:
        arr = col.valid_values()
        r = grubbs_test_column(arr, alpha)
        lines.append(f"--- {col.name} ---")
        if "error" in r:
            lines.append(f"  {r['error']}")
        else:
            lines.append(f"  N = {r['n']}")
            lines.append(f"  G = {r['G']:.4f}  (critical = {r['G_crit']:.4f})")
            if r["outlier"]:
                lines.append(f"  → OUTLIER detected: value = {r['suspect_value']:.4f}  (index {r['suspect_index']})")
            else:
                lines.append("  → No significant outlier detected.")
        lines.append("")
        all_details[col.name] = r

    return AnalysisResult(
        name=f"Grubbs test: {table.name}",
        method="grubbs",
        source_table=table.name,
        summary_text="\n".join(lines),
        details=all_details,
    )
