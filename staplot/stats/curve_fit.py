import numpy as np
from scipy.optimize import curve_fit

from ..models.project import DataTable, AnalysisResult


def _linear(x, slope, intercept):
    return slope * x + intercept


def _exponential(x, a, b):
    return a * np.exp(b * x)


def _hill(x, bottom, top, ec50, hillslope):
    """4-parameter logistic (Hill) — dose-response / sigmoidal."""
    return bottom + (top - bottom) / (1.0 + (ec50 / np.where(x == 0, 1e-12, x)) ** hillslope)


def _michaelis_menten(x, vmax, km):
    return vmax * x / (km + x)


MODELS: dict[str, tuple] = {
    "linear": (_linear, ["slope", "intercept"]),
    "exponential": (_exponential, ["a", "b"]),
    "hill": (_hill, ["Bottom", "Top", "EC50", "HillSlope"]),
    "michaelis_menten": (_michaelis_menten, ["Vmax", "Km"]),
}


def run_curve_fit(
    table: DataTable, x_col: str, y_col: str, model: str = "linear"
) -> AnalysisResult:
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose: {list(MODELS)}")

    groups = table.group_arrays()
    if x_col not in groups or y_col not in groups:
        raise ValueError("Columns not found in table.")

    # an (x, y) point is one row: drop rows where either is missing / excluded
    x, y = table.paired_arrays(x_col, y_col)

    if len(x) < 3:
        raise ValueError("Need at least 3 data points for curve fitting.")

    func, param_names = MODELS[model]
    popt, pcov = curve_fit(func, x, y, maxfev=50_000)

    y_pred = func(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    perr = np.sqrt(np.diag(pcov))

    param_header = f"{'Parameter':15s} {'Value':>15s} {'Std Error':>15s}"
    param_rows = [
        f"{name:15s} {val:>15.6g} {err:>15.6g}"
        for name, val, err in zip(param_names, popt, perr)
    ]

    lines = [
        f"=== Curve Fitting ({model}) ===",
        f"X: {x_col}  |  Y: {y_col}",
        f"N = {len(x)}",
        f"R² = {r2:.6f}",
        "",
        param_header,
        "-" * len(param_header),
        *param_rows,
    ]
    return AnalysisResult(
        name=f"Curve fit ({model}): {y_col} vs {x_col}",
        method=f"curve-fit-{model}",
        source_table=table.name,
        summary_text="\n".join(lines),
        details={
            "model": model,
            "params": {n: float(v) for n, v in zip(param_names, popt)},
            "r2": float(r2),
            "x_col": x_col,
            "y_col": y_col,
        },
    )
