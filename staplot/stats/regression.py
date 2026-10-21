"""Multiple linear regression via statsmodels OLS."""

import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

from ..models.project import DataTable, AnalysisResult
from .descriptive import format_p, significance_stars


def run_multiple_regression(
    table: DataTable,
    y_col: str,
    x_cols: list[str],
) -> AnalysisResult:
    """
    OLS multiple linear regression.
    Returns AnalysisResult with fitted values and residuals in details
    so plot_residuals() can be called downstream.
    """
    if not x_cols:
        raise ValueError("Select at least one predictor (X column).")

    # excluded cells -> NaN (rows stay aligned), then complete rows only
    df = table.analysis_dataframe()
    for col in [y_col] + x_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in table.")

    sub = df[[y_col] + x_cols].dropna()
    n   = len(sub)
    k   = len(x_cols)

    if n < k + 2:
        raise ValueError(f"Too few complete rows ({n}); need at least {k + 2}.")

    y = sub[y_col].values
    X = sm.add_constant(sub[x_cols].values, has_constant="add")
    col_names = ["(Intercept)"] + x_cols

    res = sm.OLS(y, X).fit()
    dw  = float(durbin_watson(res.resid))

    # ── Format output ──────────────────────────────────────────────────
    hdr = f"{'Variable':22s} {'Coeff':>12s} {'Std Err':>12s} {'t':>9s} {'p':>11s}"
    sep = "─" * len(hdr)
    param_rows = []
    for name, coef, se, t_val, pv in zip(
            col_names, res.params, res.bse, res.tvalues, res.pvalues):
        stars = significance_stars(float(pv))
        param_rows.append(
            f"{name:22s} {coef:>12.4f} {se:>12.4f} {t_val:>9.3f}"
            f" {format_p(float(pv)):>11s} {stars}"
        )

    lines = [
        "=== Multiple Linear Regression (OLS) ===",
        f"Dependent  : {y_col}",
        f"Predictors : {', '.join(x_cols)}",
        f"N          = {n}  |  k = {k}",
        "",
        f"R²         = {res.rsquared:.4f}",
        f"Adj. R²    = {res.rsquared_adj:.4f}",
        f"F({k},{n-k-1}) = {res.fvalue:.4f}   {format_p(float(res.f_pvalue))}  "
        f"{significance_stars(float(res.f_pvalue))}",
        f"AIC        = {res.aic:.2f}  |  BIC = {res.bic:.2f}",
        f"Durbin-Watson = {dw:.3f}",
        "",
        hdr,
        sep,
        *param_rows,
        "",
        "Significance: *** p<0.001  ** p<0.01  * p<0.05",
    ]

    return AnalysisResult(
        name=f"MLR: {y_col} ~ {' + '.join(x_cols)}",
        method="multiple-regression",
        source_table=table.name,
        summary_text="\n".join(lines),
        details={
            "r2":        float(res.rsquared),
            "adj_r2":    float(res.rsquared_adj),
            "f":         float(res.fvalue),
            "p":         float(res.f_pvalue),
            "coefs":     {n: float(c) for n, c in zip(col_names, res.params)},
            "pvalues":   {n: float(p) for n, p in zip(col_names, res.pvalues)},
            "se":        {n: float(s) for n, s in zip(col_names, res.bse)},
            "fitted":    res.fittedvalues.tolist(),
            "residuals": res.resid.tolist(),
            "y_col":     y_col,
            "x_cols":    x_cols,
            "n":         n,
        },
    )
