"""Distribution fitting, shape analysis, and test recommendation."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ..models.project import DataTable, AnalysisResult
from .descriptive import format_p


# ── Available distributions ──────────────────────────────────────────

_DISTS: dict[str, stats.rv_continuous] = {
    "Normal":      stats.norm,
    "Log-normal":  stats.lognorm,
    "Exponential": stats.expon,
    "Gamma":       stats.gamma,
    "Weibull":     stats.weibull_min,
}

# distributions that require strictly positive data
_POSITIVE_ONLY = {"Log-normal", "Exponential", "Gamma", "Weibull"}


# ── Shape description ────────────────────────────────────────────────

def describe_shape(arr: np.ndarray) -> dict:
    """Return skewness, kurtosis (excess), and human-readable labels."""
    skew = float(stats.skew(arr))
    kurt = float(stats.kurtosis(arr))   # excess kurtosis (Normal = 0)

    if abs(skew) < 0.5:
        skew_label = "symmetric"
    elif skew > 0.5:
        skew_label = "right-skewed (+)"
    else:
        skew_label = "left-skewed (-)"

    if abs(kurt) < 1.0:
        kurt_label = "mesokurtic (normal-like tails)"
    elif kurt > 1.0:
        kurt_label = "leptokurtic (heavy tails)"
    else:
        kurt_label = "platykurtic (light tails)"

    return {
        "skewness":       skew,
        "kurtosis":       kurt,
        "skewness_label": skew_label,
        "kurtosis_label": kurt_label,
    }


# ── Distribution fitting ─────────────────────────────────────────────

def fit_distributions(arr: np.ndarray) -> list[dict]:
    """
    Fit each distribution in _DISTS to *arr* using MLE.
    Returns a list of result dicts sorted by AIC (best first).
    """
    n = len(arr)
    results: list[dict] = []

    for name, dist in _DISTS.items():
        if name in _POSITIVE_ONLY and np.any(arr <= 0):
            continue
        try:
            params   = dist.fit(arr)
            log_like = float(np.sum(dist.logpdf(arr, *params)))
            k        = len(params)
            aic      = 2 * k - 2 * log_like
            bic      = k * np.log(n) - 2 * log_like
            # KS goodness-of-fit against the fitted distribution
            ks_stat, ks_p = stats.kstest(arr, dist.cdf, args=params)
            results.append({
                "distribution": name,
                "params":        params,
                "log_likelihood": log_like,
                "aic":           aic,
                "bic":           bic,
                "ks_stat":       float(ks_stat),
                "ks_p":          float(ks_p),
            })
        except Exception:
            pass

    results.sort(key=lambda x: x["aic"])
    return results


# ── Test recommendation ───────────────────────────────────────────────

def _recommend(is_normal: bool) -> dict:
    if is_normal:
        return {
            "2_groups":      "t-test (paired or unpaired)",
            "3plus_groups":  "One-Way ANOVA  (+Tukey post-hoc)",
            "correlation":   "Pearson r",
            "rm":            "Repeated-Measures ANOVA",
            "parametric":    True,
        }
    else:
        return {
            "2_groups":      "Mann-Whitney U  /  Wilcoxon signed-rank",
            "3plus_groups":  "Kruskal-Wallis  (+Dunn post-hoc)",
            "correlation":   "Spearman rho  /  Kendall tau",
            "rm":            "Friedman test",
            "parametric":    False,
        }


# ── Main entry point ─────────────────────────────────────────────────

def run_distribution_check(table: DataTable) -> AnalysisResult:
    """
    Full per-column distribution analysis:
      - Normality tests (reuses run_normality details)
      - Shape stats (skewness, kurtosis)
      - Distribution fitting (top 5 distributions by AIC)
      - Test recommendation
    """
    from .normality import run_normality

    norm_result = run_normality(table)

    lines  = ["=== Distribution Check ===", ""]
    all_details: dict = {}

    for col in table.columns:
        arr = col.valid_values()
        n   = len(arr)
        lines.append(f"{'─'*52}")
        lines.append(f"  Column : {col.name}  (n={n})")
        lines.append(f"{'─'*52}")

        if n < 3:
            lines.append("  Insufficient data (need n >= 3).")
            lines.append("")
            all_details[col.name] = {"n": n, "error": "insufficient data"}
            continue

        # ── Shape stats ───────────────────────────────────────────
        shape   = describe_shape(arr)
        skew    = shape["skewness"]
        kurt    = shape["kurtosis"]
        lines.append(f"  Skewness        : {skew:+.3f}  ({shape['skewness_label']})")
        lines.append(f"  Excess kurtosis : {kurt:+.3f}  ({shape['kurtosis_label']})")
        lines.append("")

        # ── Normality tests ───────────────────────────────────────
        nd  = norm_result.details.get(col.name, {})
        sw_p  = nd.get("shapiro_p",    float("nan"))
        dap_p = nd.get("dagostino_p",  float("nan"))
        ks_p  = nd.get("ks_p",         float("nan"))

        lines.append("  Normality tests:")
        if not np.isnan(sw_p):
            lines.append(f"    Shapiro-Wilk        p = {format_p(sw_p)}")
        if not np.isnan(dap_p):
            lines.append(f"    D'Agostino-Pearson  p = {format_p(dap_p)}")
        if not np.isnan(ks_p):
            lines.append(f"    KS (vs Normal)      p = {format_p(ks_p)}")

        # Consensus: PASS only if Shapiro-Wilk (most sensitive) says OK
        is_normal = (not np.isnan(sw_p)) and (sw_p >= 0.05)
        verdict_str = "PASS (normal)" if is_normal else "FAIL (non-normal)"
        lines.append(f"  Verdict         : {verdict_str}")
        lines.append("")

        # ── Distribution fitting ──────────────────────────────────
        fits = fit_distributions(arr)
        if fits:
            lines.append("  Distribution fitting (ranked by AIC):")
            lines.append(f"    {'Distribution':<15s}  {'AIC':>8s}  {'KS p':>8s}  {'OK?'}")
            for fit in fits[:5]:
                ok = "yes" if fit["ks_p"] >= 0.05 else "no"
                lines.append(
                    f"    {fit['distribution']:<15s}  {fit['aic']:>8.1f}  {fit['ks_p']:>8.4f}  {ok}"
                )
            best_name = fits[0]["distribution"]
            lines.append(f"  Best-fit dist   : {best_name}  (AIC={fits[0]['aic']:.1f})")
            lines.append("")
        else:
            fits = []
            best_name = "n/a"

        # ── Test recommendation ───────────────────────────────────
        rec = _recommend(is_normal)
        lines.append("  Recommended tests:")
        lines.append(f"    2 groups  : {rec['2_groups']}")
        lines.append(f"    3+ groups : {rec['3plus_groups']}")
        lines.append(f"    Correlation: {rec['correlation']}")
        lines.append(f"    Rep. measures: {rec['rm']}")
        lines.append("")

        all_details[col.name] = {
            "n":             n,
            "skewness":      skew,
            "kurtosis":      kurt,
            "is_normal":     is_normal,
            "shapiro_p":     sw_p,
            "dagostino_p":   dap_p,
            "ks_p_norm":     ks_p,
            "fits":          [
                {k: (v.tolist() if isinstance(v, np.ndarray) else
                     float(v)  if isinstance(v, (np.floating, np.integer)) else v)
                 for k, v in f.items() if k != "params"}
                for f in fits[:5]
            ],
            "best_fit":      best_name,
            "rec_2groups":   rec["2_groups"],
            "rec_3groups":   rec["3plus_groups"],
            "parametric":    rec["parametric"],
        }

    return AnalysisResult(
        name=f"Distribution check: {table.name}",
        method="distribution-check",
        source_table=table.name,
        summary_text="\n".join(lines),
        details=all_details,
    )
