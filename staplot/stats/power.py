"""Power analysis and sample size calculation via statsmodels."""

import numpy as np
from statsmodels.stats.power import TTestIndPower, TTestPower, FTestAnovaPower

from ..models.project import AnalysisResult


def _effect_size_label(d: float) -> str:
    if abs(d) < 0.2:
        return "negligible"
    if abs(d) < 0.5:
        return "small"
    if abs(d) < 0.8:
        return "medium"
    return "large"


def run_power_ttest(
    effect_size: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.80,
    n: int | None = None,
    paired: bool = False,
) -> AnalysisResult:
    analysis = TTestPower() if paired else TTestIndPower()
    # TTestIndPower uses `nobs1`; TTestPower uses `nobs`
    _nobs_kw = "nobs" if paired else "nobs1"

    if n is None:
        required_n = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power)
        required_n = int(np.ceil(required_n))
        actual_power = analysis.power(effect_size=effect_size, alpha=alpha,
                                      **{_nobs_kw: required_n})
        mode_label = "Sample size calculation"
        lines = [
            f"=== Power Analysis: {'Paired' if paired else 'Unpaired'} t-test ===",
            f"Mode: {mode_label}",
            "",
            f"Effect size (Cohen's d) = {effect_size:.3f}  ({_effect_size_label(effect_size)})",
            f"alpha (significance)    = {alpha:.3f}",
            f"Target power           = {power:.2f}",
            "",
            f"Required n per group   = {required_n}",
            f"Achieved power         = {actual_power:.4f}",
        ]
        result_n = required_n
        result_power = actual_power
    else:
        actual_power = analysis.power(effect_size=effect_size, alpha=alpha,
                                      **{_nobs_kw: n})
        mode_label = "Power calculation"
        lines = [
            f"=== Power Analysis: {'Paired' if paired else 'Unpaired'} t-test ===",
            f"Mode: {mode_label}",
            "",
            f"Effect size (Cohen's d) = {effect_size:.3f}  ({_effect_size_label(effect_size)})",
            f"alpha (significance)    = {alpha:.3f}",
            f"n per group             = {n}",
            "",
            f"Achieved power = {actual_power:.4f}",
            f"{'-> ADEQUATE power (>= 0.80)' if actual_power >= 0.8 else '-> INSUFFICIENT power (< 0.80) -- consider increasing n'}",
        ]
        result_n = n
        result_power = actual_power

    # Effect size guide
    lines += [
        "",
        "Cohen's d reference: small=0.2, medium=0.5, large=0.8",
        "Note: effect size can be estimated from pilot data as |Δmean| / pooled_SD",
    ]

    return AnalysisResult(
        name=f"Power analysis (t-test)",
        method="power-ttest",
        source_table="",
        summary_text="\n".join(lines),
        details={"effect_size": effect_size, "alpha": alpha, "power": float(result_power), "n": result_n},
    )


def run_power_anova(
    effect_size: float = 0.25,
    alpha: float = 0.05,
    power: float = 0.80,
    k_groups: int = 3,
    n: int | None = None,
) -> AnalysisResult:
    analysis = FTestAnovaPower()

    if n is None:
        required_n = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, k_groups=k_groups)
        required_n = int(np.ceil(required_n))
        actual_power = analysis.power(effect_size=effect_size, nobs=required_n, alpha=alpha, k_groups=k_groups)
        lines = [
            "=== Power Analysis: One-Way ANOVA ===",
            "",
            f"Effect size (Cohen's f) = {effect_size:.3f}",
            f"Groups (k)              = {k_groups}",
            f"alpha                   = {alpha:.3f}",
            f"Target power            = {power:.2f}",
            "",
            f"Required n per group    = {required_n}",
            f"Achieved power          = {actual_power:.4f}",
            "",
            "Cohen's f reference: small=0.10, medium=0.25, large=0.40",
        ]
        result_n, result_power = required_n, actual_power
    else:
        actual_power = analysis.power(effect_size=effect_size, nobs=n, alpha=alpha, k_groups=k_groups)
        lines = [
            "=== Power Analysis: One-Way ANOVA ===",
            "",
            f"Effect size (Cohen's f) = {effect_size:.3f}",
            f"Groups (k)              = {k_groups}",
            f"alpha                   = {alpha:.3f}",
            f"n per group             = {n}",
            "",
            f"Achieved power = {actual_power:.4f}",
        ]
        result_n, result_power = n, actual_power

    return AnalysisResult(
        name="Power analysis (ANOVA)",
        method="power-anova",
        source_table="",
        summary_text="\n".join(lines),
        details={"effect_size": effect_size, "alpha": alpha, "power": float(result_power), "n": result_n, "k": k_groups},
    )
