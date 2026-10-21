import numpy as np
from scipy import stats

from ..models.project import DataTable, AnalysisResult
from .descriptive import describe, format_p, significance_stars


def _two_group_header(ga: str, da: dict, gb: str, db: dict, metric: str = "mean") -> list[str]:
    return [
        f"{'':30s} {ga:>15s} {gb:>15s}",
        f"{'N':30s} {da['n']:>15d} {db['n']:>15d}",
        f"{'Mean':30s} {da['mean']:>15.4f} {db['mean']:>15.4f}",
        f"{'SD':30s} {da['sd']:>15.4f} {db['sd']:>15.4f}",
        f"{'SEM':30s} {da['sem']:>15.4f} {db['sem']:>15.4f}",
        f"{'Median':30s} {da['median']:>15.4f} {db['median']:>15.4f}",
    ]


def run_ttest(
    table: DataTable,
    group_a: str,
    group_b: str,
    paired: bool = False,
    alternative: str = "two-sided",
) -> AnalysisResult:
    if paired:
        # complete subjects only: both measurements present in the same row
        a, b = table.paired_arrays(group_a, group_b)
    else:
        groups = table.group_arrays()
        a, b = groups[group_a], groups[group_b]
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Each group needs at least 2 values"
                         + (" (complete pairs)." if paired else "."))

    if paired:
        t_stat, p = stats.ttest_rel(a, b, alternative=alternative)
        test_name = "Paired t-test"
    else:
        t_stat, p = stats.ttest_ind(a, b, equal_var=False, alternative=alternative)
        test_name = "Unpaired t-test (Welch)"

    da, db = describe(a), describe(b)
    lines = [
        f"=== {test_name} ===",
        f"Groups: {group_a}  vs  {group_b}",
        f"Alternative: {alternative}",
        "",
        *_two_group_header(group_a, da, group_b, db),
        "",
        f"t statistic: {t_stat:.4f}",
        f"{format_p(p)}  {significance_stars(p)}",
    ]
    return AnalysisResult(
        name=f"{test_name}: {group_a} vs {group_b}",
        method="t-test",
        source_table=table.name,
        summary_text="\n".join(lines),
        details={"t": float(t_stat), "p": float(p), "group_a": group_a, "group_b": group_b},
    )


def run_mann_whitney(
    table: DataTable,
    group_a: str,
    group_b: str,
    alternative: str = "two-sided",
) -> AnalysisResult:
    groups = table.group_arrays()
    a, b = groups[group_a], groups[group_b]
    if len(a) < 1 or len(b) < 1:
        raise ValueError("Each group needs at least 1 value.")

    u_stat, p = stats.mannwhitneyu(a, b, alternative=alternative)
    da, db = describe(a), describe(b)
    lines = [
        "=== Mann-Whitney U Test ===",
        f"Groups: {group_a}  vs  {group_b}",
        f"Alternative: {alternative}",
        "",
        *_two_group_header(group_a, da, group_b, db),
        "",
        f"U statistic: {u_stat:.0f}",
        f"{format_p(p)}  {significance_stars(p)}",
    ]
    return AnalysisResult(
        name=f"Mann-Whitney: {group_a} vs {group_b}",
        method="mann-whitney",
        source_table=table.name,
        summary_text="\n".join(lines),
        details={"U": float(u_stat), "p": float(p), "group_a": group_a, "group_b": group_b},
    )


def run_wilcoxon(
    table: DataTable,
    group_a: str,
    group_b: str,
    alternative: str = "two-sided",
) -> AnalysisResult:
    a, b = table.paired_arrays(group_a, group_b)
    if len(a) < 1:
        raise ValueError("Wilcoxon signed-rank needs complete pairs (same row in both columns).")

    w_stat, p = stats.wilcoxon(a, b, alternative=alternative)
    lines = [
        "=== Wilcoxon Signed-Rank Test ===",
        f"Groups: {group_a}  vs  {group_b}",
        f"Alternative: {alternative}",
        "",
        f"W statistic: {w_stat:.4f}",
        f"{format_p(p)}  {significance_stars(p)}",
    ]
    return AnalysisResult(
        name=f"Wilcoxon: {group_a} vs {group_b}",
        method="wilcoxon",
        source_table=table.name,
        summary_text="\n".join(lines),
        details={"W": float(w_stat), "p": float(p)},
    )


def run_descriptive(table: DataTable) -> AnalysisResult:
    header = f"{'Group':20s} {'N':>6s} {'Mean':>12s} {'SD':>12s} {'SEM':>12s} {'Median':>12s} {'Min':>10s} {'Max':>10s}"
    sep = "-" * len(header)
    rows = []
    for col in table.columns:
        d = describe(col.valid_values())
        if d["n"] == 0:
            continue
        rows.append(
            f"{col.name:20s} {d['n']:>6d} {d['mean']:>12.4f} {d['sd']:>12.4f} "
            f"{d['sem']:>12.4f} {d['median']:>12.4f} {d['min']:>10.4f} {d['max']:>10.4f}"
        )
    lines = [f"=== Descriptive Statistics: {table.name} ===", "", header, sep, *rows]
    return AnalysisResult(
        name=f"Descriptive: {table.name}",
        method="descriptive",
        source_table=table.name,
        summary_text="\n".join(lines),
        details={},
    )
