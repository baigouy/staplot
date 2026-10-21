import itertools

import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

from ..models.project import DataTable, AnalysisResult
from .descriptive import describe, format_p, significance_stars


def dunn_test(arrays) -> list[tuple[int, int, float]]:
    """Dunn's test (1964) after Kruskal-Wallis, Bonferroni-corrected.

    Compares mean ranks of the POOLED sample (the ranks Kruskal-Wallis tested),
    with the tie-corrected variance
        z_ij = (Rbar_i - Rbar_j) / sqrt((N(N+1)/12 - T/(12(N-1))) * (1/n_i + 1/n_j)),
    T = sum(t^3 - t) over tie groups; two-sided normal p-values x number of pairs.
    Returns [(i, j, p_adj), ...] for every pair of group indices."""
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    sizes = [len(a) for a in arrays]
    pooled = np.concatenate(arrays)
    n_tot = len(pooled)
    ranks = stats.rankdata(pooled)
    mean_ranks = []
    start = 0
    for n in sizes:
        mean_ranks.append(float(ranks[start:start + n].mean()))
        start += n
    _, tie_counts = np.unique(pooled, return_counts=True)
    ties = float(np.sum(tie_counts.astype(float) ** 3 - tie_counts))
    base_var = n_tot * (n_tot + 1) / 12.0 - ties / (12.0 * (n_tot - 1))

    pairs = list(itertools.combinations(range(len(arrays)), 2))
    m = len(pairs)
    out = []
    for i, j in pairs:
        se = np.sqrt(base_var * (1.0 / sizes[i] + 1.0 / sizes[j]))
        if not np.isfinite(se) or se <= 0:
            p = float("nan")
        else:
            z = abs(mean_ranks[i] - mean_ranks[j]) / se
            p = min(1.0, 2.0 * float(stats.norm.sf(z)) * m)
        out.append((i, j, p))
    return out


def run_one_way_anova(table: DataTable, posthoc: str = "tukey") -> AnalysisResult:
    groups = table.group_arrays()
    names = list(groups.keys())
    arrays = [groups[n] for n in names]

    if len(names) < 2:
        raise ValueError("Need at least 2 groups for ANOVA.")
    if any(len(a) < 2 for a in arrays):
        raise ValueError("Each group needs at least 2 values.")

    F, p = stats.f_oneway(*arrays)

    desc_header = f"{'Group':20s} {'N':>6s} {'Mean':>12s} {'SD':>12s}"
    desc_rows = []
    for n, a in zip(names, arrays):
        d = describe(a)
        desc_rows.append(f"{n:20s} {d['n']:>6d} {d['mean']:>12.4f} {d['sd']:>12.4f}")

    ph_lines: list[str] = []
    if posthoc == "tukey" and len(names) > 2:
        all_vals = np.concatenate(arrays)
        all_labels = np.concatenate([[n] * len(a) for n, a in zip(names, arrays)])
        tukey = pairwise_tukeyhsd(all_vals, all_labels)
        ph_lines = ["", "Post-hoc (Tukey HSD):", str(tukey.summary())]

    lines = [
        "=== One-Way ANOVA ===",
        "",
        desc_header,
        "-" * len(desc_header),
        *desc_rows,
        "",
        f"F statistic: {F:.4f}",
        f"{format_p(p)}  {significance_stars(p)}",
        *ph_lines,
    ]
    return AnalysisResult(
        name=f"One-Way ANOVA ({', '.join(names)})",
        method="one-way-anova",
        source_table=table.name,
        summary_text="\n".join(lines),
        details={"F": float(F), "p": float(p), "groups": names},
    )


def run_kruskal_wallis(table: DataTable) -> AnalysisResult:
    groups = table.group_arrays()
    names = list(groups.keys())
    arrays = [groups[n] for n in names]

    if len(names) < 2:
        raise ValueError("Need at least 2 groups.")

    H, p = stats.kruskal(*arrays)

    ph_lines: list[str] = []
    if len(names) > 2:
        ph_lines = ["", "Post-hoc (Dunn's test / Bonferroni):"]
        ph_lines.append(f"{'Comparison':35s} {'adj. p':>10s} {'sig':>6s}")
        for i, j, ap in dunn_test(arrays):
            label = f"{names[i]} vs {names[j]}"
            ph_lines.append(f"{label:35s} {ap:>10.4f} {significance_stars(ap):>6s}")

    lines = [
        "=== Kruskal-Wallis Test ===",
        "",
        f"H statistic: {H:.4f}",
        f"{format_p(p)}  {significance_stars(p)}",
        *ph_lines,
    ]
    return AnalysisResult(
        name=f"Kruskal-Wallis ({', '.join(names)})",
        method="kruskal-wallis",
        source_table=table.name,
        summary_text="\n".join(lines),
        details={"H": float(H), "p": float(p), "groups": names},
    )
