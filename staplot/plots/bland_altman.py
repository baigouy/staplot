"""Bland-Altman agreement plot."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure

from ..models.project import DataTable
from . import styles


@styles.styled
def plot_bland_altman(
    table: DataTable,
    col_a: str,
    col_b: str,
    title: str = "",
    color: str | None = None,
    percentage: bool = False,
) -> matplotlib.figure.Figure:
    """
    Bland-Altman plot comparing two measurement methods.
    percentage=True: plot % difference instead of absolute difference.
    """
    groups = table.group_arrays()
    if col_a not in groups or col_b not in groups:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Columns not found", ha="center", va="center", transform=ax.transAxes)
        return fig

    # both methods measured the same sample: pair by row
    a, b = table.paired_arrays(col_a, col_b)
    if len(a) < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Need at least 2 complete pairs", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    mean_vals = (a + b) / 2.0
    if percentage:
        diff = 100.0 * (a - b) / mean_vals
        ylabel = "Difference  (%)"
    else:
        diff = a - b
        ylabel = f"Difference  ({col_a} − {col_b})"

    mean_diff = float(np.mean(diff))
    std_diff  = float(np.std(diff, ddof=1))
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    c = color or styles.color(0)
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.scatter(mean_vals, diff, color=c, s=28, alpha=0.75, zorder=3,
               linewidths=0.3, edgecolors="white")

    x_lim = (mean_vals.min() - 0.05 * np.ptp(mean_vals),
             mean_vals.max() + 0.05 * np.ptp(mean_vals))
    ax.set_xlim(*x_lim)

    for y, ls, lbl in [
        (mean_diff,  "-",  f"Mean: {mean_diff:.3f}"),
        (loa_upper,  "--", f"+1.96 SD: {loa_upper:.3f}"),
        (loa_lower,  "--", f"−1.96 SD: {loa_lower:.3f}"),
    ]:
        ax.axhline(y, color="#333333", linewidth=1.2, linestyle=ls)
        ax.text(x_lim[1], y, f" {lbl}", va="center", fontsize=7, color="#333333")

    ax.set_xlabel(f"Mean of {col_a} and {col_b}")
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"Bland-Altman: {col_a} vs {col_b}")
    fig.tight_layout()
    return fig
