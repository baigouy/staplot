"""Grouped bar chart for GroupedDataTable (Prism-style side-by-side bars)."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from scipy import stats as scipy_stats

from ..models.project import GroupedDataTable
from . import styles


@styles.styled
def plot_grouped_bar(
    table: GroupedDataTable,
    error_type: str = "sd",       # "sd" | "sem" | "ci95" | "none"
    show_points: bool = True,
    title: str = "",
    ylabel: str = "",
    colors: list | None = None,
    sig_pairs: list | None = None,
) -> matplotlib.figure.Figure:
    """
    Side-by-side bar chart for a GroupedDataTable.
    Factor A (col_groups) → bar colour / legend entry.
    Factor B (row_labels) → X-axis position.
    """

    if not table.col_groups or not table.row_labels:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    n_cg   = len(table.col_groups)   # number of colour groups
    n_rows = len(table.row_labels)   # number of X positions

    bar_w  = 0.8 / n_cg
    x_base = np.arange(n_rows)

    fig_w = max(5, n_rows * n_cg * 0.55 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, 4))

    rng = np.random.default_rng(42)

    for i, cg in enumerate(table.col_groups):
        c = (colors[i] if colors and i < len(colors) else styles.color(i))
        means, errs, point_sets = [], [], []

        for rl in table.row_labels:
            raw = [v for v in table.data.get(cg, {}).get(rl, [])
                   if v is not None and not np.isnan(float(v))]
            arr = np.array(raw, dtype=float)

            if len(arr) == 0:
                means.append(np.nan)
                errs.append(0.0)
            else:
                m = float(np.mean(arr))
                means.append(m)
                if len(arr) > 1:
                    sd  = float(np.std(arr, ddof=1))
                    sem = sd / np.sqrt(len(arr))
                    if error_type == "sd":
                        errs.append(sd)
                    elif error_type == "ci95":
                        errs.append(float(scipy_stats.t.ppf(0.975, len(arr) - 1)) * sem)
                    elif error_type == "none":
                        errs.append(0.0)
                    else:
                        errs.append(sem)
                else:
                    errs.append(0.0)
            point_sets.append(arr)

        offset  = (i - n_cg / 2.0 + 0.5) * bar_w
        bx      = x_base + offset

        # bars
        ax.bar(bx, means, width=bar_w * 0.88,
               color=c, alpha=0.85, label=cg,
               zorder=3, edgecolor="white", linewidth=0.5)

        # error bars
        valid = [(bx[j], means[j], errs[j])
                 for j in range(n_rows) if not np.isnan(means[j]) and errs[j] > 0]
        if valid:
            vx, vy, ve = zip(*valid)
            ax.errorbar(vx, vy, yerr=ve, fmt="none",
                        color="#333333", linewidth=1.2, capsize=3, zorder=4)

        # jittered data points
        if show_points:
            for bx_j, arr in zip(bx, point_sets):
                if len(arr) == 0:
                    continue
                jitter = rng.uniform(-bar_w * 0.18, bar_w * 0.18, len(arr))
                ax.scatter(bx_j + jitter, arr,
                           color="white", edgecolors=c,
                           s=22, alpha=0.9, linewidths=0.8, zorder=5)

    ax.set_xticks(x_base)
    ax.set_xticklabels(
        table.row_labels,
        rotation=0 if n_rows <= 6 else 30,
        ha="right" if n_rows > 6 else "center",
    )
    ax.set_ylabel(ylabel or "Value")
    ax.set_title(title or table.name)
    # beside the axes: inside, "upper right" covered the tallest bars
    ax.legend(title="Group", fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    ax.set_xlim(-0.5, n_rows - 0.5)
    fig.tight_layout()
    return fig
