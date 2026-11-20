"""Time course / dose response: mean ± error of each group across an x factor (day,
dose, time), one line per group, each with its own colour and marker."""

import numpy as np
import matplotlib.figure

from ..models.project import GroupedDataTable
from ..stats.descriptive import describe
from . import styles


def _as_number(v):
    try:
        return float(str(v).strip().split()[-1]) if str(v).strip() else None
    except ValueError:
        return None


@styles.styled
def plot_lines(
    table: GroupedDataTable,
    error_type: str = "sem",
    show_points: bool = False,
    title: str = "",
    ylabel: str = "",
    xlabel: str = "",
    colors: list | None = None,
    swap: bool = False,
) -> matplotlib.figure.Figure:
    """One line per group (factor A, the colours) across factor B (x). x is numeric when
    every level reads as a number ("7", "day 14", "10 uM" -> 7, 14, 10), else the levels
    in order. *swap*: factor B as the lines, factor A on x."""
    groups, xs = (list(table.row_labels), list(table.col_groups)) if swap else \
        (list(table.col_groups), list(table.row_labels))
    fig, ax = styles.subplots(figsize=(4.2, 3.4))
    nums = [_as_number(x) for x in xs]
    numeric = all(n is not None for n in nums)
    order = sorted(range(len(xs)), key=lambda i: nums[i]) if numeric else list(range(len(xs)))
    xpos = [nums[i] if numeric else k for k, i in enumerate(order)]
    rng = np.random.default_rng(3)
    for g_i, g in enumerate(groups):
        color = colors[g_i] if colors and g_i < len(colors) else styles.color(g_i)
        means, errs, used = [], [], []
        for k, i in enumerate(order):
            x = xs[i]
            cell = table.data.get(x if swap else g, {}).get(g if swap else x, [])
            arr = np.asarray([v for v in cell if v is not None and np.isfinite(float(v))], float)
            if not arr.size:
                continue
            d = describe(arr)
            err = d.get(error_type, d["sem"]) if error_type != "none" else 0.0
            means.append(d["mean"])
            errs.append(0.0 if err is None or not np.isfinite(err) else err)
            used.append(xpos[k])
            if show_points:
                span = (max(xpos) - min(xpos)) if len(xpos) > 1 else 1.0
                pts = ax.scatter(np.full(arr.size, xpos[k]) + rng.uniform(-0.015, 0.015, arr.size) * span, arr,
                                 color=color, s=8, alpha=0.4, linewidths=0, zorder=2, gid="points")
        eb = ax.errorbar(used, means, yerr=errs if error_type != "none" else None, color=color,
                         marker=styles.marker(g_i), markersize=5, linewidth=1.4, capsize=3, label=str(g),
                         zorder=3, markeredgecolor="white", markeredgewidth=0.5)
        for part in [*eb.lines[1], *eb.lines[2]]:
            part.set_gid("error_bars")           # one-sided error bars find them
    if not numeric or len(xpos) <= 12:
        # the levels themselves on the axis ("day 1", "day 2"), at their place
        ax.set_xticks(xpos)
        ax.set_xticklabels([str(xs[i]) for i in order])
    ax.set_ylabel(ylabel or "Value")
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_title(title or table.name)
    err_label = {"sem": "mean ± SEM", "sd": "mean ± SD", "ci95": "mean ± 95% CI"}.get(error_type, "mean")
    ax.legend(frameon=False, fontsize=8, title=err_label, title_fontsize=7)
    if styles.target_axes() is None:
        fig.tight_layout()
    return fig
