import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure

from ..models.project import DataTable
from ..stats.descriptive import describe
from . import styles
from .significance import add_significance_brackets


@styles.styled
def plot_bar(
    table: DataTable,
    error_type: str = "sd",            # SD: what journals ask for (SEM hides the spread)
    show_points: bool = True,
    title: str = "",
    ylabel: str = "",
    colors: list | None = None,
    sig_pairs: list | None = None,
) -> matplotlib.figure.Figure:
    """sig_pairs: list of (name_a, name_b, p_value) to annotate."""
    cols = table.columns
    names = [c.name for c in cols]
    if not names:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, ax = plt.subplots(figsize=(max(3.5, len(names) * 0.9), 4))
    x_pos = np.arange(len(names))
    bar_width = 0.6

    for i, col in enumerate(cols):
        arr = col.valid_values()
        d = describe(arr)
        if d["n"] == 0:
            continue
        color = (colors[i] if colors and i < len(colors) else styles.color(i))
        err = d.get(error_type, d["sem"])
        if np.isnan(err):
            err = 0.0

        ax.bar(
            x_pos[i],
            d["mean"],
            width=bar_width,
            color=color,
            alpha=0.85,
            zorder=2,
            linewidth=0.8,
            edgecolor="#444444",
        )
        ax.errorbar(
            x_pos[i],
            d["mean"],
            yerr=err,
            fmt="none",
            color="#222222",
            linewidth=1.2,
            capsize=5,
            zorder=3,
        )

        if show_points and d["n"] > 0:
            vals, row_idxs = col.valid_values_with_indices()
            jitter = np.random.default_rng(i).uniform(-0.12, 0.12, size=len(vals))
            sc = ax.scatter(
                x_pos[i] + jitter,
                vals,
                color="#222222",
                s=25,
                zorder=4,
                alpha=0.7,
                linewidths=0.4,
                edgecolors="white",
                picker=5,
            )
            sc._staplot_meta = {"column": col.name, "rows": row_idxs, "exclude_all": False}

    err_label = {"sem": "± SEM", "sd": "± SD", "ci95": "± 95% CI"}.get(error_type, "")
    ax.set_xticks(x_pos)
    rotation = 20 if max(len(n) for n in names) > 8 else 0
    # tilted labels end at their tick; level ones are centred on it
    ax.set_xticklabels(names, rotation=rotation, ha="right" if rotation else "center")
    ax.set_ylabel(ylabel or "Value")
    ax.set_title(title or table.name)
    if err_label:
        # as the (otherwise unused) x-axis label: matplotlib places it below the
        # tick labels, so it no longer collides with rotated / long group names
        # (a fixed ax.text at y=-0.12 did), and tight bounding boxes include it
        ax.set_xlabel(f"Bars: mean {err_label}", fontsize=7, color="#666666",
                      loc="right")

    if sig_pairs:
        bar_tops: dict[str, float] = {}
        for col in cols:
            d = describe(col.valid_values())
            if d["n"] > 0:
                err = d.get(error_type, d["sem"])
                bar_tops[col.name] = d["mean"] + (err if not np.isnan(err) else 0)
        add_significance_brackets(ax, names, sig_pairs, bar_tops)

    fig.tight_layout()
    return fig
