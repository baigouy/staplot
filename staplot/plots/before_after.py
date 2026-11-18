"""Before/after (paired) plot — individual trajectories + mean±error overlay."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure

from ..models.project import DataTable
from ..stats.descriptive import describe
from . import styles


@styles.styled
def plot_before_after(
    table: DataTable,
    show_mean: bool = True,
    error_type: str = "sd",
    connect_lines: bool = True,
    line_alpha: float = 0.35,
    title: str = "",
    ylabel: str = "",
    colors: list | None = None,
) -> matplotlib.figure.Figure:
    cols = table.columns
    names = [c.name for c in cols]
    if not names:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, ax = plt.subplots(figsize=(max(3.5, len(names) * 0.9), 4))
    x_pos = list(range(len(names)))

    # Build index-aware data: use the intersection of valid row indices across all columns
    per_col_data: list[tuple[np.ndarray, list[int]]] = [
        c.valid_values_with_indices() for c in cols
    ]
    # Common subject rows = rows present in ALL columns
    common_idx: list[int] = sorted(
        set(per_col_data[0][1]).intersection(*[set(d[1]) for d in per_col_data[1:]])
    ) if len(per_col_data) > 1 else list(per_col_data[0][1]) if per_col_data else []

    # --- individual connecting lines ---
    if connect_lines and len(names) >= 2 and common_idx:
        rng = np.random.default_rng(99)
        col_val_maps = [
            {ri: rv for ri, rv in zip(d[1], d[0])} for d in per_col_data
        ]
        for orig_row in common_idx:
            y_vals = [cvm[orig_row] for cvm in col_val_maps]
            jitter = rng.uniform(-0.04, 0.04, size=len(y_vals))
            ax.plot(
                [xi + ji for xi, ji in zip(x_pos, jitter)],
                y_vals,
                color="#aaaaaa",
                alpha=line_alpha,
                linewidth=0.9,
                zorder=2,
            )

    # --- scatter points ---
    for i, (col, (vals, row_idxs)) in enumerate(zip(cols, per_col_data)):
        color = (colors[i] if colors and i < len(colors) else styles.color(i))
        rng = np.random.default_rng(i)
        jitter = rng.uniform(-0.07, 0.07, size=len(vals))
        sc = ax.scatter(
            x_pos[i] + jitter,
            vals,
            color=color,
            s=28,
            zorder=4,
            alpha=0.8,
            linewidths=0.4,
            edgecolors="white",
            picker=5,
        )
        # exclude_all=True so clicking removes the subject from all time-points
        sc._staplot_meta = {"column": col.name, "rows": row_idxs, "exclude_all": True}

    # --- mean ± error overlay ---
    if show_mean:
        for i, (col, (vals, _)) in enumerate(zip(cols, per_col_data)):
            d = describe(vals)
            if d["n"] == 0:
                continue
            color = (colors[i] if colors and i < len(colors) else styles.color(i))
            err = d.get(error_type, d["sem"])
            if np.isnan(err):
                err = 0.0
            ax.errorbar(
                x_pos[i],
                d["mean"],
                yerr=err,
                fmt="D",
                color=color,
                markersize=9,
                capsize=6,
                linewidth=2.0,
                zorder=5,
                markeredgecolor="#222222",
                markeredgewidth=0.6,
            )

    ax.set_xticks(x_pos)
    rotation = 20 if max(len(n) for n in names) > 8 else 0
    # tilted labels end at their tick; level ones are centred on it
    ax.set_xticklabels(names, rotation=rotation, ha="right" if rotation else "center")
    ax.set_ylabel(ylabel or "Value")
    ax.set_title(title or table.name)
    fig.tight_layout()
    return fig
