import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure

from ..models.project import DataTable
from . import styles
from .significance import add_significance_brackets


@styles.styled
def plot_box(
    table: DataTable,
    plot_type: str = "box",
    show_points: bool = True,
    title: str = "",
    ylabel: str = "",
    colors: list | None = None,
    sig_pairs: list | None = None,
) -> matplotlib.figure.Figure:
    cols = [c for c in table.columns if len(c.valid_values()) > 0]
    names = [c.name for c in cols]
    arrays = [c.valid_values() for c in cols]

    if not names:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, ax = plt.subplots(figsize=(max(3.5, len(names) * 0.9), 4))
    palette = [
        (colors[i] if colors and i < len(colors) else styles.color(i))
        for i in range(len(names))
    ]

    if plot_type == "violin":
        parts = ax.violinplot(arrays, positions=range(len(names)), showmedians=True,
                              showextrema=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(palette[i])
            pc.set_alpha(0.7)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            if key in parts:
                parts[key].set_color("#333333")
                parts[key].set_linewidth(1.2)
    else:
        bp = ax.boxplot(
            arrays,
            # same x as points, ticks and brackets (0..n-1); boxplot's default
            # 1..n drew every box one group to the right of its data
            positions=range(len(names)),
            patch_artist=True,
            medianprops=dict(color="#333333", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", markersize=4, alpha=0.5),
            widths=0.5,
        )
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

    if show_points:
        rng = np.random.default_rng(42)
        for i, col in enumerate(cols):
            vals, row_idxs = col.valid_values_with_indices()
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            sc = ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                color="#222222",
                s=18,
                zorder=5,
                alpha=0.65,
                linewidths=0.3,
                edgecolors="white",
                picker=5,
            )
            sc._staplot_meta = {"column": col.name, "rows": row_idxs, "exclude_all": False}

    ax.set_xticks(range(len(names)))
    rotation = 20 if max(len(n) for n in names) > 8 else 0
    # tilted labels end at their tick; level ones are centred on it
    ax.set_xticklabels(names, rotation=rotation, ha="right" if rotation else "center")
    ax.set_ylabel(ylabel or "Value")
    ax.set_title(title or table.name)

    if sig_pairs:
        # Whisker tops as bracket heights
        box_tops: dict[str, float] = {}
        for name, arr in zip(names, arrays):
            box_tops[name] = float(np.max(arr)) if len(arr) > 0 else 0.0
        add_significance_brackets(ax, names, sig_pairs, box_tops)

    fig.tight_layout()
    return fig
