import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure

from ..models.project import DataTable
from . import styles
from .significance import add_significance_brackets

VIOLIN_CUT = 2.0        # bandwidths drawn past the data (seaborn's default)
VIOLIN_HALF_WIDTH = 0.4


def _violin(ax, x, arr, color):
    """One violin at *x*: a Gaussian density (Scott's bandwidth) that tapers
    VIOLIN_CUT bandwidths past the smallest / largest value, and inside it a thin
    line over the range, a thick one over the quartiles and a white median dot."""
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    top = float(arr.max())
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    spread = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if arr.size > 1 and spread > 0:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(arr, bw_method="scott")
        bw = kde.factor * spread
        y = np.linspace(arr.min() - VIOLIN_CUT * bw, arr.max() + VIOLIN_CUT * bw, 200)
        density = kde(y)
        half = density / density.max() * VIOLIN_HALF_WIDTH
        ax.fill_betweenx(y, x - half, x + half, facecolor=color, alpha=0.7,
                         edgecolor="#333333", linewidth=0.8, zorder=2)
        top = float(y[-1])
    ax.vlines(x, arr.min(), arr.max(), color="#333333", linewidth=1.0, zorder=3)
    ax.vlines(x, q1, q3, color="#333333", linewidth=4.0, zorder=3)
    ax.scatter([x], [med], s=14, color="white", edgecolors="#333333", linewidths=0.6, zorder=4)
    return top                  # the brackets go above the tapered tip, not the data


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
        # matplotlib's violinplot stops the density at the smallest and largest value:
        # flat-cut violins (user: "why are the violins so truncated at top and
        # bottom"). The density tapers 2 bandwidths past the data, as seaborn and
        # Prism draw it, with a small box inside (range, quartiles, median).
        violin_tops = {}
        for i, arr in enumerate(arrays):
            violin_tops[names[i]] = _violin(ax, i, np.asarray(arr, float), palette[i])
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
            if plot_type == "violin" and violin_tops.get(name) is not None:
                box_tops[name] = max(box_tops[name], violin_tops[name])
        add_significance_brackets(ax, names, sig_pairs, box_tops)

    fig.tight_layout()
    return fig
