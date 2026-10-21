"""Frequency histogram with optional KDE overlay."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from scipy.stats import gaussian_kde

from ..models.project import DataTable
from . import styles


@styles.styled
def plot_histogram(
    table: DataTable,
    bins: int | str = "auto",
    density: bool = False,
    show_kde: bool = True,
    cumulative: bool = False,
    title: str = "",
    xlabel: str = "Value",
    ylabel: str = "",
    colors: list | None = None,
) -> matplotlib.figure.Figure:
    groups = table.group_arrays()
    names = [n for n, a in groups.items() if len(a) > 0]
    if not names:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, ax = plt.subplots(figsize=(5, 4))

    for i, name in enumerate(names):
        arr = groups[name]
        color = (colors[i] if colors and i < len(colors) else styles.color(i))

        alpha = 0.55 if len(names) > 1 else 0.75
        ax.hist(
            arr,
            bins=bins,
            density=density,
            cumulative=cumulative,
            alpha=alpha,
            color=color,
            label=name,
            edgecolor="white",
            linewidth=0.5,
        )

        if show_kde and not cumulative and len(arr) >= 4:
            try:
                kde = gaussian_kde(arr, bw_method="scott")
                x_line = np.linspace(arr.min() - arr.std(), arr.max() + arr.std(), 300)
                kde_y = kde(x_line)
                if not density:
                    # scale KDE to match histogram counts
                    bin_count = bins if isinstance(bins, int) else int(np.sqrt(len(arr)))
                    bin_w = (arr.max() - arr.min()) / max(bin_count, 1)
                    kde_y = kde_y * len(arr) * bin_w
                ax.plot(x_line, kde_y, color=color, linewidth=2.0, alpha=0.9)
            except Exception:
                pass

    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    elif cumulative:
        ax.set_ylabel("Cumulative frequency")
    else:
        ax.set_ylabel("Density" if density else "Count")

    ax.set_title(title or table.name)
    if len(names) > 1:
        ax.legend()
    fig.tight_layout()
    return fig
