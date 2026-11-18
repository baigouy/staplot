import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from scipy import stats

from ..models.project import DataTable
from ..stats.descriptive import format_p
from . import styles


@styles.styled
def plot_scatter(
    table: DataTable,
    x_col: str | None = None,
    y_cols: list[str] | None = None,
    show_regression: bool = True,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    colors: list | None = None,
) -> matplotlib.figure.Figure:
    groups = table.group_arrays()
    col_names = list(groups.keys())

    if not col_names:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    x_name = x_col or col_names[0]
    y_names = y_cols or col_names[1:] if len(col_names) > 1 else col_names

    fig, ax = plt.subplots(figsize=(5, 4))

    for i, y_name in enumerate(y_names):
        if x_name not in groups or y_name not in groups:
            continue
        # each point is one row: drop rows where x or y is missing / excluded
        xi, yi = table.paired_arrays(x_name, y_name)
        if len(xi) < 1:
            continue
        color = (colors[i] if colors and i < len(colors) else styles.color(i))

        # one shape per series as well as one colour: told apart in grey print and by
        # colour-blind readers (user, 2026-11-18)
        marker = styles.marker(i)
        ax.scatter(xi, yi, color=color, s=30, zorder=3, alpha=0.8, marker=marker,
                   linewidths=0.4, edgecolors="white", label=y_name)

        if show_regression and len(xi) >= 3:
            slope, intercept, r, p_val, _ = stats.linregress(xi, yi)
            x_line = np.linspace(xi.min(), xi.max(), 200)
            ax.plot(x_line, slope * x_line + intercept, color=color, linewidth=1.5,
                    linestyle="--", alpha=0.8,
                    # format_p: "p=0.0000" printed for any p < 5e-5
                    label=f"  R²={r**2:.3f}, {format_p(p_val)}")

    ax.set_xlabel(xlabel or x_name)
    ax.set_ylabel(ylabel or ", ".join(y_names))
    ax.set_title(title or table.name)
    if len(y_names) > 1 or show_regression:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
