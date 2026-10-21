"""Q-Q normal probability plots."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from scipy import stats

from ..models.project import DataTable
from . import styles


@styles.styled
def plot_qq(
    table: DataTable,
    title: str = "",
    colors: list | None = None,
) -> matplotlib.figure.Figure:
    groups = table.group_arrays()
    names = [n for n, a in groups.items() if len(a) >= 3]
    n_plots = len(names)
    if n_plots == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data (need n ≥ 3)", ha="center", va="center", transform=ax.transAxes)
        return fig

    ncols = min(n_plots, 3)
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5), squeeze=False)

    for idx, name in enumerate(names):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        arr = groups[name]
        color = (colors[idx] if colors and idx < len(colors) else styles.color(idx))

        (osm, osr), (slope, intercept, _) = stats.probplot(arr, dist="norm")
        ax.scatter(osm, osr, color=color, s=22, zorder=3, alpha=0.85, linewidths=0.3, edgecolors="white")
        x_line = np.array([osm[0], osm[-1]])
        ax.plot(x_line, slope * x_line + intercept, color="#555555", linewidth=1.5, zorder=2)

        # Shapiro p-value annotation
        if len(arr) >= 3:
            _, p_sw = stats.shapiro(arr)
            pstr = f"p = {p_sw:.4f}" if p_sw >= 0.0001 else "p < 0.0001"
            verdict = "Normal" if p_sw > 0.05 else "Non-normal"
            ax.text(0.05, 0.95, f"S-W {pstr}\n{verdict}",
                    transform=ax.transAxes, fontsize=7, va="top",
                    color="green" if p_sw > 0.05 else "red")

        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Theoretical quantiles", fontsize=8)
        ax.set_ylabel("Sample quantiles", fontsize=8)

    # Hide unused axes
    for idx in range(n_plots, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(title or f"Q-Q Plots: {table.name}", fontsize=10, y=1.01)
    fig.tight_layout()
    return fig
