"""Heatmap — values, correlation matrix, or grouped data."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib import cm

from ..models.project import DataTable
from . import styles


@styles.styled
def plot_heatmap(
    table: DataTable,
    mode: str = "values",          # "values" | "correlation"
    cmap: str = "RdYlBu_r",
    annotate: bool = True,
    fmt: str = ".2f",
    title: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
) -> matplotlib.figure.Figure:

    df = table.to_dataframe()
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    if mode == "correlation":
        mat = df.corr().values
        row_labels = col_labels = list(df.columns)
        cmap = "RdBu_r"
        vmin, vmax = -1.0, 1.0
        fmt = ".2f"
        plot_title = title or f"Correlation matrix: {table.name}"
    else:
        mat = df.values.astype(float)
        row_labels = [str(i + 1) for i in range(mat.shape[0])]
        col_labels = list(df.columns)
        plot_title = title or table.name

    nrows, ncols = mat.shape
    cell_size = max(0.5, min(1.0, 6.0 / max(nrows, ncols)))
    fig_w = max(4, ncols * cell_size + 1.5)
    fig_h = max(3, nrows * cell_size + 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mat, aspect="auto", cmap=cmap,
                   vmin=vmin if vmin is not None else np.nanmin(mat),
                   vmax=vmax if vmax is not None else np.nanmax(mat))

    ax.set_xticks(range(ncols))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(row_labels, fontsize=8)

    if annotate and nrows * ncols <= 200:
        thresh = (np.nanmax(mat) + np.nanmin(mat)) / 2
        for i in range(nrows):
            for j in range(ncols):
                val = mat[i, j]
                if np.isnan(val):
                    continue
                txt = format(val, fmt)
                color = "white" if val < thresh else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=max(6, 9 - nrows // 4), color=color)

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(plot_title)
    fig.tight_layout()
    return fig
