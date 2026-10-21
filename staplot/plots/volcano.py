"""Volcano plot — fold-change vs significance."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure

from ..models.project import DataTable
from . import styles


@styles.styled
def plot_volcano(
    table: DataTable,
    fc_col: str,
    p_col: str,
    fc_thresh: float = 1.0,    # |log2 FC| threshold
    p_thresh: float = 0.05,    # p-value threshold (on raw p, not −log10)
    title: str = "",
    label_col: str | None = None,
    label_top: int = 10,
    color_up: str | None = None,
    color_down: str | None = None,
    color_ns: str = "#aaaaaa",
) -> matplotlib.figure.Figure:
    """
    fc_col  : log2(fold-change) column
    p_col   : raw p-value column (will be −log10 transformed)
    fc_thresh: |log2FC| threshold for colouring  (default 1 = 2×)
    p_thresh : p-value significance threshold      (default 0.05)
    label_col: optional column of feature names for labelling top hits
    label_top: how many top hits (by p-value) to label
    """
    groups = table.group_arrays()

    if fc_col not in groups or p_col not in groups:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Columns not found",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    # fold change, p-value and name of a feature come from the same row
    fc_vals, fc_rows = table.get_column(fc_col).valid_values_with_indices()
    pv_vals, pv_rows = table.get_column(p_col).valid_values_with_indices()
    fc_map, pv_map = dict(zip(fc_rows, fc_vals)), dict(zip(pv_rows, pv_vals))
    rows = np.array(sorted(r for r in set(fc_rows) & set(pv_rows) if pv_map[r] > 0), dtype=int)
    fc = np.array([fc_map[r] for r in rows], dtype=float)
    pv = np.array([pv_map[r] for r in rows], dtype=float)
    log_p = -np.log10(pv)

    # colours
    c_up   = color_up   or styles.color(1)   # orange
    c_down = color_down or styles.color(0)   # blue

    up   = (fc >=  fc_thresh) & (pv <= p_thresh)
    down = (fc <= -fc_thresh) & (pv <= p_thresh)
    ns   = ~(up | down)

    fig, ax = plt.subplots(figsize=(5.5, 5))

    ax.scatter(fc[ns],   log_p[ns],   color=color_ns, s=16, alpha=0.6,
               linewidths=0, zorder=2)
    ax.scatter(fc[down], log_p[down], color=c_down,   s=18, alpha=0.8,
               linewidths=0.2, edgecolors="white", zorder=3,
               label=f"Down ({int(down.sum())})")
    ax.scatter(fc[up],   log_p[up],   color=c_up,     s=18, alpha=0.8,
               linewidths=0.2, edgecolors="white", zorder=3,
               label=f"Up ({int(up.sum())})")

    # threshold lines
    ax.axhline(-np.log10(p_thresh), color="#555", linewidth=0.9,
               linestyle="--", zorder=1)
    ax.axvline( fc_thresh,          color="#555", linewidth=0.9,
               linestyle="--", zorder=1)
    ax.axvline(-fc_thresh,          color="#555", linewidth=0.9,
               linestyle="--", zorder=1)

    # annotations for top hits
    if label_col is not None and label_col in groups:
        # labels may be text — read the raw values
        labels_raw = [str(v) if v is not None else "" for v in table.get_column(label_col).values]
        sig_idx  = np.where(up | down)[0]
        order    = np.argsort(-log_p[sig_idx])[:label_top]
        for k in order:
            real_i = rows[sig_idx[k]]          # original row of this feature
            if real_i < len(labels_raw):
                lbl = labels_raw[real_i]
                ax.annotate(
                    lbl,
                    (fc[sig_idx[k]], log_p[sig_idx[k]]),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=6, color="#222",
                )
    elif label_top > 0 and (up.any() or down.any()):
        # Label top hits by index number if no label column
        sig_i  = np.where(up | down)[0]
        top_k  = sig_i[np.argsort(-log_p[sig_i])[:label_top]]
        for k in top_k:
            ax.annotate(
                str(rows[k]),                  # original row number
                (fc[k], log_p[k]),
                textcoords="offset points", xytext=(4, 3),
                fontsize=6, color="#444",
            )

    ax.set_xlabel(r"$\log_2$(Fold Change)")
    ax.set_ylabel(r"$-\log_{10}$(p-value)")
    ax.set_title(title or f"Volcano: {fc_col} vs {p_col}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
