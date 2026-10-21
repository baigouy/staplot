"""ROC curve plot."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure

from ..models.project import DataTable
from ..stats.roc import compute_roc
from . import styles


@styles.styled
def plot_roc(
    table: DataTable,
    label_col: str,
    score_col: str,
    title: str = "",
    color: str | None = None,
) -> matplotlib.figure.Figure:
    try:
        labels, scores = table.paired_arrays(label_col, score_col)
        fpr, tpr, auc = compute_roc(labels, scores)
    except ValueError as e:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, str(e), ha="center", va="center", wrap=True,
                transform=ax.transAxes)
        return fig
    c = color or styles.color(0)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, color=c, linewidth=2.0, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#aaaaaa", linewidth=1.0, linestyle="--", label="Random")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("1 − Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title(title or f"ROC: {score_col}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig
