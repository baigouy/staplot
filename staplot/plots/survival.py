import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure

from ..models.project import DataTable
from . import styles


def _km_estimate(times: np.ndarray, events: np.ndarray):
    """Kaplan-Meier estimator. events: 1=event occurred, 0=censored."""
    order = np.argsort(times)
    times, events = times[order], events[order]

    unique_t = np.unique(times[events == 1])
    n = len(times)
    survival = 1.0
    km_t = [0.0]
    km_s = [1.0]

    for t in unique_t:
        at_risk = np.sum(times >= t)
        deaths = np.sum((times == t) & (events == 1))
        if at_risk > 0:
            survival *= 1 - deaths / at_risk
        km_t.append(float(t))
        km_s.append(float(survival))

    return np.array(km_t), np.array(km_s)


@styles.styled
def plot_survival(
    table: DataTable,
    time_col: str | None = None,
    event_col: str | None = None,
    group_col: str | None = None,
    title: str = "",
    xlabel: str = "Time",
    ylabel: str = "Survival probability",
    colors: list | None = None,
) -> matplotlib.figure.Figure:
    col_names = [c.name for c in table.columns]
    if not col_names:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, ax = plt.subplots(figsize=(5, 4))

    if time_col and event_col:
        # a subject's time and event flag come from the same row
        times, events = table.paired_arrays(time_col, event_col)
        km_t, km_s = _km_estimate(times, events)
        ax.step(km_t, km_s, where="post", color=styles.color(0), linewidth=2)
        censored = times[events == 0]
        km_s_cens = np.interp(censored, km_t, km_s)
        ax.scatter(censored, km_s_cens, marker="+", color=styles.color(0), s=60, zorder=4)
    else:
        # Treat each column as a separate survival group (values = event times, no censoring)
        groups = table.group_arrays()
        for i, (name, arr) in enumerate(groups.items()):
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                continue
            events = np.ones(len(arr))
            km_t, km_s = _km_estimate(arr, events)
            color = (colors[i] if colors and i < len(colors) else styles.color(i))
            ax.step(km_t, km_s, where="post", color=color, linewidth=2, label=name)
        if len(groups) > 1:
            ax.legend()

    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or table.name)
    fig.tight_layout()
    return fig
