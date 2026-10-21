"""Per-column distribution visualisation: histogram + KDE + fit overlay + Q-Q plot."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from scipy import stats as sp_stats

from ..models.project import DataTable
from . import styles


_SCIPY_DISTS = {
    "Normal":      sp_stats.norm,
    "Log-normal":  sp_stats.lognorm,
    "Exponential": sp_stats.expon,
    "Gamma":       sp_stats.gamma,
    "Weibull":     sp_stats.weibull_min,
}


@styles.styled
def plot_distribution(
    table: DataTable,
    col_name: str,
    fits: list[dict] | None = None,
    show_normal_ref: bool = True,
) -> matplotlib.figure.Figure:
    """
    Two-panel figure for one column:
      Left  — histogram + KDE overlay + optional fitted-distribution curve
      Right — Q-Q plot vs Normal

    *fits* is the list returned by ``stats.distribution.fit_distributions()``.
    """
    col_obj = next((c for c in table.columns if c.name == col_name), None)
    if col_obj is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Column '{col_name}' not found",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    arr = col_obj.valid_values()
    n   = len(arr)
    if n < 3:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Not enough data (n={n})", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    color = styles.color(0)
    fig, (ax_hist, ax_qq) = plt.subplots(1, 2, figsize=(8.5, 3.6))

    # ── Left panel: histogram + overlays ─────────────────────────────
    ax_hist.hist(
        arr, bins="auto", density=True, alpha=0.55,
        color=color, edgecolor="white", linewidth=0.5, zorder=2,
    )

    # KDE
    if n >= 4:
        from scipy.stats import gaussian_kde
        kde    = gaussian_kde(arr, bw_method="scott")
        span   = max(arr.std() * 3, (arr.max() - arr.min()) * 0.3)
        x_kde  = np.linspace(arr.min() - span, arr.max() + span, 300)
        ax_hist.plot(x_kde, kde(x_kde), color=color, linewidth=2.0,
                     label="KDE", zorder=4)

    # Best-fit distribution (first in *fits*)
    if fits:
        best = fits[0]
        dist = _SCIPY_DISTS.get(best["distribution"])
        if dist is not None:
            try:
                params = best.get("params") or dist.fit(arr)
                span2  = max(arr.std() * 3.5, (arr.max() - arr.min()) * 0.4)
                x_fit  = np.linspace(max(arr.min() - span2, 1e-9 if best["distribution"] in
                                         {"Log-normal","Exponential","Gamma","Weibull"} else arr.min() - span2),
                                     arr.max() + span2, 300)
                ax_hist.plot(
                    x_fit, dist.pdf(x_fit, *params),
                    color="#d04040", linewidth=1.6, linestyle="--",
                    label=f"{best['distribution']} fit", zorder=5,
                )
            except Exception:
                pass

    # Normal reference curve
    if show_normal_ref:
        mu, sigma = float(np.mean(arr)), float(np.std(arr, ddof=1))
        if sigma > 0:
            x_norm = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
            ax_hist.plot(
                x_norm, sp_stats.norm.pdf(x_norm, mu, sigma),
                color="#888888", linewidth=1.0, linestyle=":",
                alpha=0.9, label="Normal ref.", zorder=3,
            )

    ax_hist.set_xlabel(col_name)
    ax_hist.set_ylabel("Density")
    ax_hist.set_title(f"Distribution — {col_name}  (n={n})")
    ax_hist.legend(fontsize=7, loc="upper right")

    # ── Right panel: Q-Q plot vs Normal ──────────────────────────────
    (osm, osr), (slope, intercept, r) = sp_stats.probplot(arr, dist="norm")
    ax_qq.scatter(
        osm, osr, s=22, alpha=0.75, color=color,
        linewidths=0.3, edgecolors="white", zorder=3,
    )
    qq_x = np.array([osm.min(), osm.max()])
    ax_qq.plot(qq_x, slope * qq_x + intercept,
               color="#444444", linewidth=1.3, zorder=2)

    # Colour the scatter by deviation from the line
    residuals = osr - (slope * osm + intercept)
    thresh    = 2.0 * residuals.std()
    outlier_mask = np.abs(residuals) > thresh
    if outlier_mask.any():
        ax_qq.scatter(
            osm[outlier_mask], osr[outlier_mask],
            s=28, color="#e05050", zorder=4,
            linewidths=0.5, edgecolors="#800000", alpha=0.9,
        )

    ax_qq.set_xlabel("Theoretical quantiles (Normal)")
    ax_qq.set_ylabel("Sample quantiles")
    ax_qq.set_title(f"Q-Q plot — r = {r:.3f}")

    fig.tight_layout(pad=1.2)
    return fig


@styles.styled
def plot_distribution_grid(
    table: DataTable,
    dist_details: dict,          # from run_distribution_check().details
    max_cols: int = 3,
) -> matplotlib.figure.Figure:
    """
    Compact grid overview: one mini histogram per column with
    normality verdict colour-coded (green = normal, orange = non-normal).
    """
    valid_cols = [c for c in table.columns if len(c.valid_values()) >= 3]
    if not valid_cols:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    n_cols  = min(len(valid_cols), max_cols)
    n_rows  = (len(valid_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.6, n_rows * 2.2),
                             squeeze=False)

    for idx, col in enumerate(valid_cols):
        r, c = divmod(idx, n_cols)
        ax   = axes[r][c]
        arr  = col.valid_values()
        info = dist_details.get(col.name, {})
        is_normal = info.get("is_normal", True)
        bar_color = styles.color(0) if is_normal else styles.color(1)

        ax.hist(arr, bins="auto", density=True,
                color=bar_color, alpha=0.65, edgecolor="white", linewidth=0.4)

        # Tiny normal reference
        mu, sd = float(np.mean(arr)), float(np.std(arr, ddof=1))
        if sd > 0:
            x = np.linspace(mu - 3.5 * sd, mu + 3.5 * sd, 200)
            ax.plot(x, sp_stats.norm.pdf(x, mu, sd),
                    color="#555555", linewidth=0.9, linestyle=":")

        verdict = "normal" if is_normal else "non-normal"
        ax.set_title(f"{col.name}\n({verdict})", fontsize=7.5,
                     color="#1a7a1a" if is_normal else "#b84000")
        ax.tick_params(labelsize=6)
        ax.set_ylabel("")

    # Hide unused axes
    for idx in range(len(valid_cols), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"Distribution overview — {table.name}", fontsize=9, y=1.01)
    fig.tight_layout(pad=0.6)
    return fig
