"""Residual diagnostic plots after regression or ANOVA."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from scipy import stats

from . import styles


@styles.styled
def plot_residuals(
    fitted: np.ndarray,
    residuals: np.ndarray,
    title: str = "Residual Diagnostics",
    color: str | None = None,
) -> matplotlib.figure.Figure:
    """Four-panel residual diagnostic figure."""
    c = color or styles.color(0)
    std_res = residuals / (np.std(residuals, ddof=1) or 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    # 1. Residuals vs Fitted
    ax = axes[0][0]
    ax.scatter(fitted, residuals, color=c, s=22, alpha=0.75, linewidths=0.3, edgecolors="white")
    ax.axhline(0, color="#888", linewidth=1, linestyle="--")
    ax.set_xlabel("Fitted values"); ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")

    # 2. Q-Q of residuals
    ax = axes[0][1]
    (osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
    ax.scatter(osm, osr, color=c, s=22, alpha=0.75, linewidths=0.3, edgecolors="white")
    x_line = np.array([osm[0], osm[-1]])
    ax.plot(x_line, slope * x_line + intercept, color="#555", linewidth=1.5)
    ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Sample quantiles")
    ax.set_title("Normal Q-Q")

    # 3. Scale-Location (√|std residuals| vs fitted)
    ax = axes[1][0]
    ax.scatter(fitted, np.sqrt(np.abs(std_res)), color=c, s=22, alpha=0.75,
               linewidths=0.3, edgecolors="white")
    ax.set_xlabel("Fitted values"); ax.set_ylabel("√|Standardised residuals|")
    ax.set_title("Scale-Location")

    # 4. Histogram of residuals
    ax = axes[1][1]
    ax.hist(residuals, bins="auto", color=c, alpha=0.7, edgecolor="white")
    x_line2 = np.linspace(residuals.min(), residuals.max(), 200)
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(residuals)
        scale = len(residuals) * (residuals.max() - residuals.min()) / max(int(np.sqrt(len(residuals))), 1)
        ax.plot(x_line2, kde(x_line2) * scale, color="#333", linewidth=1.5)
    except Exception:
        pass
    ax.set_xlabel("Residuals"); ax.set_ylabel("Count")
    ax.set_title("Residual distribution")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def residuals_from_curve_fit(result) -> matplotlib.figure.Figure | None:
    """Generate residual plot from a curve-fit AnalysisResult."""
    d = result.details
    if "r2" not in d:
        return None
    # We don't store raw residuals in the result yet — return None gracefully
    return None
