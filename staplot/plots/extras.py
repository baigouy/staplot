"""More kinds and axis options for staplot.api.plot (user, 2026-11-20):

* stacked bars (counts or % per category), with a chi-square test of the counts;
* dose-response curves: a 4-parameter logistic fit per series (EC50, Hill slope, R²);
* a broken y axis (one tall group no longer flattens the others);
* tick step, number format and log base of the y axis;
* one-sided error bars (only up / only down);
* an image with its coordinate axes (pixels or calibrated units), e.g. a mask.
"""
from __future__ import annotations

import numpy as np

from . import styles

Y_FORMATS = ("plain", "sci", "percent")
ERROR_SIDES = ("both", "up", "down")


# ---------------------------------------------------------------------------
# stacked bars
# ---------------------------------------------------------------------------
def plot_stacked(table, percent: bool = False, title: str = "", ylabel: str = "", xlabel: str = "",
                 colors=None, legend_title: str = ""):
    """One bar per level of factor B (x), stacked by factor A (the colours): the sum of
    each cell (counts) or, with *percent*, each stack as % of its bar. Returns
    (fig, {"counts": {x: {stack: n}}})."""
    xs = list(table.row_labels)
    stacks = list(table.col_groups)
    sums = {x: {s: float(np.nansum([v for v in table.data[s][x] if v is not None] or [0]))
                for s in stacks} for x in xs}
    fig, ax = styles.subplots()
    bottom = np.zeros(len(xs))
    for i, s in enumerate(stacks):
        h = np.array([sums[x][s] for x in xs])
        if percent:
            tot = np.array([sum(sums[x].values()) or 1.0 for x in xs])
            h = 100.0 * h / tot
        c = (colors[i] if colors and i < len(colors) else styles.color(i))
        bars = ax.bar(range(len(xs)), h, bottom=bottom, width=0.62, color=c, edgecolor="white",
                      linewidth=0.8, label=str(s), zorder=2)
        for b in bars:
            b.set_gid("bar")
        bottom += h
    ax.set_xticks(range(len(xs)), [str(x) for x in xs])
    ax.set_ylabel(ylabel or ("%" if percent else "count"))
    if xlabel:
        ax.set_xlabel(xlabel)
    if percent:
        ax.set_ylim(0, 100)
    ax.legend(title=legend_title or None, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, {"counts": sums}


def chi_square(counts: dict) -> dict:
    """Chi-square test of a {x: {stack: count}} table (Fisher's exact for 2 x 2 with small counts)."""
    from scipy import stats
    xs = list(counts)
    stacks = list(next(iter(counts.values()))) if counts else []
    m = np.array([[counts[x][s] for s in stacks] for x in xs], float)
    if m.shape[0] < 2 or m.shape[1] < 2 or (m.sum(axis=0) == 0).any() or (m.sum(axis=1) == 0).any():
        return {"test": "chi-square", "error": "needs at least 2 x 2 non-empty categories"}
    if np.any(np.abs(m - np.round(m)) > 1e-9):
        return {"test": "chi-square", "error": "the cells are not counts (whole numbers)"}
    chi2, p, dof, expected = stats.chi2_contingency(m)
    if m.shape == (2, 2) and (expected < 5).any():
        _odds, p = stats.fisher_exact(m)
        return {"test": "Fisher's exact test", "p": float(p)}
    return {"test": "chi-square", "chi2": float(chi2), "dof": int(dof), "p": float(p)}


# ---------------------------------------------------------------------------
# dose-response
# ---------------------------------------------------------------------------
def logistic4(x, bottom, top, ec50, hill):
    return bottom + (top - bottom) / (1.0 + (ec50 / np.asarray(x, float)) ** hill)


def fit_logistic4(x, y) -> dict:
    """4-parameter logistic fit: {bottom, top, ec50, hill, r2} (or {"error"})."""
    from scipy.optimize import curve_fit
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[ok], y[ok]
    if len(np.unique(x)) < 4:
        return {"error": "a dose-response fit needs at least 4 different positive doses"}
    order = np.argsort(x)
    lo, hi = float(np.min(y)), float(np.max(y))
    rising = np.mean(y[order][-3:]) >= np.mean(y[order][:3])
    p0 = [lo, hi, float(np.exp(np.median(np.log(x)))), 1.0 if rising else -1.0]
    try:
        p, _cov = curve_fit(logistic4, x, y, p0=p0, maxfev=20000)
    except Exception as e:                       # no convergence: said, not raised
        return {"error": f"the fit did not converge ({e})"}
    pred = logistic4(x, *p)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
    return {"bottom": float(p[0]), "top": float(p[1]), "ec50": float(abs(p[2])), "hill": float(p[3]),
            "r2": 1.0 - ss_res / ss_tot, "n": int(len(x))}


def plot_dose_response(table, error: str = "sem", title: str = "", xlabel: str = "", ylabel: str = "",
                       colors=None, log_x: bool = True, annotate: bool = True):
    """The first column = the dose, every other column = a response series: mean ± error
    per dose and the fitted 4PL curve, its EC50 in the legend. Returns (fig, record)."""
    from ..stats.descriptive import describe
    groups = table.group_arrays()
    names = list(groups)
    if len(names) < 2:
        raise ValueError("a dose-response plot needs the dose column, then one column per response")
    x_all = np.asarray(groups[names[0]], float)
    fig, ax = styles.subplots()
    fits = {}
    for i, name in enumerate(names[1:]):
        y_all = np.asarray(groups[name], float)
        n = min(len(x_all), len(y_all))
        x, y = x_all[:n], y_all[:n]
        c = (colors[i] if colors and i < len(colors) else styles.color(i))
        doses = sorted({float(v) for v in x if np.isfinite(v)})
        means, errs = [], []
        for d in doses:
            vals = y[(x == d) & np.isfinite(y)]
            s = describe(vals) if vals.size else None
            means.append(s["mean"] if s else np.nan)
            e = 0.0 if (not s or error == "none") else s.get(error, s["sem"])
            errs.append(0.0 if not np.isfinite(e) else e)
        eb = ax.errorbar(doses, means, yerr=errs if error != "none" else None, fmt="o", color=c, ms=4,
                         capsize=3, zorder=3)
        for part in [*eb.lines[1], *eb.lines[2]]:
            part.set_gid("error_bars")
        fit = fit_logistic4(x, y)
        fits[name] = fit
        label = str(name)
        if "error" not in fit:
            pos = np.array([d for d in doses if d > 0])
            grid = np.geomspace(pos.min(), pos.max(), 200) if log_x else np.linspace(pos.min(), pos.max(), 200)
            line, = ax.plot(grid, logistic4(grid, fit["bottom"], fit["top"], fit["ec50"], fit["hill"]), "-",
                            color=c, lw=1.4, zorder=2, gid="fit")
            if annotate:
                label += f"  EC50 = {fit['ec50']:.3g}"
        ax.plot([], [], "o-", color=c, label=label)
    if log_x and np.nanmin(x_all) > 0:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel or str(names[0]))
    ax.set_ylabel(ylabel or ("response" if len(names) > 2 else str(names[1])))
    ax.legend(frameon=False, fontsize=8)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, {"test": "4-parameter logistic fit", "fits": fits}


# ---------------------------------------------------------------------------
# axis options on a drawn plot
# ---------------------------------------------------------------------------
def set_y_ticks(ax, step=None, fmt=None, log_base=None):
    """The y axis' tick step, number format ("plain", "sci", "percent", or a format
    such as "{:.1f}" / "%.1f") and log base (with a log axis)."""
    from matplotlib import ticker
    if log_base:
        if ax.get_yscale() != "log":
            ax.set_yscale("log", base=float(log_base))
        else:
            ax.set_yscale("log", base=float(log_base))
    if step:
        step = float(step)
        if step <= 0:
            raise ValueError("y_step must be positive")
        if ax.get_yscale() == "log":
            raise ValueError("y_step is for a linear axis; a log axis takes log_base")
        ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
    if fmt:
        f = str(fmt)
        if f == "plain":
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        elif f == "sci":
            sf = ticker.ScalarFormatter(useMathText=True)
            sf.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(sf)
        elif f == "percent":
            ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        elif "{" in f:
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter(f.replace("{:", "{x:")))
        elif "%" in f:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(f))
        else:
            raise ValueError(f"y_format is one of {Y_FORMATS} or a format such as '{{:.1f}}'")


def one_sided_errors(fig, side: str):
    """Error bars drawn only above ("up") or below ("down") the mean."""
    if side in (None, "both"):
        return
    if side not in ERROR_SIDES:
        raise ValueError(f"error_side must be one of {ERROR_SIDES}")
    from ..api import plot_part
    for ax in fig.axes:
        parts = [a for a in ax.get_children() if plot_part(a) == "error_bars"]
        mids = {}
        for a in parts:
            if hasattr(a, "get_segments"):
                segs = []
                for seg in a.get_segments():
                    (x0, y0), (x1, y1) = seg[0], seg[-1]
                    lo, hi = min(y0, y1), max(y0, y1)
                    mid = (lo + hi) / 2
                    mids[round(float(x0), 9)] = mid
                    segs.append([(x0, mid), (x1, hi)] if side == "up" else [(x0, lo), (x1, mid)])
                a.set_segments(segs)
        for a in parts:
            if hasattr(a, "get_xdata") and not hasattr(a, "get_segments"):
                xs, ys = np.asarray(a.get_xdata(), float), np.asarray(a.get_ydata(), float)
                below = [ys[i] < mids.get(round(float(xs[i]), 9), ys[i]) for i in range(len(xs))]
                if len(xs) and all(below) and side == "up" or len(xs) and not any(below) and side == "down":
                    a.set_visible(False)


def broken_axes_marks(top, bottom):
    """The two halves of a broken y axis: no spine between them, small diagonal marks."""
    top.spines["bottom"].set_visible(False)
    bottom.spines["top"].set_visible(False)
    top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    d = 0.012
    for ax, y in ((top, 0.0), (bottom, 1.0)):
        kw = dict(transform=ax.transAxes, color="#222222", clip_on=False, lw=0.9)
        h = d * (3.0 if ax is top else 1.0)
        ax.plot((-d, +d), (y - h, y + h), **kw)


# ---------------------------------------------------------------------------
# an image with its axes
# ---------------------------------------------------------------------------
def image_plot(path_or_array, *, pixel_size: float | None = None, unit: str = "px", cmap: str = "gray",
               vmin=None, vmax=None, colorbar: bool = False, colorbar_label: str = "", title: str = "",
               xlabel: str | None = None, ylabel: str | None = None, contour=None, contour_color: str = "#ff3b30",
               width: float | None = None, height: float | None = None, origin: str = "upper",
               interpolation: str = "nearest"):
    """An image (a mask, a segmentation, a heat map, a micrograph) drawn with its
    coordinate axes, as matplotlib's imshow: in pixels, or in *unit* with *pixel_size*.
    *contour*: another image (a mask) whose outlines are drawn on top. Returns a Figure."""
    import matplotlib.pyplot as plt
    img = _image(path_or_array)
    if img.ndim == 3 and img.shape[-1] not in (3, 4):
        img = img.max(axis=0)                    # a stack: its maximum projection
    h, w = img.shape[:2]
    s = float(pixel_size) if pixel_size else 1.0
    extent = (0, w * s, h * s, 0) if origin == "upper" else (0, w * s, 0, h * s)
    if width or height:
        fw = float(width or (float(height) * w / h + (0.8 if colorbar else 0.3)))
        fh = float(height or (float(width) * h / w + 0.3))
    else:
        fw, fh = 3.6 + (0.8 if colorbar else 0), 3.6 * h / w + 0.2
    fig, ax = plt.subplots(figsize=(fw, fh))
    im = ax.imshow(img, cmap=None if img.ndim == 3 else cmap, vmin=vmin, vmax=vmax, extent=extent,
                   origin=origin, interpolation=interpolation)
    if contour is not None:
        m = _image(contour)
        ax.contour(np.linspace(0, w * s, m.shape[1]), np.linspace(0, h * s, m.shape[0]),
                   (m > 0).astype(float) if m.max() <= 1 or len(np.unique(m)) <= 2 else m, levels=[0.5],
                   colors=[contour_color], linewidths=0.9)
    u = unit if pixel_size else "px"
    ax.set_xlabel(f"x ({u})" if xlabel is None else xlabel)
    ax.set_ylabel(f"y ({u})" if ylabel is None else ylabel)
    if colorbar and img.ndim == 2:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if colorbar_label:
            cb.set_label(colorbar_label)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def _image(src):
    if isinstance(src, np.ndarray):
        return src
    path = str(src)
    if path.lower().endswith(".npy"):
        return np.load(path)
    try:
        import tifffile
        if path.lower().endswith((".tif", ".tiff")):
            return np.asarray(tifffile.imread(path))
    except ImportError:
        pass
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im)
