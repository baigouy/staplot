import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure

from ..models.project import DataTable
from . import styles
from .significance import add_significance_brackets

VIOLIN_CUT = 2.0        # bandwidths drawn past the data (seaborn's default)
VIOLIN_HALF_WIDTH = 0.4


VIOLIN_INNERS = ("lines", "box", "none")
VIOLIN_SIDES = ("both", "left", "right")


def _violin(ax, x, arr, color, inner="lines", side="both", label=None):
    """One violin at *x*: a Gaussian density (Scott's bandwidth) that tapers
    VIOLIN_CUT bandwidths past the smallest / largest value, and inside it (*inner*):
    "lines", a thin line over the range, a thick one over the quartiles and a white
    median dot; "box", a mini box plot (quartile box, median line, whiskers to the
    furthest values within 1.5 IQR); "none", nothing. *side* "left" / "right" draws only
    that half (a half violin), its inner marks just inside the flat edge."""
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    top = float(arr.max())
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    spread = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if arr.size > 1 and spread > 0:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(arr, bw_method="scott")
        bw = kde.factor * spread
        y = np.linspace(arr.min() - VIOLIN_CUT * bw, arr.max() + VIOLIN_CUT * bw, 200)
        density = kde(y)
        half = density / density.max() * VIOLIN_HALF_WIDTH
        lo_edge = x if side == "right" else x - half
        hi_edge = x if side == "left" else x + half
        ax.fill_betweenx(y, lo_edge, hi_edge, facecolor=color, alpha=0.7,
                         edgecolor="#333333", linewidth=0.8, zorder=2, label=label, gid="violin")
        top = float(y[-1])
    # a half violin: its inner marks inside the half, next to the flat edge
    x = x + {"left": -1, "right": 1}.get(side, 0) * VIOLIN_HALF_WIDTH * 0.15
    if inner == "box":
        from matplotlib.patches import Rectangle
        iqr = q3 - q1
        lo = float(arr[arr >= q1 - 1.5 * iqr].min())
        hi = float(arr[arr <= q3 + 1.5 * iqr].max())
        w = VIOLIN_HALF_WIDTH * (0.3 if side == "both" else 0.18)
        ax.vlines(x, lo, q1, color="#333333", linewidth=0.9, zorder=3, gid="inner")
        ax.vlines(x, q3, hi, color="#333333", linewidth=0.9, zorder=3, gid="inner")
        ax.add_patch(Rectangle((x - w / 2, q1), w, q3 - q1, facecolor="white", edgecolor="#333333",
                               linewidth=0.9, zorder=3, gid="inner"))
        ax.hlines(med, x - w / 2, x + w / 2, color="#333333", linewidth=1.4, zorder=4, gid="median")
    elif inner != "none":
        ax.vlines(x, arr.min(), arr.max(), color="#333333", linewidth=1.0, zorder=3, gid="inner")
        ax.vlines(x, q1, q3, color="#333333", linewidth=4.0 if side == "both" else 3.0, zorder=3, gid="inner")
        ax.scatter([x], [med], s=14 if side == "both" else 9, color="white", edgecolors="#333333", linewidths=0.6, zorder=4,
                   gid="median")
    return top                  # the brackets go above the tapered tip, not the data


@styles.styled
def plot_box(
    table: DataTable,
    plot_type: str = "box",
    show_points: bool = True,
    title: str = "",
    ylabel: str = "",
    colors: list | None = None,
    sig_pairs: list | None = None,
    violin_inner: str = "lines",
    violin_side: str = "both",
) -> matplotlib.figure.Figure:
    half = plot_type == "violin" and violin_side in ("left", "right")
    cols = [c for c in table.columns if len(c.valid_values()) > 0]
    names = [c.name for c in cols]
    arrays = [c.valid_values() for c in cols]

    if not names:
        fig, ax = styles.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, ax = styles.subplots(figsize=(max(3.5, len(names) * 0.9), 4))
    palette = [
        (colors[i] if colors and i < len(colors) else styles.color(i))
        for i in range(len(names))
    ]

    if plot_type == "violin":
        # matplotlib's violinplot stops the density at the smallest and largest value:
        # flat-cut violins (user: "why are the violins so truncated at top and
        # bottom"). The density tapers 2 bandwidths past the data, as seaborn and
        # Prism draw it, with a small box inside (range, quartiles, median).
        violin_tops = {}
        for i, arr in enumerate(arrays):
            violin_tops[names[i]] = _violin(ax, i, np.asarray(arr, float), palette[i], inner=violin_inner,
                                         side=violin_side)
    else:
        bp = ax.boxplot(
            arrays,
            # same x as points, ticks and brackets (0..n-1); boxplot's default
            # 1..n drew every box one group to the right of its data
            positions=range(len(names)),
            patch_artist=True,
            medianprops=dict(color="#333333", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", markersize=4, alpha=0.5),
            widths=0.5,
        )
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        # named parts, for element_colors ("make the medians red")
        for part, gid in (("boxes", "box"), ("medians", "median"), ("whiskers", "whiskers"),
                          ("caps", "whiskers"), ("fliers", "outliers")):
            for artist in bp.get(part, []):
                artist.set_gid(gid)

    if show_points:
        rng = np.random.default_rng(42)
        for i, col in enumerate(cols):
            vals, row_idxs = col.valid_values_with_indices()
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            if half:
                # half violins: the points on the open side (a "raincloud")
                jitter = jitter * 0.6 + (0.18 if violin_side == "left" else -0.18)
            sc = ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                color="#222222",
                s=18,
                zorder=5,
                alpha=0.65,
                linewidths=0.3,
                edgecolors="white",
                picker=5,
                gid="points",
            )
            sc._staplot_meta = {"column": col.name, "rows": row_idxs, "exclude_all": False}

    ax.set_xticks(range(len(names)))
    rotation = 20 if max(len(n) for n in names) > 8 else 0
    # tilted labels end at their tick; level ones are centred on it
    ax.set_xticklabels(names, rotation=rotation, ha="right" if rotation else "center")
    ax.set_ylabel(ylabel or "Value")
    ax.set_title(title or table.name)

    if sig_pairs:
        # Whisker tops as bracket heights
        box_tops: dict[str, float] = {}
        for name, arr in zip(names, arrays):
            box_tops[name] = float(np.max(arr)) if len(arr) > 0 else 0.0
            if plot_type == "violin" and violin_tops.get(name) is not None:
                box_tops[name] = max(box_tops[name], violin_tops[name])
        add_significance_brackets(ax, names, sig_pairs, box_tops)

    if styles.target_axes() is None:
        fig.tight_layout()
    return fig


@styles.styled
def plot_split_violin(
    x_levels: list,
    halves: list,
    arrays: dict,
    pair_p: dict | None = None,
    show_points: bool = True,
    title: str = "",
    ylabel: str = "",
    xlabel: str = "",
    colors: list | None = None,
    violin_inner: str = "lines",
    legend_title: str = "",
) -> matplotlib.figure.Figure:
    """Split violins: at each *x_levels* position, the left half is ``halves[0]`` and
    the right half ``halves[1]`` (two conditions side by side: sex, treatment...), a
    legend naming the halves. *arrays*[(x_level, half)] = the values. *pair_p*:
    {x_level: p} of the two halves' comparison, drawn as a short bracket with stars."""
    from ..stats.descriptive import significance_stars
    fig, ax = styles.subplots(figsize=(max(3.5, len(x_levels) * 1.0 + 1.0), 4))
    palette = [(colors[i] if colors and i < len(colors) else styles.color(i)) for i in range(2)]
    tops = {}
    rng = np.random.default_rng(42)
    for i, xl in enumerate(x_levels):
        for k, (hv, side) in enumerate(zip(halves, ("left", "right"))):
            arr = np.asarray(arrays.get((xl, hv), []), float)
            top = _violin(ax, i, arr, palette[k], inner=violin_inner, side=side,
                          label=str(hv) if i == 0 else None)
            if top is not None:
                tops[xl] = max(tops.get(xl, top), top)
            if show_points and arr.size:
                arr = arr[np.isfinite(arr)]
                off = -0.2 if side == "left" else 0.2
                ax.scatter(i + off + rng.uniform(-0.04, 0.04, arr.size), arr, color="#222222", s=10,
                           zorder=5, alpha=0.6, linewidths=0.3, edgecolors="white", gid="points")
    ax.set_xticks(range(len(x_levels)))
    ax.set_xticklabels([str(v) for v in x_levels])
    ax.set_ylabel(ylabel or "Value")
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_title(title)
    # the halves named in the plot (which side is which condition)
    # above the axes: inside, it ran into the brackets
    leg = ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3 if legend_title else 2,
                    fontsize=8, handlelength=1.0, borderaxespad=0.2)
    if legend_title:
        # the factor named above its two levels ("sex": female, male)
        leg.set_title(legend_title, prop={"size": 8})
        leg._legend_box.align = "left"
    if pair_p:
        y_lo, y_hi = ax.get_ylim()
        rng_y = y_hi - y_lo
        new_top = y_hi
        for i, xl in enumerate(x_levels):
            p = pair_p.get(xl)
            if p is None or not np.isfinite(p):
                continue
            y = tops.get(xl, y_hi) + 0.03 * rng_y
            tip = 0.02 * rng_y
            ax.plot([i - 0.2, i - 0.2, i + 0.2, i + 0.2], [y - tip, y, y, y - tip], "-",
                    color="#222222", lw=1.0, clip_on=False, gid="brackets")
            label = significance_stars(p)
            ax.text(i, y + tip * 0.3, label, ha="center", va="bottom", fontsize=9,
                    fontweight="bold" if label not in ("ns", "n/a") else "normal", color="#222222",
                    gid="brackets")
            new_top = max(new_top, y + 0.1 * rng_y)
        if new_top > y_hi:
            ax.set_ylim(top=new_top)
    if styles.target_axes() is None:
        fig.tight_layout()
    return fig
