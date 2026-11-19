"""Prism-style significance brackets drawn on bar / box axes."""

import numpy as np
from matplotlib.axes import Axes

from ..stats.descriptive import significance_stars, format_p


def add_significance_brackets(
    ax: Axes,
    group_names: list[str],
    pairs: list[tuple[str, str, float]],
    heights: dict[str, float],
    y_step_frac: float = 0.10,
    tip_frac: float = 0.04,
    fontsize: float = 9,
    use_stars: bool = True,
    color: str = "#222222",
    lw: float = 1.2,
    positions: dict | None = None,
):
    """
    Draw significance brackets above a bar / box plot.

    Parameters
    ----------
    group_names   : ordered list of group names (matches x-tick order)
    pairs         : list of (name_a, name_b, p_value)
    heights       : dict {name: y_top} — top of bar+error or whisker top
    y_step_frac   : gap between bracket levels as fraction of y-range
    tip_frac      : downward tick length as fraction of y-range
    positions     : {name: x} when the groups are not at 0, 1, 2... (bars side by
                    side within one x level of a grouped plot)
    """
    if not pairs:
        return

    name_to_x = ({n: float(positions[n]) for n in group_names} if positions
                 else {n: float(i) for i, n in enumerate(group_names)})
    y_lo, y_hi = ax.get_ylim()
    y_range = y_hi - y_lo
    step = y_step_frac * y_range
    tip = tip_frac * y_range

    # Sort by span (short = lower brackets)
    sorted_pairs = sorted(pairs, key=lambda t: abs(name_to_x.get(t[1], 0) - name_to_x.get(t[0], 0)))

    # Track the "ceiling" at each group index position
    ceiling = {n: heights.get(n, y_hi * 0.6) for n in group_names}

    drawn: list[tuple[float, float, float, float, str]] = []  # (xa, xb, y, p, label)

    for ga, gb, p_val in sorted_pairs:
        if ga not in name_to_x or gb not in name_to_x:
            continue
        xa, xb = name_to_x[ga], name_to_x[gb]
        lo_name, hi_name = (ga, gb) if xa < xb else (gb, ga)
        lo_x, hi_x = min(xa, xb), max(xa, xb)

        # All groups in the span
        span_names = [n for n in group_names if lo_x <= name_to_x[n] <= hi_x]

        # Bracket must sit above everything in its span + any brackets underneath
        base = max(ceiling[n] for n in span_names) + step * 0.25

        y_bracket = base
        y_top_of_bracket = y_bracket + step * 0.4

        # Update ceiling for all names in span
        for n in span_names:
            ceiling[n] = y_top_of_bracket + step * 0.15

        label = significance_stars(p_val) if use_stars else format_p(p_val)
        drawn.append((lo_x, hi_x, y_bracket, p_val, label))

    for xa, xb, y, p_val, label in drawn:
        # Horizontal bar
        ax.plot([xa, xb], [y, y], "-", color=color, lw=lw, clip_on=False, gid="brackets")
        # Left tick
        ax.plot([xa, xa], [y - tip, y], "-", color=color, lw=lw, clip_on=False, gid="brackets")
        # Right tick
        ax.plot([xb, xb], [y - tip, y], "-", color=color, lw=lw, clip_on=False, gid="brackets")
        # Label
        mid = (xa + xb) / 2
        bold = label not in ("ns", "n/a")
        txt = ax.text(
            mid, y + tip * 0.3, label,
            ha="center", va="bottom",
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            color=color,
            gid="brackets",
        )
        txt._staplot_p = p_val

    # Expand y-axis to show topmost bracket + padding
    if drawn:
        new_top = max(ceiling.values()) + step * 0.4
        if new_top > y_hi:
            ax.set_ylim(top=new_top)
