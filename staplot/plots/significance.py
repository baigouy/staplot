"""Prism-style significance brackets drawn on bar / box axes."""

import numpy as np
from matplotlib.artist import Artist
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

    if not hasattr(ax, "_staplot_top0"):
        ax._staplot_top0 = y_hi                 # the top before any bracket raised it
    layout = []
    for xa, xb, y, p_val, label in drawn:
        # Horizontal bar
        bar, = ax.plot([xa, xb], [y, y], "-", color=color, lw=lw, clip_on=False, gid="brackets")
        # Left tick
        left, = ax.plot([xa, xa], [y - tip, y], "-", color=color, lw=lw, clip_on=False, gid="brackets")
        # Right tick
        right, = ax.plot([xb, xb], [y - tip, y], "-", color=color, lw=lw, clip_on=False, gid="brackets")
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
        span = [n for n in group_names if xa <= name_to_x[n] <= xb]
        layout.append((bar, left, right, txt, span))

    # Expand y-axis to show topmost bracket + padding
    if drawn:
        new_top = max(ceiling.values()) + step * 0.4
        if new_top > y_hi:
            ax.set_ylim(top=new_top)
        ax.add_artist(_BracketLayout(ax, layout, {n: heights.get(n, y_hi * 0.6) for n in group_names},
                                     y_step_frac, tip_frac, name_to_x))


class _BracketLayout(Artist):
    """Re-stacks the brackets each time the plot is drawn, from the real height of
    their labels: a level clears the stars below it at any size. The levels were set
    in data units from a 9 pt label; a plot shrunk into a figure panel with its text
    at print size (7 pt on the page, so relatively larger) had its stars run into the
    bracket above (user 2026-11-19: "asterisk position in plots ... does not read
    well"). Drawn first (lowest zorder), so the lines, labels and y axis drawn after
    it use the new positions; it draws nothing itself."""

    def __init__(self, ax, layout, heights, y_step_frac, tip_frac, name_to_x=None):
        super().__init__()
        self._x = dict(name_to_x or {})
        self.set_zorder(-1e9)
        self._ax, self._layout, self._heights = ax, layout, heights
        self._step_frac, self._tip_frac = y_step_frac, tip_frac

    def draw(self, renderer):
        ax = self._ax
        try:
            for _ in range(8):                   # the text height in data units depends on the top
                if not self._stack(ax, renderer):
                    break
        except Exception:                        # never break a plot over its brackets
            pass

    def _stack(self, ax, renderer):
        y_lo, y_hi = ax.get_ylim()
        y_range = y_hi - y_lo
        px = ax.get_window_extent(renderer).height
        if px <= 0 or y_range <= 0:
            return False
        per_px = y_range / px
        x_lo, x_hi = ax.get_xlim()
        width_px = ax.get_window_extent(renderer).width
        x_per_px = (x_hi - x_lo) / width_px if width_px > 0 else 0.0
        tip = self._tip_frac * y_range
        ceiling = dict(self._heights)
        for bar, left, right, txt, span in self._layout:
            text_h = txt.get_window_extent(renderer).height * per_px
            gap = max(0.025 * y_range, 0.3 * text_h)
            y = max(ceiling.get(n, y_lo) for n in span) + gap + tip * 0.5
            xa, xb = bar.get_xdata()[0], bar.get_xdata()[-1]
            bar.set_data([xa, xb], [y, y])
            left.set_data([xa, xa], [y - tip, y])
            right.set_data([xb, xb], [y - tip, y])
            txt.set_y(y + tip * 0.3)
            # a label wider than its bracket covers its neighbours' columns too
            half = txt.get_window_extent(renderer).width * abs(x_per_px) / 2
            mid = (xa + xb) / 2
            covered = set(span) | {n for n, x in self._x.items() if mid - half <= x <= mid + half}
            for n in covered:
                ceiling[n] = max(ceiling.get(n, y_lo), y + tip * 0.3 + text_h * 1.15)
        need = max(ceiling.values()) + max(0.025 * y_range, 0.0)
        tops = getattr(ax, "_staplot_bracket_tops", {})
        tops[id(self)] = need
        ax._staplot_bracket_tops = tops
        top = max([getattr(ax, "_staplot_top0", y_hi)] + list(tops.values()))
        if abs(top - y_hi) > 1e-6 * max(1.0, abs(y_range)):
            ax.set_ylim(top=top)
            return True                          # the scale changed: stack again
        return False
