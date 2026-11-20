"""Extra layers on a drawn plot (user 2026-11-20: "several plots for the same data such as
point cloud and lines") and a second y axis on the right ("graphs with dual y axes").

* layers - drawn over the groups of a bar / box / violin / paired plot:
    "mean_line"   a line joining the groups' means,
    "median_line" a line joining their medians,
    "mean_error"  each group's mean +- error (sd, sem, ci95) as a dot and whiskers.
* y2 - another measure on a right-hand axis, at the same x positions: the levels of the
  y2 table (its columns, or the values of its first column) matched to the plot's x
  (group names, or the x values of a line plot), drawn as a line, points or bars of
  mean +- error.
"""
from __future__ import annotations

import numpy as np

from ..stats.descriptive import describe

LAYERS = {
    "mean_line": "a line joining the groups' means",
    "median_line": "a line joining the groups' medians",
    "mean_error": "each group's mean ± error (the plot's error: sd, sem, ci95) as a dot and whiskers",
}
Y2_KINDS = ("line", "points", "bar")


def _summary(arr, error):
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return None
    d = describe(arr)
    err = 0.0 if error == "none" else float(d.get(error, d["sem"]))
    return {"mean": float(d["mean"]), "median": float(np.median(arr)),
            "err": 0.0 if not np.isfinite(err) else err, "n": int(arr.size)}


def add_layers(ax, groups: dict, layers, error: str = "sem", color: str | None = None):
    """*layers* over the groups (name -> values) drawn at x = 0, 1, 2... on *ax*."""
    layers = [layers] if isinstance(layers, str) else list(layers or [])
    bad = [l for l in layers if l not in LAYERS]
    if bad:
        raise ValueError(f"unknown layer(s) {bad}; layers are {list(LAYERS)}")
    stats = [_summary(v, error) for v in groups.values()]
    xs = [i for i, s in enumerate(stats) if s is not None]
    s_ok = [s for s in stats if s is not None]
    color = color or "#222222"
    for layer in layers:
        if layer in ("mean_line", "median_line"):
            key = "mean" if layer == "mean_line" else "median"
            line, = ax.plot(xs, [s[key] for s in s_ok], "-", color=color, lw=1.2, zorder=7,
                            marker="o", ms=3)
            line._staplot_part = "layers"
        elif layer == "mean_error":
            eb = ax.errorbar([x + 0.0 for x in xs], [s["mean"] for s in s_ok], yerr=[s["err"] for s in s_ok],
                             fmt="o", color=color, ms=4, capsize=3, lw=1.1, zorder=7)
            for a in [eb.lines[0], *eb.lines[1], *eb.lines[2]]:
                a._staplot_part = "layers"


def _levels(spec: dict, read_table, select_columns):
    """{level name: values} of the y2 table: [x, value] columns -> the values per x;
    otherwise each numeric column is a level."""
    path = spec.get("csv")
    if not path:
        raise ValueError('y2 needs the table: {"csv": "path", "columns": [x, value]} (or the '
                         'columns of a wide table, one per x level)')
    cols = spec.get("columns")
    df = select_columns(read_table(path, spec.get("skip_rows")), cols, spec.get("where"))
    import pandas as pd
    if cols and len(cols) == 2:
        x, v = df.columns[0], df.columns[1]
        vals = pd.to_numeric(df[v], errors="coerce")
        out = {}
        for level, sub in vals.groupby(df[x], sort=False):
            out[str(level)] = sub.to_numpy(float)
        return out
    return {str(c): pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().any()}


def _as_number(s):
    try:
        return float(str(s).replace(",", "."))
    except ValueError:
        return None


def add_y2(fig, ax, spec: dict, read_table, select_columns, error: str = "sem"):
    """A right-hand y axis on *ax* from *spec* ({"csv", "columns", "where", "kind",
    "label", "color", "error", "ylim", "log"}). Returns the new axes."""
    if not isinstance(spec, dict):
        raise ValueError('y2 is {"csv": path, "columns": [x, value], "kind": "line"|"points"|"bar", '
                         '"label": ..., "color": ...}')
    kind = spec.get("kind") or "line"
    if kind not in Y2_KINDS:
        raise ValueError(f"y2 kind must be one of {Y2_KINDS}")
    levels = _levels(spec, read_table, select_columns)
    if not levels:
        raise ValueError("the y2 table has no numbers")
    # where each level goes: a group name under a tick, or a number on a numeric x axis
    ticks = list(ax.get_xticks())
    labels = [t.get_text().split("\n")[0].strip().lower() for t in ax.get_xticklabels()]
    axis_numeric = all(_as_number(l.replace("−", "-")) is not None for l in labels if l)
    pos, missing = {}, []
    for name in levels:
        key = name.strip().lower()
        if key in labels:
            pos[name] = ticks[labels.index(key)]
        elif _as_number(name) is not None and axis_numeric:
            pos[name] = _as_number(name)
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"y2 levels {missing} are not on the plot's x axis "
                         f"(it shows {[t.get_text() for t in ax.get_xticklabels()]})")
    err_type = spec.get("error") or error
    pts = sorted(((pos[n], _summary(v, err_type)) for n, v in levels.items()), key=lambda p: p[0])
    pts = [(x, s) for x, s in pts if s is not None]
    color = spec.get("color") or "#c0392b"
    ax2 = ax.twinx()
    ax2._staplot_y2 = True
    xs = [x for x, _s in pts]
    means = [s["mean"] for _x, s in pts]
    errs = [s["err"] for _x, s in pts] if err_type != "none" else None
    if kind == "bar":
        bars = ax2.bar(xs, means, width=0.25, color=color, alpha=0.55, zorder=1)
        for b in bars:
            b._staplot_part = "y2"
        if errs:
            eb = ax2.errorbar(xs, means, yerr=errs, fmt="none", ecolor=color, capsize=2, lw=0.9)
            for a in [*eb.lines[1], *eb.lines[2]]:
                a._staplot_part = "y2"
    else:
        eb = ax2.errorbar(xs, means, yerr=errs, fmt="-o" if kind == "line" else "o", color=color,
                          ms=4, capsize=2.5, lw=1.2, zorder=8)
        for a in [eb.lines[0], *eb.lines[1], *eb.lines[2]]:
            a._staplot_part = "y2"
    ax2.set_ylabel(spec.get("label") or (spec.get("columns") or ["y2"])[-1], color=color)
    ax2.tick_params(axis="y", colors=color)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(color)
    for side in ("top", "left", "bottom"):
        ax2.spines[side].set_visible(False)
    if spec.get("log"):
        ax2.set_yscale("log")
    lim = spec.get("ylim")
    if lim:
        lo, hi = (list(lim) + [None, None])[:2]
        cur = ax2.get_ylim()
        ax2.set_ylim(cur[0] if lo is None else float(lo), cur[1] if hi is None else float(hi))
    elif kind == "bar" and min(means) >= 0:
        ax2.set_ylim(bottom=0)
    ax.spines["right"].set_visible(False)
    return ax2
