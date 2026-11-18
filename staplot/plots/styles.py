"""Publication-quality matplotlib style — clean, minimal, journal-ready.

The style is applied only while staplot builds a plot (:func:`styled` /
:func:`context`), never to matplotlib's global rcParams: staplot is used as a
library (e.g. PyFigures data-plot panels) and must not change how every later
matplotlib figure of the process looks."""

import functools

import matplotlib as mpl
from matplotlib import font_manager

_SANS = ["Arial", "Helvetica", "DejaVu Sans"]
# only installed ones: a family list warns about every missing family at draw time
_installed = {f.name for f in font_manager.fontManager.ttflist}
_FONT_FAMILY = [f for f in _SANS if f in _installed] or ["DejaVu Sans"]

# Base publication style
PUBLICATION_RC = {
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.labelcolor": "#222222",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "legend.fontsize": 9,
    "legend.frameon": False,
    # concrete families, not "sans-serif": texts keep their family, but a generic
    # one is looked up in rcParams["font.sans-serif"] when the figure is DRAWN,
    # i.e. after the style context has closed
    "font.family": _FONT_FAMILY,
    "font.sans-serif": _SANS,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "errorbar.capsize": 4,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
}

# Colorblind-friendly palette (Wong 2011)
PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#D55E00",  # vermilion
    "#000000",  # black
]


# ── Journal / venue presets ───────────────────────────────────────────
JOURNAL_PRESETS: dict[str, dict] = {
    "Default (Staplot)": PUBLICATION_RC,

    "Nature / Science": {
        **PUBLICATION_RC,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "figure.dpi": 300,
    },

    "Cell / JCB": {
        **PUBLICATION_RC,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "errorbar.capsize": 3,
    },

    "Classic (Prism-like)": {
        **PUBLICATION_RC,
        "axes.spines.top":   True,
        "axes.spines.right": True,
        "axes.grid":         True,
        "axes.grid.axis":    "y",
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "xtick.direction":   "in",
        "ytick.direction":   "in",
    },

    "Monochrome": {
        **PUBLICATION_RC,
        "axes.prop_cycle": mpl.cycler(
            color=["#000000", "#444444", "#888888", "#aaaaaa"]
        ),
    },
}

_current_preset: str = "Default (Staplot)"


def rc(preset: str | None = None) -> dict:
    """The rcParams of *preset* (or of the current preset)."""
    return dict(JOURNAL_PRESETS.get(preset or _current_preset, PUBLICATION_RC))


def context(preset: str | None = None):
    """``with context():`` — the preset's rcParams, restored on exit."""
    return mpl.rc_context(rc(preset))


_TARGET_AXES = []          # set by drawing_into(ax): the plotters draw there (facets)


class drawing_into:
    """``with drawing_into(ax):`` the plotters draw into *ax* instead of a new figure
    (one facet of a faceted plot)."""

    def __init__(self, ax):
        self.ax = ax

    def __enter__(self):
        _TARGET_AXES.append(self.ax)
        return self.ax

    def __exit__(self, *exc):
        _TARGET_AXES.pop()


def subplots(**kwargs):
    """``plt.subplots()``, or the axes set by :class:`drawing_into`."""
    if _TARGET_AXES:
        ax = _TARGET_AXES[-1]
        return ax.figure, ax
    import matplotlib.pyplot as plt
    return plt.subplots(**kwargs)


def target_axes():
    return _TARGET_AXES[-1] if _TARGET_AXES else None


def styled(func):
    """Decorator: build the plot inside the current preset's :func:`context`."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with context():
            return func(*args, **kwargs)
    return wrapper


def apply(preset: str | None = None) -> dict:
    """Select *preset* (if given) and return its rcParams.

    Formerly updated matplotlib's global rcParams; plots now apply the preset
    themselves, only while they are built (:func:`styled`)."""
    if preset is not None:
        set_preset(preset)
    return rc()


def set_preset(preset: str):
    global _current_preset
    _current_preset = preset


def current_preset() -> str:
    return _current_preset


def color(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


# marker shapes for series of points, in this order (with the colours: a series is
# told apart by its shape too)
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def marker(index: int) -> str:
    return MARKERS[index % len(MARKERS)]
