"""staplot styles are applied only while staplot builds/draws a plot: making a
plot used to update matplotlib's GLOBAL rcParams (fonts, dpi, tight bbox...), so
every later matplotlib figure of the process (e.g. PyFigures equations, colour
bars) changed look after one data plot."""
import io
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pytest

from staplot import api
from staplot import plots
from staplot.models.project import DataTable
from staplot.plots import styles


def _table():
    rng = np.random.default_rng(0)
    t = DataTable("t")
    for name, mean in (("Ctrl", 5), ("Treated", 7), ("Mutant", 6)):
        t.add_column(name).values = list(rng.normal(mean, 1, 12))
    return t


def _grouped(tmp_path):
    rows = [("A", "x", 1.0), ("A", "x", 1.2), ("A", "y", 2.0), ("A", "y", 2.1),
            ("B", "x", 1.5), ("B", "x", 1.4), ("B", "y", 3.0), ("B", "y", 3.2)]
    csv = tmp_path / "grouped.csv"
    csv.write_text("factor_a,factor_b,value\n"
                   + "".join(f"{a},{b},{v}\n" for a, b, v in rows), encoding="utf-8")
    return api.grouped_table_from_csv(str(csv))


BUILDERS = {
    "bar": lambda t: plots.plot_bar(t),
    "box": lambda t: plots.plot_box(t),
    "violin": lambda t: plots.plot_box(t, plot_type="violin"),
    "scatter": lambda t: plots.plot_scatter(t),
    "before_after": lambda t: plots.plot_before_after(t),
    "histogram": lambda t: plots.plot_histogram(t),
    "qq": lambda t: plots.plot_qq(t),
    "heatmap": lambda t: plots.plot_heatmap(t),
    "api.plot": lambda t: api.plot(t, kind="bar")[0],
}


def _snapshot():
    return {k: str(v) for k, v in mpl.rcParams.items()}


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    buf.seek(0)
    return mpimg.imread(buf)


@pytest.mark.parametrize("name", list(BUILDERS))
def test_making_a_plot_leaves_global_rcparams_alone(name):
    before = _snapshot()
    fig = BUILDERS[name](_table())
    _png(fig)
    plt.close(fig)
    after = _snapshot()
    assert {k for k in before if before[k] != after[k]} == set()


def test_grouped_api_plot_leaves_global_rcparams_alone(tmp_path):
    table = _grouped(tmp_path)
    before = _snapshot()
    fig, rec = api.plot(table, kind="grouped")
    assert "error" not in rec              # the ANOVA note is drawn too
    _png(fig)
    plt.close(fig)
    assert _snapshot() == before


@pytest.mark.parametrize("kind", ["bar", "box", "scatter"])
def test_api_plot_show_title(kind):
    """show_title=False: no title unless one is given (the table name otherwise)."""
    t = _table()
    assert api.plot(t, kind=kind)[0].axes[0].get_title() == "t"
    assert api.plot(t, kind=kind, show_title=False)[0].axes[0].get_title() == ""
    assert api.plot(t, kind=kind, title="Day 3", show_title=False)[0].axes[0].get_title() == "Day 3"
    plt.close("all")


def test_apply_selects_a_preset_without_touching_rcparams():
    before = _snapshot()
    try:
        rc = styles.apply("Nature / Science")
        assert styles.current_preset() == "Nature / Science"
        assert rc["axes.labelsize"] == 7
        assert _snapshot() == before
    finally:
        styles.set_preset("Default (Staplot)")


@pytest.mark.parametrize("name", list(BUILDERS))
def test_plots_look_as_when_the_style_was_global(name):
    """Pixel-identical to the old behaviour (style in the global rcParams while
    the plot was built AND drawn) when staplot draws it inside its style."""
    with mpl.rc_context(styles.rc()):
        old = _png(BUILDERS[name](_table()))
    fig = BUILDERS[name](_table())
    with styles.context():
        new = _png(fig)
    plt.close("all")
    assert old.shape == new.shape and np.array_equal(old, new)


def test_text_keeps_the_staplot_font_when_drawn_outside_the_style():
    """Font families are resolved at draw time for generic names ("sans-serif"),
    so the style names concrete families: the plot's text keeps them."""
    fig = plots.plot_bar(_table())
    families = {tuple(t.get_fontfamily()) for t in fig.findobj(matplotlib.text.Text)}
    plt.close(fig)
    assert families == {tuple(styles.rc()["font.family"])}
    assert mpl.rcParams["font.family"] != styles.rc()["font.family"]
