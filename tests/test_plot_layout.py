"""Layout and size of a plot, only when asked (user, 2026-11-18: "the plot part of the
MCP should allow layout changes, and/or size changes"): aspect, width / height,
font_scale, group order, y range, log y, rotated x labels, legend place."""
import os

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from staplot import api  # noqa: E402


@pytest.fixture
def csv(tmp_path):
    rng = np.random.default_rng(1)
    path = tmp_path / "t.csv"
    pd.DataFrame({"WT": rng.normal(10, 1, 8), "mutA": rng.normal(13, 1, 8),
                  "mutB": rng.normal(10.5, 1, 8)}).to_csv(path, index=False)
    return str(path)


def test_the_default_figure_is_unchanged(csv):
    t = api.table_from_csv(csv)
    assert list(api.plot(t, "box")[0].get_size_inches()) == list(api.plot(t, "box", order=None, legend=None)[0].get_size_inches())


def test_size(csv):
    t = api.table_from_csv(csv)
    w0, h0 = api.plot(t, "box")[0].get_size_inches()
    assert list(api.plot(t, "box", width=3, height=2)[0].get_size_inches()) == [3, 2]
    w, h = api.plot(t, "box", width=3)[0].get_size_inches()          # one given: same shape
    assert abs(w / h - w0 / h0) < 1e-9 and w == 3


def test_font_scale(csv):
    t = api.table_from_csv(csv)
    base = api.plot(t, "bar")[0].axes[0].yaxis.label.get_fontsize()
    assert api.plot(t, "bar", font_scale=1.5)[0].axes[0].yaxis.label.get_fontsize() == pytest.approx(1.5 * base)


def test_group_order_and_the_test_follows(csv):
    t = api.table_from_csv(csv)
    fig, rec = api.plot(t, "bar", order=["mutB", "wt"])
    assert [x.get_text() for x in fig.axes[0].get_xticklabels()] == ["mutB", "WT"]
    assert rec["test"].startswith(("Welch", "Mann"))                  # two groups left
    with pytest.raises(ValueError, match="no group named"):
        api.plot(t, "bar", order=["nope"])
    code = api.plot_code(csv, "bar", order=["mutB", "WT"])
    assert "order=['mutB', 'WT']" in code
    ns = {}
    cwd = os.getcwd()
    os.chdir(os.path.dirname(csv))
    try:
        exec(compile(code, "plot_code", "exec"), ns)
    finally:
        os.chdir(cwd)
    assert ns["names"] == ["mutB", "WT"]


def test_axis_and_labels(csv):
    t = api.table_from_csv(csv)
    ax = api.plot(t, "bar", ylim=(0, None), xtick_rotation=45)[0].axes[0]
    assert ax.get_ylim()[0] == 0 and ax.get_xticklabels()[0].get_rotation() == 45
    assert api.plot(t, "box", log_y=True)[0].axes[0].get_yscale() == "log"


def test_legend_place(csv):
    t = api.table_from_csv(csv)
    assert api.plot(t, "scatter", legend="none")[0].axes[0].get_legend() is None
    assert api.plot(t, "scatter", legend="outside")[0].axes[0].get_legend() is not None
    with pytest.raises(ValueError):
        api.plot(t, "scatter", legend="left")


def test_auto_picks_the_plot_that_shows_the_data(tmp_path):
    rng = np.random.default_rng(0)
    few = tmp_path / "few.csv"
    pd.DataFrame({"A": [1.0, 2, 3], "B": [2.0, 3, 4]}).to_csv(few, index=False)
    some = tmp_path / "some.csv"
    pd.DataFrame({"WT": rng.normal(10, 1, 8), "mut": rng.normal(12, 1, 8)}).to_csv(some, index=False)
    many = tmp_path / "many.csv"
    pd.DataFrame({"WT": rng.normal(10, 1, 40), "mut": rng.normal(12, 1, 40)}).to_csv(many, index=False)
    xy = tmp_path / "xy.csv"
    pd.DataFrame({"area (um2)": rng.uniform(100, 300, 20), "intensity (a.u.)": rng.uniform(1, 9, 20)}).to_csv(xy, index=False)
    series = tmp_path / "series.csv"
    pd.DataFrame({"t": np.arange(10.0), "signal": rng.normal(5, 1, 10)}).to_csv(series, index=False)
    two = tmp_path / "two.csv"
    pd.DataFrame([(g, t, rng.normal()) for g in ("WT", "mut") for t in ("ctrl", "drug") for _ in range(4)],
                 columns=["genotype", "treatment", "value"]).to_csv(two, index=False)
    tidy = tmp_path / "tidy.csv"
    pd.DataFrame([(g, t, d, rng.normal(), rng.normal()) for g in ("WT", "mut") for t in ("a", "b")
                  for d in ("d1", "d2") for _ in range(3)],
                 columns=["genotype", "treatment", "day", "length", "width"]).to_csv(tidy, index=False)
    picks = {name: api.suggest_kind(str(tmp_path / f"{name}.csv"))["kind"]
             for name in ("few", "some", "many", "xy", "series", "two", "tidy")}
    assert picks == {"few": "bar", "some": "box", "many": "violin", "xy": "scatter", "series": "scatter",
                     "two": "grouped", "tidy": None}
    fig, _ = api.plot(api.table_from_csv(str(many)), "auto", annotate=False)
    assert any(type(c).__name__ in ("PolyCollection", "FillBetweenPolyCollection") for c in fig.axes[0].collections)
    assert "kind='violin'" in api.plot_code(str(many), "auto")
    with pytest.raises(ValueError, match="tidy table"):
        api.plot_code(str(tidy), "auto")


def test_rename_groups_and_tilt_the_names_both_ways(csv):
    """"rename mutA into mutant A"; "rotate x labels by 45 degrees left" (= anticlockwise,
    rising "/"), then "the opposite direction"."""
    t = api.table_from_csv(csv)
    fig, rec = api.plot(t, "box", rename={"muta": "mutant A"}, xtick_rotation=45)
    ticks = fig.axes[0].get_xticklabels()
    assert [x.get_text() for x in ticks] == ["WT", "mutant A", "mutB"]
    assert "mutant A" in {c["a"] for c in rec["comparisons"]} | {c["b"] for c in rec["comparisons"]}
    assert ticks[0].get_rotation() == 45 and ticks[0].get_ha() == "right"      # "/" ends at its tick
    ticks = api.plot(t, "box", xtick_rotation=-45)[0].axes[0].get_xticklabels()
    assert ticks[0].get_rotation() == 315 and ticks[0].get_ha() == "left"      # "\" starts at its tick
    with pytest.raises(ValueError, match="no group named"):
        api.plot(t, "box", rename={"nope": "x"})
