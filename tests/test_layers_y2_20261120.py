"""User, 2026-11-20: "can it make graphs with dual y axes? or several plots for the same data
such as point cloud and lines". layers= adds mean / median lines and mean ± error over the
groups; y2= draws another measure on a right-hand axis at the same x."""
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from staplot import api


@pytest.fixture
def csv(tmp_path):
    p = tmp_path / "area.csv"
    p.write_text("WT,KO,mutA\n10,15,12\n11,16,13\n9,14,11.5\n10.5,15.5,12.2\n", encoding="utf-8")
    return str(p)


def _parts(fig, part):
    return [a for a in fig.findobj() if api.plot_part(a) == part]


def test_box_with_points_and_a_line_through_the_means(csv):
    fig, _ = api.plot(api.table_from_csv(csv), kind="box", layers=["mean_line"],
                      element_colors={"layers": "#ff0000"})
    lines = _parts(fig, "layers")
    assert len(lines) == 1
    assert np.allclose(lines[0].get_ydata(), [10.125, 15.125, 12.175])
    assert list(lines[0].get_xdata()) == [0, 1, 2]
    assert matplotlib.colors.to_hex(lines[0].get_color()) == "#ff0000"
    assert _parts(fig, "points")                          # the point cloud is still there


def test_layers_follow_the_order_and_names(csv):
    fig, _ = api.plot(api.table_from_csv(csv), kind="violin", order=["KO", "WT"],
                      rename={"KO": "knock-out"}, layers=["median_line", "mean_error"])
    line = [a for a in _parts(fig, "layers") if len(getattr(a, "get_xdata", lambda: [])()) == 2][0]
    assert np.allclose(line.get_ydata(), [15.25, 10.25])
    with pytest.raises(ValueError, match="layers are"):
        api.plot(api.table_from_csv(csv), kind="bar", layers=["spline"])


def test_a_second_measure_on_a_right_hand_axis(csv, tmp_path):
    counts = tmp_path / "counts.csv"
    counts.write_text("genotype,cells\nWT,100\nWT,110\nKO,50\nKO,60\nmutA,80\nmutA,82\n", encoding="utf-8")
    fig, _ = api.plot(api.table_from_csv(csv), kind="bar",
                      y2={"csv": str(counts), "columns": ["genotype", "cells"], "kind": "line",
                          "label": "cells per field", "color": "#0000ff"})
    assert len(fig.axes) == 2
    ax2 = fig.axes[1]
    assert getattr(ax2, "_staplot_y2", False) and ax2.get_ylabel() == "cells per field"
    line = [a for a in _parts(fig, "y2") if hasattr(a, "get_xdata") and len(a.get_xdata()) == 3][0]
    assert list(line.get_xdata()) == [0, 1, 2] and np.allclose(line.get_ydata(), [105, 55, 81])
    with pytest.raises(ValueError, match="not on the plot's x axis"):
        api.plot(api.table_from_csv(csv), kind="bar",
                 y2={"csv": str(counts), "columns": ["genotype", "cells"], "kind": "line"}, order=["WT", "KO"])


def test_y2_on_a_time_course(tmp_path):
    growth = tmp_path / "growth.csv"
    growth.write_text("genotype,day,length\n" + "".join(
        f"{g},{d},{d * (2 if g == 'WT' else 1) + k}\n" for g in ("WT", "KO") for d in (0, 2, 4) for k in (0, 1)),
        encoding="utf-8")
    temp = tmp_path / "temp.csv"
    temp.write_text("day,temperature\n0,20\n2,25\n4,30\n", encoding="utf-8")
    fig, _ = api.plot(api.grouped_table_from_csv(str(growth)), kind="line",
                      y2={"csv": str(temp), "columns": ["day", "temperature"], "kind": "points"})
    line = [a for a in _parts(fig, "y2") if hasattr(a, "get_xdata") and len(a.get_xdata()) == 3][0]
    assert list(line.get_xdata()) == [0, 2, 4] and list(line.get_ydata()) == [20, 25, 30]


def test_the_code_of_the_plot_keeps_them(csv):
    code = api.plot_code(csv, "box", layers=["mean_line"])
    assert "layers=['mean_line']" in code


def test_inverted_axes_stay_inverted_once_drawn(csv, tmp_path):
    """User, 2026-11-20: "invert axis(es) ... I often found it painful to do"."""
    fig, _ = api.plot(api.table_from_csv(csv), kind="box", invert_y=True)
    fig.canvas.draw()
    assert fig.axes[0].yaxis_inverted() and not fig.axes[0].xaxis_inverted()
    fig, _ = api.plot(api.table_from_csv(csv), kind="bar", invert_x=True, ylim=[0, 30])
    fig.canvas.draw()
    assert fig.axes[0].xaxis_inverted() and tuple(fig.axes[0].get_ylim()) == (0, 30)
    tidy = tmp_path / "tidy.csv"
    tidy.write_text("g,day,v\n" + "".join(f"{g},{d},{d + k}\n" for g in "ab" for d in (1, 2) for k in (0, 1, 2)))
    fig, _ = api.plot_facets(str(tidy), "box", "day", columns=["g", "v"], invert_y=True)
    assert all(a.yaxis_inverted() for a in fig.axes if a.get_visible() and a.has_data())
    assert "invert_y=True" in api.plot_code(csv, "box", invert_y=True)
