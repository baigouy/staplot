"""The visual plot editor (staplot.ui.plot_editor): the data beside the plot, a click on a
part of the plot selects it, its settings become staplot.api.plot arguments."""
import os

import pytest

os.environ.setdefault("QT_API", "pyqt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def csv(tmp_path):
    p = tmp_path / "counts.csv"
    p.write_text("WT,KO\n10,15\n11,16\n9,14\n10.5,15.5\n", encoding="utf-8")
    return str(p)


def _center(ed, artist):
    bb = artist.get_window_extent(ed.canvas.get_renderer())
    return (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2


def _bars(ed):
    from staplot import api
    return [a for a in ed.fig.findobj() if api.plot_part(a) == "bar"]


def test_click_a_bar_then_its_colour_is_that_groups(app, csv):
    from staplot.ui.plot_editor import PlotEditor
    ed = PlotEditor(csv, "bar")
    bars = _bars(ed)
    assert len(bars) == 2
    kind, key = ed.select_at(*_center(ed, bars[1]))
    assert (kind, key) == ("group", "KO")
    ed.set_in("colors", key, "#ff0000")
    assert ed.changed_kwargs() == {"colors": {"KO": "#ff0000"}}
    from matplotlib.colors import to_hex
    assert to_hex(_bars(ed)[1].get_facecolor()) == "#ff0000"
    ed.undo()
    assert ed.changed_kwargs() == {}
    ed.redo()
    assert ed.kwargs()["colors"] == {"KO": "#ff0000"}


def test_click_the_y_label_sets_its_text_and_the_size_of_every_axis_label(app, csv):
    from staplot.ui.plot_editor import PlotEditor
    ed = PlotEditor(csv, "box")
    ax = ed.fig.axes[0]
    kind, key = ed.select_at(*_center(ed, ax.yaxis.label))
    assert (kind, key) == ("text", "axis_labels")
    ed.set("ylabel", "Cells")
    ed.set_part("axis_labels", "pt", 14)
    ax = ed.fig.axes[0]
    assert ax.get_ylabel() == "Cells" and ax.yaxis.label.get_fontsize() == 14
    assert ed.changed_text_parts() == {"axis_labels": {"pt": 14}}
    # the tick labels: a group renamed
    kind, key = ed.select_at(*_center(ed, ax.get_xticklabels()[0]))
    assert (kind, key) == ("text", "ticks")
    ed.set_in("rename", "WT", "wild type")
    assert [t.get_text() for t in ed.fig.axes[0].get_xticklabels()][0] == "wild type"


def test_click_the_points_and_the_plot_area(app, csv):
    from staplot import api
    from staplot.ui.plot_editor import PlotEditor
    ed = PlotEditor(csv, "bar")
    pts = [a for a in ed.fig.findobj() if api.plot_part(a) == "points"]
    off = pts[0].get_offsets()[0]
    x, y = ed.fig.axes[0].transData.transform(off)
    assert ed.select_at(x, y)[:1] == ("element",)
    ax = ed.fig.axes[0]
    x0, y1 = ax.bbox.x0 + 3, ax.bbox.y1 - 3
    assert ed.select_at(x0, y1)[0] == "axes"
    ed.set("ylim", [0, 30])
    assert tuple(ed.fig.axes[0].get_ylim()) == (0, 30)


def test_edited_data_redraws_and_is_saved(app, csv):
    from staplot.ui.plot_editor import PlotEditor
    ed = PlotEditor(csv, "bar")
    idx = ed.model.index(0, 1)
    assert ed.model.setData(idx, "100")
    ed.render()
    assert ed._source()[0] != csv          # drawn from the edited table
    ed.save_data(confirm=False)
    text = open(csv, encoding="utf-8").read()
    assert text.splitlines()[1].split(",")[1] in ("100", "100.0")
    assert ed.data_saved


def test_a_bad_setting_is_reported_not_raised(app, csv):
    from staplot.ui.plot_editor import PlotEditor
    ed = PlotEditor(csv, "bar")
    ed.set("ylim", "nonsense")
    assert ed.message.text()
