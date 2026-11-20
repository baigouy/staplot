"""The plot editor's Layers menu, right-hand axis and inverted axes (2026-11-20)."""
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
    p = tmp_path / "t.csv"
    p.write_text("genotype,area,cells\nWT,10,100\nWT,11,110\nKO,15,50\nKO,16,60\n", encoding="utf-8")
    return str(p)


def _center(ed, artist):
    bb = artist.get_window_extent(ed.canvas.get_renderer())
    return (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2


def test_layers_y2_and_inverted_axes(app, csv):
    from staplot import api
    from staplot.ui.plot_editor import PlotEditor
    ed = PlotEditor(csv, "box", table_kwargs={"columns": ["genotype", "area"]})
    ed.layer_actions["mean_line"].trigger()
    assert ed.kwargs()["layers"] == ["mean_line"] and ed.layer_actions["mean_line"].isChecked()
    assert [a for a in ed.fig.findobj() if api.plot_part(a) == "layers"]
    ed.set("y2", {"csv": csv, "columns": ["genotype", "cells"], "label": "cells"})
    assert len(ed.fig.axes) == 2
    # a click on the right axis' label edits that label
    kind, part = ed.select_at(*_center(ed, ed.fig.axes[1].yaxis.label))
    assert (kind, part) == ("text", "axis_labels")
    labels = [ed.inspector_form.itemAt(i).widget() for i in range(ed.inspector_form.count())]
    assert any("Right axis" in (getattr(w, "text", lambda: "")() or "") for w in labels if w is not None)
    ed.set("invert_y", True)
    assert ed.fig.axes[0].yaxis_inverted()
    ed.undo()
    assert not ed.fig.axes[0].yaxis_inverted()
    ed.layer_actions["mean_line"].trigger()
    assert "layers" not in ed.kwargs()
    ed._set_kind("scatter")
    assert not ed.layers_button.isEnabled()
