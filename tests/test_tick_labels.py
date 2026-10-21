"""x tick labels: centred on their tick when level, ending at it when tilted."""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from staplot.models.project import DataTable


def table(**columns):
    t = DataTable("t")
    for name, values in columns.items():
        t.add_column(name).values = list(values)
    return t


def _plots():
    from staplot.plots.bar import plot_bar
    from staplot.plots.before_after import plot_before_after
    from staplot.plots.box import plot_box
    return [plot_bar, plot_box, plot_before_after]


@pytest.mark.parametrize("index", range(3))
def test_level_tick_labels_are_centred_and_tilted_ones_end_at_the_tick(index):
    plot = _plots()[index]
    short = plot(table(WT=[1, 2, 3], KO=[2, 3, 4]))
    labels = short.axes[0].get_xticklabels()
    assert [t.get_ha() for t in labels] == ["center", "center"]
    assert all(t.get_rotation() == 0 for t in labels)
    long = plot(table(wild_type_control=[1, 2, 3], knockout_mutant=[2, 3, 4]))
    labels = long.axes[0].get_xticklabels()
    assert [t.get_ha() for t in labels] == ["right", "right"]
    assert all(t.get_rotation() == 20 for t in labels)
