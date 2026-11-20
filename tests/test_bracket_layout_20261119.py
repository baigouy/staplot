"""2026-11-19 (user): "asterisk/significance position in plots is not ok, it does not
read well". Bracket levels were spaced in data units for a 9 pt label; a plot shrunk
into a figure panel with its text at print size had its stars run into the bracket
above. They are now stacked at draw time from the labels' real height."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from staplot.plots.significance import add_significance_brackets


def _boxes(fig, ax):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in ax.texts if t.get_gid() == "brackets"]
    bars = [l for l in ax.lines if l.get_gid() == "brackets" and l.get_ydata()[0] == l.get_ydata()[-1]]
    return [t.get_window_extent(r) for t in texts], [l.get_window_extent(r) for l in bars]


@pytest.mark.parametrize("fontsize, size", [(9, (3.5, 3)), (16, (1.6, 1.4)), (24, (1.6, 1.4))])
def test_no_label_runs_into_a_bracket(fontsize, size):
    fig, ax = plt.subplots(figsize=size, dpi=150)
    ax.bar([0, 1, 2], [30, 24, 28])
    ax.set_ylim(0, 35)
    add_significance_brackets(ax, ["a", "b", "c"], [("a", "b", 1e-5), ("a", "c", 1e-5), ("b", "c", 0.003)],
                              {"a": 31, "b": 25, "c": 29}, fontsize=fontsize)
    labels, bars = _boxes(fig, ax)
    assert len(labels) == 3 and len(bars) == 3
    for label in labels:
        for bar in bars:
            overlaps_x = label.x0 < bar.x1 and bar.x0 < label.x1
            assert not (overlaps_x and label.y0 < bar.y1 + 0.5 and bar.y0 - 0.5 < label.y1), (label, bar)
    plt.close(fig)


def test_the_labels_grow_with_the_text_after_the_plot_was_made():
    """PyFigures sets the text size when it draws the panel, after the plot is made."""
    fig, ax = plt.subplots(figsize=(1.6, 1.4), dpi=150)
    ax.bar([0, 1], [30, 24])
    ax.set_ylim(0, 35)
    add_significance_brackets(ax, ["a", "b"], [("a", "b", 1e-5)], {"a": 31, "b": 25}, fontsize=9)
    for t in ax.texts:
        t.set_fontsize(20)
    labels, bars = _boxes(fig, ax)
    assert labels[0].y1 <= ax.get_window_extent(fig.canvas.get_renderer()).y1 + 1   # the axis grew to hold it
    plt.close(fig)


def test_a_y_axis_top_the_user_set_stays_after_drawing(tmp_path):
    """2026-11-20: ylim=[0, 30] read back 0-18.3 once drawn (the brackets set the top)."""
    from staplot import api
    p = tmp_path / "c.csv"
    p.write_text("WT,KO\n10,15\n11,16\n9,14\n10.5,15.5\n")
    fig, _rec = api.plot(api.table_from_csv(str(p)), kind="bar", ylim=[0, 30])
    fig.canvas.draw()
    assert tuple(fig.axes[0].get_ylim()) == (0, 30)
