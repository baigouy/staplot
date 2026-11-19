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
