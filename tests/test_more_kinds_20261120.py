"""User, 2026-11-20: a broken y axis, dose-response fits, stacked / % bars, histograms and
heatmaps through api.plot, tick step / format / log base, one-sided error bars, and an
image (a mask) drawn with its coordinate axes."""
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from staplot import api


@pytest.fixture
def groups(tmp_path):
    p = tmp_path / "g.csv"
    p.write_text("WT,KO,big\n10,15,400\n11,16,420\n9,14,410\n10.5,15.5,405\n", encoding="utf-8")
    return str(p)


def test_broken_y_axis(groups):
    fig, rec = api.plot(api.table_from_csv(groups), kind="bar", ybreak=[20, 350], ylabel="area")
    top, bottom = fig.axes[:2]
    fig.canvas.draw()
    assert bottom.get_ylim()[1] == 20 and top.get_ylim()[0] == 350
    assert rec["ybreak"] == [20, 350]
    assert fig._supylabel.get_text() == "area"
    with pytest.raises(ValueError, match="low must be below"):
        api.plot(api.table_from_csv(groups), kind="bar", ybreak=[50, 10])


def test_dose_response_fit(tmp_path):
    doses = [0.01, 0.1, 1, 10, 100, 1000]
    rows = ["dose,response"]
    for d in doses:
        for k in (-1, 0, 1):
            rows.append(f"{d},{5 + 95 / (1 + (3.0 / d) ** 1.2) + k}")
    p = tmp_path / "dr.csv"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    fig, rec = api.plot(api.table_from_csv(str(p)), kind="dose_response")
    fit = rec["fits"]["response"]
    assert abs(fit["ec50"] - 3.0) / 3.0 < 0.1 and fit["r2"] > 0.98
    assert fig.axes[0].get_xscale() == "log"
    assert "EC50" in fig.axes[0].get_legend().get_texts()[0].get_text()


def test_stacked_and_percent_bars_with_their_test(tmp_path):
    p = tmp_path / "pheno.csv"
    p.write_text("phenotype,genotype,n\nnormal,WT,40\nnormal,KO,10\ndefect,WT,5\ndefect,KO,30\n", encoding="utf-8")
    fig, rec = api.plot(api.grouped_table_from_csv(str(p)), kind="stacked", percent=True)
    tops = {}
    for b in [a for a in fig.findobj() if api.plot_part(a) == "bar"]:
        tops[round(b.get_x(), 3)] = max(tops.get(round(b.get_x(), 3), 0), b.get_y() + b.get_height())
    assert all(abs(v - 100) < 1e-6 for v in tops.values())
    assert rec["test"] == "chi-square" and rec["p"] < 0.001


def test_histogram_and_heatmap(groups):
    fig, _ = api.plot(api.table_from_csv(groups), kind="histogram", bins=5)
    assert fig.axes[0].patches
    fig, _ = api.plot(api.table_from_csv(groups), kind="heatmap", cmap="viridis")
    assert fig.axes[0].images


def test_y_ticks_and_one_sided_errors(groups):
    fig, _ = api.plot(api.table_from_csv(groups), kind="bar", y_step=100, y_format="{:.0f}", error_side="up")
    fig.canvas.draw()
    ticks = fig.axes[0].get_yticks()
    assert np.allclose(np.diff(ticks), 100)
    segs = [s for a in fig.findobj() if api.plot_part(a) == "error_bars" and hasattr(a, "get_segments")
            for s in a.get_segments()]
    bars = {round(b.get_x() + b.get_width() / 2, 6): b.get_height()
            for b in fig.findobj() if api.plot_part(b) == "bar"}
    for seg in segs:
        x = round(float(seg[0][0]), 6)
        assert min(seg[0][1], seg[-1][1]) == pytest.approx(bars[x])     # starts at the mean
    fig, _ = api.plot(api.table_from_csv(groups), kind="box", log_y=True, log_base=2)
    assert fig.axes[0].get_yscale() == "log"


def test_an_image_with_its_axes(tmp_path):
    mask = np.zeros((40, 60), np.uint8)
    mask[10:30, 20:45] = 1
    np.save(tmp_path / "mask.npy", mask)
    fig = api.image_plot(str(tmp_path / "mask.npy"), pixel_size=0.5, unit="µm", cmap="viridis", colorbar=True,
                         contour=mask)
    ax = fig.axes[0]
    assert ax.get_xlim() == (0, 30) and ax.get_xlabel() == "x (µm)" and len(fig.axes) == 2
    fig = api.image_plot(mask)
    assert fig.axes[0].get_xlabel() == "x (px)"


def test_the_code_keeps_the_new_options(groups):
    code = api.plot_code(groups, "bar", ybreak=[20, 350], y_step=50, error_side="up")
    assert "ybreak=[20, 350]" in code and "y_step=50" in code and "error_side='up'" in code
