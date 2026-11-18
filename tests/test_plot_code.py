"""The code of a plot (user, 2026-11-18: "can staplot print the code it used to generate
the fig") and the violin shape ("why are the violins so truncated at top and bottom")."""
import contextlib
import io
import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from staplot import api  # noqa: E402


def _run(code, folder):
    namespace = {}
    cwd = os.getcwd()
    os.chdir(folder)                         # part 1 saves plot.pdf where it runs
    try:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            exec(compile(code, "plot_code", "exec"), namespace)
    finally:
        os.chdir(cwd)
    return namespace, out.getvalue()


@pytest.fixture
def wide(tmp_path):
    p = tmp_path / "counts.csv"
    p.write_text("WT,mutA,mutB\n42.9,36.6,40.3\n38.9,,38.3\n44.3,NA,45.9\n44.8,35.4,abc\n36.1,35.1,39.3\n"
                 "38.1,31.1,39.6\n42.4,36.6,43.1\n41.1,33.9,42.7\n", encoding="utf-8")
    return str(p)


def test_the_code_redraws_the_plot_and_reruns_the_test_on_the_same_values(wide, tmp_path):
    code = api.plot_code(wide, "box")
    assert "api.plot(table, kind='box'" in code
    ns, printed = _run(code, tmp_path)
    assert (tmp_path / "plot.pdf").is_file()
    table = api.table_from_csv(wide)
    assert {n: len(a) for n, a in zip(ns["names"], ns["arrays"])} == {
        c.name: len(c.valid_values()) for c in table.columns}          # NA, empty, 'abc' left out alike
    rec = api.auto_test(table)
    if rec["test"].startswith("One-way"):
        assert f"F = {rec['omnibus']['stat']:.3f}" in printed


def test_dunn_in_the_code_gives_staplots_p_values(wide, tmp_path):
    code = api.plot_code(wide, "bar", stats_mode="nonparametric")
    assert "scikit" not in code                                         # no extra package
    _ns, printed = _run(code, tmp_path)
    rec = api.auto_test(api.table_from_csv(wide), only_significant=False, parametric=False)
    for c in rec["comparisons"]:
        assert f"{c['a']} vs {c['b']}: Dunn" in printed
        assert f"p (Bonferroni) = {c['p']:.3g}" in printed


def test_a_french_excel_text_table_is_read_alike(tmp_path):
    p = tmp_path / "fr.txt"
    p.write_text("WT;mut\n41,2;35,1\n40,1;abc\n42,5;34,3\n43,0;36,0\n", encoding="utf-8")
    ns, _ = _run(api.plot_code(str(p), "box"), tmp_path)
    assert [len(a) for a in ns["arrays"]] == [4, 3]


def test_the_grouped_and_scatter_code_runs(tmp_path):
    g = tmp_path / "two.csv"
    rows = ["genotype,treatment,value"] + [f"{a},{b},{10 + i}" for a in "AB" for b in "xy" for i in range(4)]
    g.write_text("\n".join(rows) + "\n", encoding="utf-8")
    _ns, printed = _run(api.plot_code(str(g), "grouped"), tmp_path)
    assert "C(A):C(B)" in printed
    s = tmp_path / "xy.csv"
    s.write_text("x,y\n" + "".join(f"{i},{2 * i + (i % 3)}\n" for i in range(12)), encoding="utf-8")
    _ns, printed = _run(api.plot_code(str(s), "scatter"), tmp_path)
    assert "n = 12" in printed


def test_violins_taper_past_the_data_and_brackets_clear_them(wide):
    table = api.table_from_csv(wide)
    fig, rec = api.plot(table, kind="violin")
    ax = fig.axes[0]
    shapes = [c for c in ax.collections if type(c).__name__ in ("PolyCollection", "FillBetweenPolyCollection")]
    assert len(shapes) == 3
    tops = [float(s.get_paths()[0].vertices[:, 1].max()) for s in shapes]
    data_max = max(float(np.max(c.valid_values())) for c in table.columns)
    assert max(tops) > data_max                                        # not cut flat at the data
    bracket_lows = [min(line.get_ydata()) for line in ax.lines if len(line.get_ydata()) == 4]
    if bracket_lows:
        assert min(bracket_lows) >= min(tops) - 1e-9


def test_sd_is_the_default_error_bar():
    import inspect
    assert inspect.signature(api.plot).parameters["error"].default == "sd"
