"""2026-11-19 (user): the two-way ANOVA written under a grouped plot is not elegant;
by default a figure shows stars (as box plots do), the text only on request."""
import os

import matplotlib
matplotlib.use("Agg")

from staplot import api

HERE = os.path.dirname(__file__)


def _table(tmp_path):
    import numpy as np
    rng = np.random.default_rng(0)
    rows = ["genotype,treatment,value"]
    for g, m in (("WT", 30), ("mutA", 22), ("mutB", 29)):
        for t, d in (("ctrl", 0), ("drug", 4)):
            rows += [f"{g},{t},{v:.2f}" for v in rng.normal(m + d, 1.0, 8)]
    p = tmp_path / "two.csv"
    p.write_text("\n".join(rows), encoding="utf-8")
    return str(p)


def _texts(fig):
    return [t.get_text() for t in fig.texts] + [t.get_text() for a in fig.axes for t in a.texts]


def test_grouped_shows_stars_not_the_anova_text(tmp_path):
    fig, rec = api.plot(api.grouped_table_from_csv(_table(tmp_path)), "grouped")
    texts = _texts(fig)
    assert not any("ANOVA" in t for t in texts)
    assert any(set(t) == {"*"} for t in texts)
    assert rec["effects"] and rec["within_x"][0]["sig_pairs"]


def test_stats_text_brings_the_anova_back(tmp_path):
    fig, _ = api.plot(api.grouped_table_from_csv(_table(tmp_path)), "grouped", stats_text=True)
    assert any("Two-way ANOVA" in t for t in _texts(fig))


def test_line_star_per_x_and_split_violin_without_text(tmp_path):
    fig, rec = api.plot(api.grouped_table_from_csv(_table(tmp_path)), "line")
    assert not any("ANOVA" in t for t in _texts(fig))
    assert any(set(t) == {"*"} for t in _texts(fig))
    fig, _ = api.plot(api.grouped_table_from_csv(_table(tmp_path), columns=["treatment", "genotype", "value"]),
                      "split_violin")
    assert not any("ANOVA" in t for t in _texts(fig))


def test_facets_keep_their_own_title(tmp_path):
    fig, _ = api.plot_facets(_table(tmp_path), "box", "treatment", columns=["genotype", "value"])
    assert fig._suptitle is None or fig._suptitle.get_text() == ""


def test_code_reruns_the_tests_of_each_x_level(tmp_path):
    code = api.plot_code(_table(tmp_path), "grouped")
    assert "anova_lm" in code and "treatment = ctrl" in code and "pairwise_tukeyhsd" in code
