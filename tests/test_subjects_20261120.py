"""Several cells per animal (user 2026-11-20, "repeated measures"): subject= keeps each
value's subject; the groups are compared on the subjects' means (SuperPlot) and a mixed
model is reported; each subject's mean is drawn as a large dot."""
import matplotlib

matplotlib.use("Agg")
import numpy as np

from staplot import api


def _csv(tmp_path):
    rng = np.random.default_rng(0)
    rows = ["genotype,animal,area"]
    for g, base in (("WT", 10.0), ("KO", 13.0)):
        for a in range(4):
            shift = rng.normal(0, 1.0)
            for _ in range(15):
                rows.append(f"{g},{g}{a},{base + shift + rng.normal(0, 0.5):.3f}")
    p = tmp_path / "cells.csv"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(p)


def test_the_subjects_are_the_replicates(tmp_path):
    p = _csv(tmp_path)
    t = api.table_from_csv(p, columns=["genotype", "area"], subject="animal")
    fig, rec = api.plot(t, kind="box")
    assert rec["unit"] == "subject means"
    assert rec["mixed_model"]["test"].startswith("linear mixed model")
    dots = [a for a in fig.findobj() if api.plot_part(a) == "subject_means"]
    assert sum(len(d.get_offsets()) for d in dots) == 8
    # the same data without subjects: 30 "replicates" per group, a much smaller p
    _f, rec_cells = api.plot(api.table_from_csv(p, columns=["genotype", "area"]), kind="box")
    p_sub = rec["comparisons"][0]["p"]
    p_cells = rec_cells["comparisons"][0]["p"]
    assert p_sub > p_cells
