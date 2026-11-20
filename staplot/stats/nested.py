"""Repeated measures within subjects (several cells per animal, several fields per
embryo): the subjects, not the cells, are the replicates (user 2026-11-20). The groups are
compared on each subject's mean ("SuperPlot", Lord et al. 2020), and a linear mixed
model (value ~ group, random intercept per subject) is reported beside it."""
from __future__ import annotations

import numpy as np


def has_subjects(table) -> bool:
    return any(getattr(c, "subjects", None) for c in getattr(table, "columns", []))


def subject_means(table) -> dict:
    """{group: {subject: mean}} (the order of the table's groups and of first appearance)."""
    out = {}
    for col in table.columns:
        subs = getattr(col, "subjects", None) or []
        vals = col.as_array()
        per: dict = {}
        for s, v in zip(subs, vals):
            if s is None or not np.isfinite(v):
                continue
            per.setdefault(str(s), []).append(float(v))
        out[col.name] = {s: float(np.mean(v)) for s, v in per.items()}
    return out


def means_table(table):
    """A table of the subjects' means (one value per subject per group)."""
    from ..api import table_from_columns
    t = table_from_columns({g: list(m.values()) for g, m in subject_means(table).items()},
                           name=getattr(table, "name", "data"))
    t.value_name = getattr(table, "value_name", "")
    return t


def mixed_model(table) -> dict:
    """value ~ group with a random intercept per subject (statsmodels MixedLM): the
    fixed effects' p-values against the first group, or {"error"}."""
    import pandas as pd
    rows = []
    for col in table.columns:
        for s, v in zip(getattr(col, "subjects", None) or [], col.as_array()):
            if s is not None and np.isfinite(v):
                rows.append({"group": col.name, "subject": f"{col.name}|{s}", "value": float(v)})
    df = pd.DataFrame(rows)
    if df.empty or df["group"].nunique() < 2:
        return {"error": "needs two groups"}
    try:
        import warnings
        import statsmodels.formula.api as smf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm("value ~ C(group)", df, groups=df["subject"]).fit(reml=True)
    except Exception as e:
        return {"error": f"the mixed model did not fit ({type(e).__name__})"}
    ref = str(table.columns[0].name)
    effects = []
    for name, p in fit.pvalues.items():
        if name.startswith("C(group)"):
            g = name.split("[T.", 1)[-1].rstrip("]")
            effects.append({"a": ref, "b": g, "p": float(p), "estimate": float(fit.params[name])})
    return {"test": "linear mixed model (random intercept per subject)", "effects": effects}


def draw_subject_means(ax, table, size: float = 70.0):
    """Each subject's mean as a large marker at its group (a colour per subject, the
    same subject the same colour in every group)."""
    from ..plots import styles
    means = subject_means(table)
    subjects = []
    for m in means.values():
        for s in m:
            if s not in subjects:
                subjects.append(s)
    for i, (g, m) in enumerate(means.items()):
        if not m:
            continue
        xs = np.full(len(m), float(i)) + np.linspace(-0.12, 0.12, len(m)) * (len(m) > 1)
        colors = [styles.color(subjects.index(s) % 8) for s in m]
        sc = ax.scatter(xs, list(m.values()), s=size, c=colors, edgecolors="#222222", linewidths=0.9,
                        zorder=8, marker="o")
        sc._staplot_part = "subject_means"
    return subjects
