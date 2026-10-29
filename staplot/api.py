"""UI-free façade over staplot's statistics + plotting engine.

This module is the *embedding* entry point for other tools (e.g. PyFigures) that
want to turn a CSV of measurements into a publication-styled plot panel with
automatic significance brackets — without launching the staplot GUI.

Import safety
-------------
It imports only ``numpy`` / ``scipy`` / ``pandas`` / ``matplotlib`` /
``statsmodels`` and staplot's *model* + *stats* + *plots* subpackages, none of
which touch Qt. ``import staplot.api`` therefore never pulls in a GUI toolkit and
is safe to run headless (e.g. inside a PyFigures ``custom_loading_script``).

Public API
----------
``table_from_csv`` / ``table_from_columns``  -> build a column DataTable
``grouped_table_from_csv``                   -> build a two-factor GroupedDataTable
``auto_test``                                -> pick + run the right test, return
                                                machine-readable stats + sig_pairs
``plot``                                     -> render a Figure (+ stats record),
                                                auto-annotated with brackets

The test-selection logic mirrors what a careful analyst would do by hand:

* **2 groups**   normal -> Welch's t-test  ; else Mann-Whitney U.
* **>2 groups**  normal -> one-way ANOVA + Tukey HSD (family-wise corrected);
                 else Kruskal-Wallis + Dunn's test (Bonferroni-corrected).

Normality is judged per group with Shapiro-Wilk; any group failing drops the
comparison to the non-parametric branch. When normality *cannot* be tested
(some group has n<3) the parametric branch is used and the record says so: rank
tests are powerless at such n (Mann-Whitney 3 vs 3 can never reach p<0.05), so
defaulting to them would silently hide every effect.

For >2 groups, post-hoc brackets are only drawn when the omnibus test
(ANOVA / Kruskal-Wallis) is significant; all pairwise p-values are still
reported in ``comparisons``.
"""
from __future__ import annotations

import itertools
import os

import numpy as np
from scipy import stats as _sp

from .models.project import DataTable, GroupedDataTable
from .stats.descriptive import describe, significance_stars, format_p
from .stats.tests import run_ttest, run_mann_whitney
from . import plots as _plots

__all__ = [
    "table_from_columns",
    "table_from_csv",
    "grouped_table_from_csv",
    "auto_test",
    "plot",
    "PLOT_KINDS",
]

PLOT_KINDS = ("bar", "box", "violin", "scatter", "grouped")


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------
def _to_float_or_none(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def table_from_columns(columns: dict, name: str = "data") -> DataTable:
    """Build a column DataTable from ``{group_name: [values...]}``."""
    t = DataTable(str(name))
    for cname, vals in columns.items():
        col = t.add_column(str(cname))
        col.values = [_to_float_or_none(v) for v in vals]
    return t


def csv_format(sample: str) -> tuple[str, str]:
    """(separator, decimal mark) of a CSV from its first lines: ';' or tab when
    every line has the same number of them (Excel in French, German... writes
    ``1,5;2,5``), with a decimal comma when the numbers use one; else ','."""
    import re
    lines = [ln for ln in sample.splitlines()[:20] if ln.strip()]
    if len(lines) > 1 and not sample.endswith("\n"):
        lines = lines[:-1]                    # the sample may cut the last line
    for sep in (";", "\t"):
        counts = {ln.count(sep) for ln in lines}
        if lines and len(counts) == 1 and counts.pop() > 0:
            return sep, ("," if re.search(r"\d,\d", sample) else ".")
    return ",", "."


def _read_csv(path):
    import pandas as pd
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        sep, decimal = csv_format(f.read(4096))
    return pd.read_csv(path, sep=sep, decimal=decimal, encoding="utf-8-sig")


def table_from_csv(path: str, layout: str = "auto", name: str | None = None) -> DataTable:
    """Build a column DataTable from a CSV.

    *layout*:
      - ``"wide"`` — every column is a group; rows are replicates (Prism-style).
      - ``"long"`` — first column = group label, second column = value.
      - ``"auto"`` — long when there are exactly two columns and the first is
        non-numeric; wide otherwise.
    """
    import pandas as pd
    df = _read_csv(path)
    nm = name or os.path.splitext(os.path.basename(str(path)))[0]
    if df.shape[1] == 0:
        return DataTable(nm)

    if layout == "auto":
        first_numeric = pd.to_numeric(df.iloc[:, 0], errors="coerce").notna().mean()
        layout = "long" if (df.shape[1] == 2 and first_numeric < 0.5) else "wide"

    if layout == "long":
        gcol, vcol = df.columns[0], df.columns[1]
        vals = pd.to_numeric(df[vcol], errors="coerce")
        columns: dict = {}
        for g, v in zip(df[gcol].astype(str), vals):
            columns.setdefault(g, []).append(v)
        return table_from_columns(columns, name=nm)

    # wide
    columns = {}
    for c in df.columns:
        columns[str(c)] = pd.to_numeric(df[c], errors="coerce").tolist()
    return table_from_columns(columns, name=nm)


def grouped_table_from_csv(path: str, name: str | None = None) -> GroupedDataTable:
    """Build a two-factor GroupedDataTable from a *long* CSV with three columns:
    ``factor_a`` (column group), ``factor_b`` (row label), ``value``."""
    import pandas as pd
    df = _read_csv(path)
    nm = name or os.path.splitext(os.path.basename(str(path)))[0]
    if df.shape[1] < 3:
        raise ValueError("grouped plot needs a long CSV with 3 columns: "
                         "factor_a, factor_b, value")
    fa, fb, val = df.columns[0], df.columns[1], df.columns[2]
    values = pd.to_numeric(df[val], errors="coerce")

    def numeric_share(col):
        return pd.to_numeric(df[col], errors="coerce").notna().mean()
    # a wide table (one column per condition) read as long gives a nonsense
    # plot and a degenerate ANOVA instead of an error
    if values.notna().mean() < 0.5 or numeric_share(fa) > 0.5 or numeric_share(fb) > 0.5:
        raise ValueError(
            "grouped plot needs a long CSV: one row per measurement, with columns "
            "factor_a (text), factor_b (text), value (number), e.g. 'Genotype,Day,Length' "
            "then 'WT,Day 1,10.2'; this file's first three columns are "
            f"{', '.join(map(str, df.columns[:3]))}")
    # count replicates per (a, b) so n_reps is large enough for every cell
    cells: dict = {}
    for a, b, v in zip(df[fa].astype(str), df[fb].astype(str), values):
        cells.setdefault((a, b), []).append(_to_float_or_none(v))
    n_reps = max((len(v) for v in cells.values()), default=1)
    t = GroupedDataTable(nm, n_reps=n_reps)
    for a in df[fa].astype(str):
        t.add_col_group(a)
    for b in df[fb].astype(str):
        t.add_row_label(b)
    for (a, b), vs in cells.items():
        for rep, v in enumerate(vs):
            t.set_value(a, b, rep, v)
    return t


# ---------------------------------------------------------------------------
# Automatic test selection
# ---------------------------------------------------------------------------
def _normality_testable(arr: np.ndarray) -> bool:
    """Shapiro-Wilk needs n>=3 and some spread."""
    arr = arr[~np.isnan(arr)]
    return len(arr) >= 3 and float(np.std(arr, ddof=1)) > 0.0


def _min_mann_whitney_p(n1: int, n2: int) -> float:
    """Smallest two-sided exact p-value Mann-Whitney U can produce for n1 vs n2
    (both samples completely separated): 2 / C(n1+n2, n1)."""
    from math import comb
    return min(1.0, 2.0 / comb(n1 + n2, n1))


def _is_normal(arr: np.ndarray) -> bool:
    arr = arr[~np.isnan(arr)]
    if len(arr) < 3:
        return False
    if float(np.std(arr, ddof=1)) == 0.0:
        return False
    try:
        _, p = _sp.shapiro(arr)
    except Exception:
        return False
    return bool(p > 0.05)


def _tukey_pairs(names, arrays):
    """Return [(name_a, name_b, p_adj), ...] from Tukey HSD (already corrected)."""
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    all_vals = np.concatenate(arrays)
    all_labels = np.concatenate([[n] * len(a) for n, a in zip(names, arrays)])
    res = pairwise_tukeyhsd(all_vals, all_labels)
    groups_sorted = list(res.groupsunique)
    pairs = list(itertools.combinations(groups_sorted, 2))
    return [(str(a), str(b), float(p)) for (a, b), p in zip(pairs, res.pvalues)]


def _dunn_pairs(names, arrays):
    """Dunn's test after Kruskal-Wallis, Bonferroni-corrected (see
    stats.anova.dunn_test). Previously this ran separate pairwise Mann-Whitney
    tests — a different procedure from the "Dunn" reported in Methods text."""
    from .stats.anova import dunn_test
    return [(str(names[i]), str(names[j]), float(p)) for i, j, p in dunn_test(arrays)]


def auto_test(table: DataTable, only_significant: bool = True,
              parametric: bool | None = None) -> dict:
    """Pick and run the appropriate comparison for *table*'s groups.

    Returns a JSON-serialisable dict with the chosen test, omnibus result (for
    >2 groups), a per-group descriptive summary, every pairwise comparison, and
    ``sig_pairs`` (``[(a, b, p), ...]``) ready to hand to a plotter. When
    *only_significant* is True (default) ``sig_pairs`` keeps only p<0.05 pairs so
    the figure isn't cluttered with 'ns' brackets; the full set is always in
    ``comparisons``.

    *parametric*: ``None`` (default) chooses parametric vs non-parametric from a
    per-group Shapiro normality check; ``True`` forces the parametric branch
    (t-test / ANOVA), ``False`` forces the non-parametric branch (Mann-Whitney /
    Kruskal).
    """
    groups = table.group_arrays()
    names = [n for n in groups if len(groups[n]) > 0]
    arrays = [np.asarray(groups[n], dtype=float) for n in names]
    k = len(names)

    rec: dict = {
        "n_groups": k,
        "groups": {n: describe(a) for n, a in zip(names, arrays)},
        "test": "none",
        "omnibus": None,
        "correction": None,
        "comparisons": [],
        "sig_pairs": [],
    }
    if k < 2 or any(len(a) < 2 for a in arrays):
        rec["note"] = "Need at least 2 groups with >=2 values each."
        return rec

    warnings: list[str] = []
    if parametric is None:
        if all(_normality_testable(a) for a in arrays):
            normal = all(_is_normal(a) for a in arrays)
            rec["normality"] = "normal" if normal else "non-normal"
        else:
            # can't test normality (n<3 or no spread): a rank test would be
            # powerless here, so assume normality and say so
            normal = True
            rec["normality"] = "untestable (n<3 or constant group); normality assumed"
            warnings.append("Normality could not be tested (a group has n<3 or no "
                            "variance); the parametric test was used.")
    else:
        normal = bool(parametric)
        rec["normality"] = "forced-parametric" if normal else "forced-nonparametric"

    omnibus_sig = True
    if k == 2:
        if normal:
            r = run_ttest(table, names[0], names[1])
            rec["test"] = "Welch's t-test"
        else:
            r = run_mann_whitney(table, names[0], names[1])
            rec["test"] = "Mann-Whitney U test"
            p_floor = _min_mann_whitney_p(len(arrays[0]), len(arrays[1]))
            if p_floor >= 0.05:
                warnings.append(
                    f"With n={len(arrays[0])} vs n={len(arrays[1])}, Mann-Whitney "
                    f"cannot reach p<0.05 (smallest possible p={p_floor:.3g}); "
                    "absence of a bracket is not evidence of no effect.")
        p = float(r.details["p"])
        rec["comparisons"] = [_cmp(names[0], names[1], p)]
    else:
        if normal:
            F, p_omni = _sp.f_oneway(*arrays)
            rec["test"] = "One-way ANOVA + Tukey HSD"
            rec["omnibus"] = {"stat_name": "F", "stat": float(F), "p": float(p_omni)}
            rec["correction"] = "Tukey HSD (family-wise)"
            pairs = _tukey_pairs(names, arrays)
        else:
            H, p_omni = _sp.kruskal(*arrays)
            rec["test"] = "Kruskal-Wallis + Dunn's test"
            rec["omnibus"] = {"stat_name": "H", "stat": float(H), "p": float(p_omni)}
            rec["correction"] = "Bonferroni"
            pairs = _dunn_pairs(names, arrays)
        rec["comparisons"] = [_cmp(a, b, p) for a, b, p in pairs]
        omnibus_sig = bool(np.isfinite(p_omni) and p_omni < 0.05)
        rec["posthoc_shown"] = omnibus_sig
        if not omnibus_sig:
            warnings.append("Omnibus test not significant (p="
                            f"{format_p(float(p_omni))}); post-hoc brackets are not drawn.")

    sig = [(c["a"], c["b"], c["p"]) for c in rec["comparisons"]
           if omnibus_sig and not np.isnan(c["p"])
           and (c["p"] < 0.05 or not only_significant)]
    rec["sig_pairs"] = sig
    if warnings:
        rec["warnings"] = warnings
    return rec


def _cmp(a, b, p):
    return {"a": str(a), "b": str(b), "p": float(p),
            "stars": significance_stars(p), "p_text": format_p(p)}


def two_way_test(gtable: GroupedDataTable) -> dict:
    """Run a two-way ANOVA on a GroupedDataTable and return the main effects and
    interaction as a machine-readable record (for grouped-bar panels)."""
    from .stats.two_way_anova import run_two_way_anova
    rec: dict = {"test": "Two-way ANOVA", "effects": []}
    try:
        r = run_two_way_anova(gtable)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec
    d = r.details
    for key, lbl in (("fa_p", "factor A"), ("fb_p", "factor B"),
                     ("interaction_p", "A x B")):
        p = d.get(key)
        pf = None if (p is None or (isinstance(p, float) and np.isnan(p))) else float(p)
        rec["effects"].append({"effect": lbl, "p": pf,
                               "stars": significance_stars(pf) if pf is not None else "n/a"})
    return rec


def correlation_test(table: DataTable, method: str = "pearson") -> dict:
    """Correlate the first two numeric columns of *table* (x, y) — for scatter
    panels. Returns r, r^2, p, n."""
    from .stats.correlation import run_correlation
    names = [c.name for c in table.columns if len(c.valid_values()) > 0]
    rec: dict = {"test": f"{method.capitalize()} correlation"}
    if len(names) < 2:
        rec["error"] = "need at least 2 columns (x, y)"
        return rec
    try:
        r = run_correlation(table, names[0], names[1], method=method)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec
    d = r.details
    rec.update({"x": names[0], "y": names[1], "r": d["r"], "r2": d["r2"],
                "p": d["p"], "n": d["n"], "stars": significance_stars(d["p"])})
    return rec


def _annotate_two_way(fig, rec):
    parts = []
    for e in rec.get("effects", []):
        if e["p"] is None:
            continue
        tail = "" if e["stars"] in ("ns", "n/a") else " " + e["stars"]
        parts.append(f"{e['effect']}: p={e['p']:.3g}{tail}")
    if parts:
        fig.text(0.5, 0.005, "Two-way ANOVA — " + "; ".join(parts),
                 ha="center", va="bottom", fontsize=7, color="#444444")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
_STATS_MODE = {"auto": None, "parametric": True, "nonparametric": False}


@_plots.styles.styled  # the two-way ANOVA note is added after the plot is built
def plot(table, kind: str = "bar", *, error: str = "sem", annotate: bool = True,
         only_significant: bool = True, stats_mode: str = "auto", title: str = "",
         ylabel: str = "", xlabel: str = "", colors=None, show_title: bool = True):
    """Render *table* to a matplotlib Figure and return ``(fig, stats_record)``.

    *kind*: ``"bar"`` | ``"box"`` | ``"violin"`` | ``"scatter"`` | ``"grouped"``.
    When *annotate* is set:
      * bar/box/violin — groups compared with :func:`auto_test`; significant
        pairs drawn as brackets.
      * grouped — two-way ANOVA (:func:`two_way_test`); main effects + interaction
        printed under the axes.
      * scatter — correlation of the first two columns (:func:`correlation_test`);
        R^2/p already shown in the plot legend.

    *stats_mode*: ``"auto"`` (normality-based), ``"parametric"``, or
    ``"nonparametric"`` — forces the test family for group comparisons.

    *show_title*: False draws no title unless *title* is given (otherwise the
    table name is used) — journals rarely want titles on figure panels.
    """
    kind = kind.lower()
    if kind not in PLOT_KINDS:
        raise ValueError(f"unknown plot kind {kind!r}; expected one of {PLOT_KINDS}")
    parametric = _STATS_MODE.get(stats_mode, None)

    stats_record = None
    sig_pairs = None
    if annotate and kind in ("bar", "box", "violin"):
        stats_record = auto_test(table, only_significant=only_significant,
                                 parametric=parametric)
        sig_pairs = stats_record["sig_pairs"] or None

    if kind == "bar":
        fig = _plots.plot_bar(table, error_type=error, title=title, ylabel=ylabel,
                              colors=colors, sig_pairs=sig_pairs)
    elif kind in ("box", "violin"):
        fig = _plots.plot_box(table, plot_type=kind, title=title, ylabel=ylabel,
                              colors=colors, sig_pairs=sig_pairs)
    elif kind == "scatter":
        fig = _plots.plot_scatter(table, title=title, xlabel=xlabel, ylabel=ylabel,
                                  colors=colors)
        if annotate:
            stats_record = correlation_test(table)
    elif kind == "grouped":
        fig = _plots.plot_grouped_bar(table, error_type=error, title=title,
                                      ylabel=ylabel, colors=colors)
        if annotate:
            stats_record = two_way_test(table)
            if "error" not in stats_record:
                _annotate_two_way(fig, stats_record)
    if not show_title and not title:
        for ax in fig.axes:
            ax.set_title("")
        fig.tight_layout()  # give the title's room back to the axes
    return fig, stats_record
