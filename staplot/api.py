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


EXCEL_EXTENSIONS = (".xlsx", ".xlsm", ".xls")
SQLITE_EXTENSIONS = (".sqlite", ".sqlite3", ".db", ".db3")
SHEET_SEP = "::"


def split_sheet(path: str) -> tuple[str, str | None]:
    """("data.xlsx", "Sheet2") for "data.xlsx::Sheet2"; (path, None) otherwise.
    A sheet is named by its name or its number from 1. For an SQLite database
    ("cells.db::measurements"), the part after :: is a table (or view) name, its
    number from 1, or an SQL query ("cells.db::SELECT genotype, area FROM cells")."""
    path = str(path)
    if SHEET_SEP in path:
        file, sheet = path.split(SHEET_SEP, 1)
        if file.lower().endswith(EXCEL_EXTENSIONS + SQLITE_EXTENSIONS) or _sqlite_header(file):
            return file, (sheet or None)
    return path, None


def _sqlite_header(file: str) -> bool:
    """The file starts with SQLite's magic header (a database under any name)."""
    try:
        with open(file, "rb") as f:
            return f.read(16) == b"SQLite format 3\x00"
    except (OSError, ValueError):
        return False


def is_sqlite(path: str) -> bool:
    """An SQLite database, by its header: a ".db" may be something else, and a
    database may have any name."""
    return _sqlite_header(split_sheet(path)[0])


def _sqlite_tables(file: str) -> list[str]:
    import sqlite3
    con = sqlite3.connect(f"file:{file}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view') "
                           "AND name NOT LIKE 'sqlite_%' ORDER BY type, rowid").fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]


def _sqlite_query(file: str, sheet: str | None) -> str:
    """The SQL that reads the table asked for: a query as it is, a table or view by
    name (any case) or number from 1, else the first table."""
    names = _sqlite_tables(file)
    if sheet and sheet.strip().lower().startswith(("select ", "with ")):
        return sheet.strip()
    if not names:
        raise ValueError(f"{os.path.basename(file)} holds no table")
    if sheet is None:
        chosen = names[0]
    else:
        low = {n.lower(): n for n in names}
        if sheet.strip().lower() in low:
            chosen = low[sheet.strip().lower()]
        elif sheet.strip().isdigit() and 1 <= int(sheet) <= len(names):
            chosen = names[int(sheet) - 1]
        else:
            raise ValueError(f"no table {sheet!r} in {os.path.basename(file)}; its tables are "
                             f"{', '.join(names)} (or give an SQL query: SELECT ...)")
    return 'SELECT * FROM "' + chosen.replace('"', '""') + '"' 


def is_excel(path: str) -> bool:
    return split_sheet(path)[0].lower().endswith(EXCEL_EXTENSIONS)


def _excel_engine(file: str) -> str:
    """openpyxl for .xlsx/.xlsm, xlrd for the old .xls; a clear message when missing."""
    old = file.lower().endswith(".xls")
    module = "xlrd" if old else "openpyxl"
    try:
        __import__(module)
    except ImportError:
        raise ImportError(f"reading {os.path.basename(file)} needs the '{module}' package "
                          f"(pip install {module}), or save the sheet as CSV") from None
    return module


def list_sheets(path: str) -> list[str]:
    """The sheet names of an Excel workbook, the tables and views of an SQLite
    database ([] for a text table)."""
    file, _ = split_sheet(path)
    if is_sqlite(file):
        return _sqlite_tables(file)
    if not is_excel(file):
        return []
    import pandas as pd
    with pd.ExcelFile(file, engine=_excel_engine(file)) as book:
        return [str(s) for s in book.sheet_names]


def read_table(path: str):
    """A table as a DataFrame: CSV / TSV / TXT (separator and decimal mark detected)
    or an Excel sheet (.xlsx, .xlsm, .xls; "book.xlsx::Sheet2" picks a sheet, by name
    or number from 1; the first one otherwise). Rows and columns left wholly empty in
    the sheet are dropped (a sheet often has a blank margin)."""
    import pandas as pd
    file, sheet = split_sheet(path)
    if is_sqlite(file):
        # an SQLite database (user, 2026-11-18: "I very often use SQLite DBs"), read only
        import sqlite3
        con = sqlite3.connect(f"file:{file}?mode=ro", uri=True)
        try:
            df = pd.read_sql_query(_sqlite_query(file, sheet), con)
        finally:
            con.close()
        return _numbers_from_text(df, decimal_comma=False)
    if not is_excel(file):
        with open(file, encoding="utf-8-sig", errors="replace", newline="") as f:
            sep, decimal = csv_format(f.read(4096))
        df = pd.read_csv(file, sep=sep, decimal=decimal, encoding="utf-8-sig")
        # one typo ("abc") in a decimal-comma column keeps the whole column as text,
        # and pandas then leaves its "41,2" unconverted: the group vanished
        return _numbers_from_text(df, decimal_comma=(decimal == ","))
    engine = _excel_engine(file)
    with pd.ExcelFile(file, engine=engine) as book:
        names = [str(s) for s in book.sheet_names]
        if sheet is None:
            chosen = names[0]
        elif sheet in names:
            chosen = sheet
        elif sheet.strip().isdigit() and 1 <= int(sheet) <= len(names):
            chosen = names[int(sheet) - 1]
        else:
            raise ValueError(f"no sheet {sheet!r} in {os.path.basename(file)}; "
                             f"its sheets are {', '.join(names)}")
        raw = book.parse(chosen, header=None)
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    if raw.empty:
        return pd.DataFrame()
    header = [str(v).strip() if not (isinstance(v, float) and v != v) else f"column {i + 1}"
              for i, v in enumerate(raw.iloc[0])]
    df = raw.iloc[1:].reset_index(drop=True)
    df.columns = header
    # Excel cells typed as text ("12", "3,5") are numbers to a CSV reader too
    return _numbers_from_text(df, decimal_comma=True)


def _numbers_from_text(df, decimal_comma: bool):
    """Text columns that are mostly numbers become numbers (cells that are not, like a
    typo 'abc', stay as they are: the cleaning report names them). A decimal comma is
    read only where the table uses one (else "1,234" could be a thousands mark)."""
    import pandas as pd
    for col in df.columns:
        # text columns: object, or pandas 3's string dtype
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            continue
        text = df[col].astype(str).str.strip()
        if decimal_comma:
            text = text.str.replace(",", ".", regex=False)
        converted = pd.to_numeric(text.where(df[col].notna()), errors="coerce")
        if converted.notna().sum() >= max(1, df[col].notna().sum() // 2):
            df[col] = converted.where(converted.notna(), df[col])
    return df.infer_objects()


def _read_csv(path):
    return read_table(path)


def _detect_layout(df, layout: str = "auto") -> str:
    import pandas as pd
    if layout != "auto":
        return layout
    first_numeric = pd.to_numeric(df.iloc[:, 0], errors="coerce").notna().mean()
    return "long" if (df.shape[1] == 2 and first_numeric < 0.5) else "wide"


def _has_label(value) -> bool:
    import pandas as pd
    return not pd.isna(value) and str(value).strip() != ""


def csv_cleaning_report(path: str, layout: str = "auto") -> dict:
    """What reading *path* for a plot keeps and leaves out, so it can be told to the
    reader (or to a language model asked "what did you do with the empty cells?").

    Returns ``{"layout", "groups": {group: {"used", "missing", "non_numeric",
    "non_numeric_examples"}}, "rows_without_group", "summary"}``. *missing* counts empty
    cells and NA / nan / n/a markers; *non_numeric* counts text that is not a number
    (often a typo: "4O" for "40", "abc"). Both are left out of the plot and the
    statistics, group by group; nothing is imputed. In a long table a row with no group
    label is left out too (it is not a group of its own)."""
    import pandas as pd
    df = _read_csv(path)
    out = {"layout": None, "groups": {}, "rows_without_group": 0, "summary": "empty table"}
    if df.shape[1] == 0:
        return out
    layout = _detect_layout(df, layout)
    out["layout"] = layout

    def tally(raw):
        missing = int(raw.isna().sum())
        numeric = pd.to_numeric(raw, errors="coerce")
        bad = raw[raw.notna() & numeric.isna()]
        return {"used": int(numeric.notna().sum()), "missing": missing, "non_numeric": int(len(bad)),
                "non_numeric_examples": [str(v) for v in bad.astype(str).unique()[:5]]}

    if layout == "long":
        gcol, vcol = df.columns[0], df.columns[1]
        labelled = df[gcol].map(_has_label)
        out["rows_without_group"] = int((~labelled).sum())
        for g, sub in df[labelled].groupby(df[gcol][labelled].astype(str).str.strip(), sort=False):
            out["groups"][str(g)] = tally(sub[vcol])
    else:
        for c in df.columns:
            out["groups"][str(c)] = tally(df[c])
    parts = []
    for g, t in out["groups"].items():
        left = []
        if t["missing"]:
            left.append(f"{t['missing']} empty/NA")
        if t["non_numeric"]:
            left.append(f"{t['non_numeric']} not a number ({', '.join(repr(x) for x in t['non_numeric_examples'])})")
        parts.append(f"{g}: n={t['used']}" + (f" ({'; '.join(left)} left out)" if left else ""))
    if out["rows_without_group"]:
        parts.append(f"{out['rows_without_group']} row(s) without a group label left out")
    total_left = sum(t["missing"] + t["non_numeric"] for t in out["groups"].values()) + out["rows_without_group"]
    out["summary"] = ("; ".join(parts) + ("" if total_left else " — no value left out")
                      + (". Left-out cells are skipped per group, never filled in." if total_left else "."))
    return out


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

    layout = _detect_layout(df, layout)

    if layout == "long":
        gcol, vcol = df.columns[0], df.columns[1]
        vals = pd.to_numeric(df[vcol], errors="coerce")
        columns: dict = {}
        for g, v in zip(df[gcol], vals):
            # a row without a group label is not a group named "nan" (it was plotted as one)
            if not _has_label(g):
                continue
            columns.setdefault(str(g).strip(), []).append(v)
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
# Automatic plot kind
# ---------------------------------------------------------------------------

# n per group (the smallest group) from which a box, then a violin, shows the data best
BOX_FROM_N = 5
VIOLIN_FROM_N = 20
_XY_NAMES = ({"x", "y"}, {"time", "value"}, {"dose", "response"}, {"concentration", "signal"})


def _looks_two_factor(df) -> bool:
    import pandas as pd
    if df.shape[1] != 3:
        return False
    share = [pd.to_numeric(df.iloc[:, i], errors="coerce").notna().mean() for i in range(3)]
    return share[0] < 0.5 and share[1] < 0.5 and share[2] >= 0.5


def _unit(name) -> str | None:
    """The unit of a column name: "area (um2)" -> "um2", "length [µm]" -> "µm"."""
    import re
    m = re.search(r"[(\[]([^)\]]+)[)\]]\s*$", str(name))
    return m.group(1).strip().lower() if m else None


def _looks_xy(df) -> bool:
    """Two numeric columns that are one series (x, y), not two groups: named x / y
    (time, dose…), in different units (area (um2), intensity (a.u.)), or the first
    column a strictly increasing series. Two groups in the same unit stay groups, even
    when they correlate (before / after measures do)."""
    import pandas as pd
    if df.shape[1] != 2:
        return False
    x = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    if x.notna().mean() < 0.9 or y.notna().mean() < 0.9:
        return False
    names = {str(c).strip().lower() for c in df.columns}
    if any(pair <= names for pair in _XY_NAMES) or any(" vs " in n for n in names):
        return True
    units = [_unit(c) for c in df.columns]
    if all(units) and units[0] != units[1]:
        return True
    x = x.dropna()
    return len(x) >= 5 and bool((x.diff().dropna() > 0).all())


def suggest_kind(source) -> dict:
    """The plot that shows a table best, and why (``kind="auto"``):

    * three columns, two of text and one of numbers -> ``grouped`` (two factors);
    * two numeric columns making one x/y series (named x / y, time, dose…, or the
      first column strictly increasing) -> ``scatter``;
    * groups -> by the smallest group's n: under 5 points ``bar`` (mean +- SD with
      every point: a box or a violin of 3 points says nothing), 5 to 19 ``box``,
      20 or more ``violin`` (enough points for a shape).

    *source*: a table path (CSV, TXT, Excel ``book.xlsx::sheet``) or a DataTable /
    GroupedDataTable. Returns ``{"kind", "reason", "n_min"}``."""
    if isinstance(source, GroupedDataTable):
        return {"kind": "grouped", "reason": "a two-factor table", "n_min": None}
    if isinstance(source, str):
        df = read_table(source)
        if _looks_two_factor(df):
            return {"kind": "grouped", "n_min": None,
                    "reason": f"two factors ({df.columns[0]}, {df.columns[1]}) and a value ({df.columns[2]})"}
        if _looks_xy(df):
            return {"kind": "scatter", "n_min": None,
                    "reason": f"one series, {df.columns[1]} against {df.columns[0]}"}
        import pandas as pd
        text = [str(c) for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().mean() < 0.5]
        if text and _detect_layout(df) == "wide":
            numeric = [str(c) for c in df.columns if str(c) not in text]
            if not numeric:
                return {"kind": None, "n_min": None, "reason": "no column of numbers to plot"}
            return {"kind": None, "n_min": None,
                    "reason": (f"a tidy table with factors ({', '.join(text)}) and measures "
                               f"({', '.join(numeric)}): say which measure against which factor "
                               f"(in PyFigures: a seaborn plot, e.g. catplot)")}
        table = table_from_csv(source)
    else:
        table = source
    ns = [len(c.valid_values()) for c in table.columns]
    ns = [n for n in ns if n] or [0]
    n_min = min(ns)
    if n_min < BOX_FROM_N:
        kind, why = "bar", f"few points per group (n = {n_min}): mean +- SD with every point shown"
    elif n_min < VIOLIN_FROM_N:
        span = f"{n_min}" if n_min == max(ns) else f"{n_min} to {max(ns)}"
        kind, why = "box", f"n = {span} per group: median and quartiles, every point shown"
    else:
        kind, why = "violin", f"n >= {n_min} per group: enough points to show each distribution's shape"
    return {"kind": kind, "reason": why, "n_min": n_min}


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
        # one effect per line: on one line it was wider than the plot, and a panel sized
        # to it squeezed the bars (PyFigures, 2026-11-18)
        fig.text(0.5, 0.005, "Two-way ANOVA\n" + "\n".join(parts),
                 ha="center", va="bottom", fontsize=7, color="#444444", linespacing=1.25)
        fig._staplot_note_lines = len(parts) + 1


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
_STATS_MODE = {"auto": None, "parametric": True, "nonparametric": False}


@_plots.styles.styled  # the two-way ANOVA note is added after the plot is built
def plot(table, kind: str = "bar", *, error: str = "sd", annotate: bool = True,
         only_significant: bool = True, stats_mode: str = "auto", title: str = "",
         ylabel: str = "", xlabel: str = "", colors=None, show_title: bool = True,
         aspect: float | None = None, width: float | None = None, height: float | None = None,
         font_scale: float | None = None, order=None, ylim=None, log_y: bool = False,
         xtick_rotation: float | None = None, legend: str | None = None,
         show_ylabel: bool = True, show_xlabel: bool = True, rename: dict | None = None,
         show_points: bool = True, violin_inner: str = "lines"):
    """Render *table* to a matplotlib Figure and return ``(fig, stats_record)``.

    *kind*: ``"bar"`` | ``"box"`` | ``"violin"`` | ``"scatter"`` | ``"grouped"``, or
    ``"auto"``: the one that shows the data best (:func:`suggest_kind`).
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

    Layout and size (all optional; the default figure is unchanged):
      * *aspect* — width / height ("wider" 1.5-2, "taller" 0.6-0.8);
      * *width*, *height* — the figure size in inches; a smaller figure makes the
        text and lines relatively bigger once the plot is scaled into a panel;
      * *font_scale* — every text times this (1.3 = 30 % bigger);
      * *order* — the groups in this order (names; groups left out are dropped);
      * *ylim* — ``(low, high)`` of the y axis (``None`` for either keeps it automatic);
      * *log_y* — a log y axis;
      * *xtick_rotation* — degrees for the x tick labels (long group names),
        anticlockwise as matplotlib counts: 45 tilts them "/" (rotated left, the usual
        way), -45 tilts them "\\" (rotated right);
      * *rename* — ``{"mutA": "mutant A"}``: the names shown for groups (axis, legend
        and the statistics), the table unchanged;
      * *show_points* — False: no individual points on bar / box / violin plots;
      * *violin_inner* — inside each violin: ``"lines"`` (range, quartiles, median
        dot), ``"box"`` (a mini box plot) or ``"none"``;
      * *legend* — ``"inside"``, ``"outside"`` (right of the axes) or ``"none"``;
      * *show_ylabel* / *show_xlabel* — False: no axis label at all (an empty
        *ylabel* means the automatic one, "Value" or the column's name).
    """
    kind = kind.lower()
    if rename and isinstance(table, DataTable):
        table = _renamed(table, rename)
        order = [rename.get(o, o) for o in order] if order else order
    if order and isinstance(table, DataTable):
        table = _reordered(table, order)
    if kind == "auto":
        kind = suggest_kind(table)["kind"]
    if kind not in PLOT_KINDS:
        raise ValueError(f"unknown plot kind {kind!r}; expected one of {PLOT_KINDS} or 'auto'")
    parametric = _STATS_MODE.get(stats_mode, None)

    stats_record = None
    sig_pairs = None
    if annotate and kind in ("bar", "box", "violin"):
        stats_record = auto_test(table, only_significant=only_significant,
                                 parametric=parametric)
        sig_pairs = stats_record["sig_pairs"] or None

    if violin_inner not in ("lines", "box", "none"):
        raise ValueError("violin_inner must be 'lines', 'box' or 'none'")
    if kind == "bar":
        fig = _plots.plot_bar(table, error_type=error, title=title, ylabel=ylabel,
                              colors=colors, sig_pairs=sig_pairs, show_points=show_points)
    elif kind in ("box", "violin"):
        fig = _plots.plot_box(table, plot_type=kind, title=title, ylabel=ylabel,
                              colors=colors, sig_pairs=sig_pairs, show_points=show_points,
                              violin_inner=violin_inner)
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
    if aspect:
        # width / height of the plot: in a figure row a plot is as tall as its
        # neighbours, so a wider plot takes more of the row (user: "make k plot wider")
        _w, h = fig.get_size_inches()
        fig.set_size_inches(max(0.2, float(aspect)) * h, h)
        fig.tight_layout()
    if not show_ylabel or not show_xlabel:
        # "I want the label of the y axis gone": not the automatic "Value" instead
        for ax in fig.axes[:1]:
            if not show_ylabel:
                ax.set_ylabel("")
            if not show_xlabel:
                ax.set_xlabel("")
        fig.tight_layout()
    if rename and not isinstance(table, DataTable):
        # a grouped table: its factor levels as shown (ticks and legend)
        for ax in fig.axes:
            names = [t.get_text() for t in ax.get_xticklabels()]
            ax.set_xticks(ax.get_xticks(), [str(rename.get(n, n)) for n in names])
            for t in (ax.get_legend().get_texts() if ax.get_legend() else []):
                if t.get_text() in rename:
                    t.set_text(str(rename[t.get_text()]))
    _apply_layout(fig, width=width, height=height, font_scale=font_scale, ylim=ylim,
                  log_y=log_y, xtick_rotation=xtick_rotation, legend=legend)
    if kind == "grouped" and annotate and stats_record and "error" not in stats_record:
        # room for the ANOVA line under the axis labels (they ran into each other)
        lines = getattr(fig, "_staplot_note_lines", 1)
        fig.tight_layout(rect=(0, min(0.4, 0.012 + 0.026 * lines * 4 / max(fig.get_size_inches()[1], 1)), 1, 1))
    return fig, stats_record


def _renamed(table, rename):
    """A copy of *table* whose groups named in *rename* (case-insensitive) show the new
    names; unknown names raise."""
    import copy
    by_name = {c.name.lower(): c for c in table.columns}
    missing = [o for o in rename if str(o).lower() not in by_name]
    if missing:
        raise ValueError(f"no group named {missing}; groups: {[c.name for c in table.columns]}")
    new = {str(k).lower(): str(v) for k, v in rename.items()}
    out = copy.copy(table)
    out.columns = []
    for c in table.columns:
        c2 = copy.copy(c)
        c2.name = new.get(c.name.lower(), c.name)
        out.columns.append(c2)
    return out


def _reordered(table, order):
    """A copy of *table* with its groups in *order* (names, case-insensitive);
    unknown names raise, groups left out are dropped."""
    import copy
    by_name = {c.name.lower(): c for c in table.columns}
    missing = [o for o in order if str(o).lower() not in by_name]
    if missing:
        raise ValueError(f"no group named {missing}; groups: {[c.name for c in table.columns]}")
    out = copy.copy(table)
    out.columns = [by_name[str(o).lower()] for o in order]
    return out


LEGEND_PLACES = ("inside", "outside", "none")


def _apply_layout(fig, *, width=None, height=None, font_scale=None, ylim=None, log_y=False,
                  xtick_rotation=None, legend=None):
    """The size and layout options of :func:`plot`, on a drawn figure."""
    if not any((width, height, font_scale, ylim, log_y, xtick_rotation is not None, legend)):
        return
    w0, h0 = fig.get_size_inches()
    if width or height:
        w = float(width) if width else w0 * (float(height) / h0)     # one given: keep the shape
        h = float(height) if height else h0 * (float(width) / w0)
        fig.set_size_inches(max(0.5, w), max(0.5, h))
    if font_scale:
        for text in fig.findobj(lambda o: hasattr(o, "get_fontsize") and hasattr(o, "set_fontsize")):
            try:
                text.set_fontsize(text.get_fontsize() * float(font_scale))
            except (TypeError, ValueError):
                pass
    for ax in fig.axes:
        if log_y:
            ax.set_yscale("log")
        if ylim:
            lo, hi = (list(ylim) + [None, None])[:2]
            cur_lo, cur_hi = ax.get_ylim()
            ax.set_ylim(cur_lo if lo is None else float(lo), cur_hi if hi is None else float(hi))
        if xtick_rotation is not None:
            for t in ax.get_xticklabels():
                angle = float(xtick_rotation) % 180
                t.set_rotation(float(xtick_rotation))
                # the end of the text at its tick: "/" ends on the right, "\\" on the left
                t.set_ha("center" if angle in (0, 90) else "right" if angle < 90 else "left")
                t.set_rotation_mode("anchor")
        if legend:
            if legend not in LEGEND_PLACES:
                raise ValueError(f"legend must be one of {LEGEND_PLACES}")
            old = ax.get_legend()
            if old is not None:
                if legend == "none":
                    old.remove()
                else:
                    handles, labels = ax.get_legend_handles_labels()
                    title = old.get_title().get_text()
                    size = old.get_texts()[0].get_fontsize() if old.get_texts() else None
                    old.remove()
                    if legend == "outside":
                        ax.legend(handles, labels, title=title or None, fontsize=size, frameon=False,
                                  loc="upper left", bbox_to_anchor=(1.0, 1.0))
                    else:
                        ax.legend(handles, labels, title=title or None, fontsize=size, loc="best")
    fig.tight_layout()


def plot_code(path: str, kind: str = "bar", *, error: str = "sd", annotate: bool = True,
              only_significant: bool = True, stats_mode: str = "auto", title: str = "",
              ylabel: str = "", xlabel: str = "", colors=None, show_title: bool = False, aspect: float | None = None,
              width: float | None = None, height: float | None = None,
              font_scale: float | None = None, order=None, ylim=None, log_y: bool = False,
              xtick_rotation: float | None = None, legend: str | None = None,
              show_ylabel: bool = True, show_xlabel: bool = True, rename: dict | None = None,
              show_points: bool = True, violin_inner: str = "lines") -> str:
    """The Python code of a plot (user: "can staplot print the code it used to generate
    the fig"): a runnable script whose first part redraws the figure with staplot, and
    whose second part reads the table with pandas as staplot does (separator, decimal
    comma, Excel sheet, blank margin, numbers typed as text, empty / NA / typo cells
    left out) and runs the same statistics with scipy / statsmodels alone, so a
    reader or reviewer can check every step without staplot."""
    from . import __version__
    if kind == "auto":
        auto = suggest_kind(path)
        if not auto["kind"]:
            raise ValueError(auto["reason"])
        kind = auto["kind"]
    file, sheet = split_sheet(path)
    path = os.path.abspath(file) + (SHEET_SEP + sheet if sheet else "")    # runs from any folder
    args = [f"kind={kind!r}"]
    for name, value, default in (("error", error, "sd"), ("annotate", annotate, True),
                                 ("only_significant", only_significant, True),
                                 ("stats_mode", stats_mode, "auto"), ("title", title, ""),
                                 ("ylabel", ylabel, ""), ("xlabel", xlabel, ""),
                                 ("colors", list(colors) if colors else None, None),
                                 ("show_title", show_title, True), ("aspect", aspect, None),
                                 ("width", width, None), ("height", height, None),
                                 ("font_scale", font_scale, None),
                                 ("order", list(order) if order else None, None),
                                 ("ylim", list(ylim) if ylim else None, None), ("log_y", log_y, False),
                                 ("xtick_rotation", xtick_rotation, None), ("legend", legend, None),
                                 ("show_ylabel", show_ylabel, True), ("show_xlabel", show_xlabel, True),
                                 ("rename", dict(rename) if rename else None, None),
                                 ("show_points", show_points, True), ("violin_inner", violin_inner, "lines")):
        if value != default:
            args.append(f"{name}={value!r}")
    builder = "grouped_table_from_csv" if kind == "grouped" else "table_from_csv"
    out = [f"# Made by staplot {__version__} from {os.path.basename(split_sheet(path)[0])}"
           + (f" (sheet {split_sheet(path)[1]})" if split_sheet(path)[1] else "") + ".",
           "# Part 1 redraws the figure; part 2 runs the statistics without staplot.",
           "",
           "# --- 1. the figure " + "-" * 60,
           "from staplot import api",
           "",
           f"table = api.{builder}(r{str(path)!r})",
           "fig, stats_record = api.plot(table, " + ", ".join(args) + ")",
           'fig.savefig("plot.pdf")             # vector; or .png / .svg',
           "print(stats_record)",
           ""]
    if not annotate:
        return "\n".join(out)
    out += ["# --- 2. the statistics with scipy / statsmodels alone " + "-" * 24]
    if kind == "grouped":
        out += _grouped_code(path)
        return "\n".join(out)
    table = table_from_csv(path)
    # the data come from the table itself, read and cleaned in the script as staplot does
    out +=["import numpy as np", "import pandas as pd", "from scipy import stats", ""]
    out += _loading_code(path)
    layout = _detect_layout(read_table(path))
    if layout == "long":
        out += ["# long table: first column = group, second = value; rows without a group are skipped",
                "group_col, value_col = df.columns[0], df.columns[1]",
                "df = df[df[group_col].notna() & (df[group_col].astype(str).str.strip() != '')]",
                "groups = {}",
                "for g, v in zip(df[group_col].astype(str).str.strip(), pd.to_numeric(df[value_col], errors='coerce')):",
                "    groups.setdefault(g, []).append(v)",
                "groups = {g: [v for v in vs if np.isfinite(v)] for g, vs in groups.items()}"]
    else:
        out += ["# wide table: one column per group; empty, NA and non-numeric cells are left out",
                "groups = {str(c): pd.to_numeric(df[c], errors='coerce').dropna().tolist() for c in df.columns}"]
    out += ["groups = {g: v for g, v in groups.items() if v}"]
    if order:
        out += ["# the plot's group order (groups left out are not plotted nor tested)",
                "_by_name = {g.lower(): g for g in groups}",
                f"groups = {{_by_name[o.lower()]: groups[_by_name[o.lower()]] for o in {[str(o) for o in order]!r}}}"]
    out += ["names = list(groups)",
            "arrays = [np.asarray(groups[n], float) for n in names]",
            'print({n: len(a) for n, a in zip(names, arrays)})      # n per group', ""]
    if kind == "scatter":
        out += ["# Pearson correlation of the first two columns (x, y), on the rows where both are numbers",
                "x = pd.to_numeric(df.iloc[:, 0], errors='coerce')",
                "y = pd.to_numeric(df.iloc[:, 1], errors='coerce')",
                "both = x.notna() & y.notna()",
                "r, p = stats.pearsonr(x[both], y[both])",
                'print(f"r = {r:.3f}, R² = {r * r:.3f}, p = {p:.3g}, n = {int(both.sum())}")']
        return "\n".join(out)
    if order:
        table = _reordered(table, order)
    rec = auto_test(table, only_significant=only_significant,
                    parametric=_STATS_MODE.get(stats_mode))
    test = str(rec.get("test", ""))
    out.append(f"# staplot chose: {test}" + (f" ({rec['normality']})" if rec.get("normality") else ""))
    if test.startswith("Welch"):
        out += ["t, p = stats.ttest_ind(arrays[0], arrays[1], equal_var=False)",
                'print(f"{names[0]} vs {names[1]}: Welch t = {t:.3f}, p = {p:.3g}")']
    elif test.startswith("Mann"):
        out += ["u, p = stats.mannwhitneyu(arrays[0], arrays[1], alternative='two-sided')",
                'print(f"{names[0]} vs {names[1]}: Mann-Whitney U = {u:.1f}, p = {p:.3g}")']
    elif test.startswith("One-way"):
        out += ["f, p = stats.f_oneway(*arrays)",
                'print(f"one-way ANOVA: F = {f:.3f}, p = {p:.3g}")',
                "from statsmodels.stats.multicomp import pairwise_tukeyhsd",
                "values = np.concatenate(arrays)",
                "labels = np.concatenate([[n] * len(a) for n, a in zip(names, arrays)])",
                "print(pairwise_tukeyhsd(values, labels))      # p-adj: the brackets' p-values"]
    elif test.startswith("Kruskal"):
        out += ["h, p = stats.kruskal(*arrays)",
                'print(f"Kruskal-Wallis: H = {h:.3f}, p = {p:.3g}")',
                "# Dunn's test (1964) on the pooled ranks, tie-corrected, Bonferroni-corrected",
                "pooled = np.concatenate(arrays)",
                "ranks = stats.rankdata(pooled)",
                "bounds = np.cumsum([0] + [len(a) for a in arrays])",
                "mean_ranks = [ranks[bounds[k]:bounds[k + 1]].mean() for k in range(len(arrays))]",
                "N = len(pooled)",
                "_, tie_counts = np.unique(pooled, return_counts=True)",
                "variance = N * (N + 1) / 12 - np.sum(tie_counts ** 3 - tie_counts) / (12 * (N - 1))",
                "n_pairs = len(arrays) * (len(arrays) - 1) // 2",
                "for i in range(len(arrays)):",
                "    for j in range(i + 1, len(arrays)):",
                "        z = abs(mean_ranks[i] - mean_ranks[j]) / np.sqrt(variance * (1 / len(arrays[i]) + 1 / len(arrays[j])))",
                "        p_adj = min(1.0, 2 * stats.norm.sf(z) * n_pairs)",
                '        print(f"{names[i]} vs {names[j]}: Dunn z = {z:.3f}, p (Bonferroni) = {p_adj:.3g}")']
    else:
        out.append(f"# {test or 'no test'}: nothing to compare")
    return "\n".join(out)


def _loading_code(path: str) -> list[str]:
    """pandas code that reads *path* into ``df`` as read_table does: the separator and
    decimal mark of a text table, or an Excel sheet without its blank margin and with
    numbers typed as text read as numbers."""
    file, sheet = split_sheet(path)
    if is_sqlite(file):
        return ["# the table: from the SQLite database, as staplot reads it",
                "import sqlite3",
                f"con = sqlite3.connect(r{('file:' + file + '?mode=ro')!r}, uri=True)",
                f"df = pd.read_sql_query({_sqlite_query(file, sheet)!r}, con)",
                "con.close()",
                ""]
    if not is_excel(file):
        with open(file, encoding="utf-8-sig", errors="replace", newline="") as f:
            sep, decimal = csv_format(f.read(4096))
        lines = ["# the table, as staplot reads it",
                 f"df = pd.read_csv(r{file!r}, sep={sep!r}, decimal={decimal!r}, encoding='utf-8-sig')"]
        if decimal == ",":
            lines += ["# a typo in a column keeps it as text: its decimal commas are read here",
                      "for c in df.columns:",
                      "    if df[c].dtype == object or str(df[c].dtype).startswith('str'):",
                      "        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors='coerce')"
                      ".where(lambda s: s.notna(), df[c])"]
        return lines + [""]
    engine = "xlrd" if file.lower().endswith(".xls") else "openpyxl"
    sheet_arg = (repr(int(sheet) - 1) if sheet and sheet.strip().isdigit() else repr(sheet)) if sheet else "0"
    return ["# the table: the Excel sheet, as staplot reads it",
            f"raw = pd.read_excel(r{file!r}, sheet_name={sheet_arg}, header=None, engine={engine!r})",
            "raw = raw.dropna(how='all').dropna(axis=1, how='all')      # blank margin rows / columns",
            "df = raw.iloc[1:].reset_index(drop=True)",
            "df.columns = [str(v).strip() for v in raw.iloc[0]]          # first row = headers",
            "for c in df.columns:                                        # numbers typed as text ('5,5')",
            "    as_number = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors='coerce')",
            "    if as_number.notna().sum() >= max(1, df[c].notna().sum() // 2):",
            "        df[c] = as_number.where(as_number.notna(), df[c])",
            ""]


def _grouped_code(path: str) -> list[str]:
    df = read_table(path)
    if df.shape[1] < 3:
        return ["# a grouped plot needs 3 columns (factor A, factor B, value)"]
    lines = ["import pandas as pd", "import statsmodels.formula.api as smf",
             "from statsmodels.stats.anova import anova_lm", ""]
    lines += _loading_code(path)
    lines += ["# long table: factor A, factor B, value (rows with a missing cell are left out)",
              "data = df.iloc[:, :3].copy()",
              "data.columns = ['A', 'B', 'value']",
              "data['value'] = pd.to_numeric(data['value'], errors='coerce')",
              "data = data.dropna()",
              "data[['A', 'B']] = data[['A', 'B']].astype(str)"]
    lines += ["model = smf.ols('value ~ C(A) * C(B)', data=data).fit()",
              "print(anova_lm(model, typ=2))      # factor A, factor B and A x B rows"]
    return lines
