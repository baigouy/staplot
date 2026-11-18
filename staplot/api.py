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

PLOT_KINDS = ("bar", "box", "violin", "scatter", "grouped", "split_violin", "paired")
VIOLIN_SIDES = ("both", "left", "right")
# the parts of a plot whose colour can be set (element_colors); the groups' own colours
# are *colors*
PLOT_ELEMENTS = {
    "points": "the individual points",
    "edges": "the outlines of the bars, boxes and violins",
    "median": "the median line (box) or dot / line (violin)",
    "whiskers": "the box plot whiskers and caps",
    "outliers": "the box plot outliers",
    "error_bars": "the error bars (bar, grouped)",
    "inner": "the marks inside a violin (lines or mini box)",
    "brackets": "the significance brackets and their stars",
    "fit": "the regression line (scatter)",
    "axes": "the axes, ticks and their labels",
    "background": "the plot's background",
    "mean": "the mean marker (show_mean)",
    "pair_lines": "the lines joining each subject's values (paired)",
    "reference": "the reference lines (hlines)",
}
_PART_TAGS = set(PLOT_ELEMENTS) | {"bar", "box", "violin"}


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


def _text_preamble(text: str, skip_rows=None) -> tuple[int, str, str]:
    """(lines to skip before the header, separator, decimal mark) of a text table. An
    instrument or a script often writes lines above the table ("Plate reader export",
    "Date: ...", "# comment"): they are skipped when they do not look like the table
    (another number of separators than the header and the rows below it). *skip_rows*:
    that many lines skipped, as the user says."""
    lines = text.splitlines()
    if skip_rows not in (None, "auto"):
        skip = max(0, int(skip_rows))
        body = "\n".join(lines[skip:skip + 40]) + "\n"
        sep, decimal = csv_format(body)
        return skip, sep, decimal
    # the separator from the bulk of the table (not its preamble)
    tail = [ln for ln in lines[len(lines) // 3:] if ln.strip()][:40] or lines
    sep, decimal = csv_format("\n".join(tail) + "\n")
    import csv as _csv

    def fields(ln):
        # quoted cells may hold the separator ("1,234")
        try:
            return len(next(_csv.reader([ln], delimiter=sep))) - 1
        except (StopIteration, _csv.Error):
            return ln.count(sep)
    counts = [fields(ln) if ln.strip() and not ln.lstrip().startswith("#") else -1 for ln in lines]
    seen = [c for c in counts if c > 0]
    if not seen:
        return 0, sep, decimal
    modal = max(set(seen), key=seen.count)
    skip = 0
    for i, c in enumerate(counts):
        if c == modal and (i + 1 >= len(counts) or counts[i + 1] in (modal, -1)):
            skip = i
            break
    return skip, sep, decimal


def preamble_note(path: str) -> str | None:
    """What reading *path* skipped above its table ("the first 3 lines: 'Plate reader
    export', ..."), or None."""
    file, _sheet = split_sheet(path)
    try:
        df = read_table(path)
    except Exception:
        return None
    lines = df.attrs.get("skipped_lines") or []
    if not lines:
        return None
    shown = ", ".join(repr(str(t)[:40]) for t in lines[:3]) + (", ..." if len(lines) > 3 else "")
    return (f"the first {len(lines)} line(s) above the table were skipped ({shown}); "
            f"skip_rows=0 reads them")


def read_table(path: str, skip_rows=None):
    """A table as a DataFrame: CSV / TSV / TXT (separator and decimal mark detected)
    or an Excel sheet (.xlsx, .xlsm, .xls; "book.xlsx::Sheet2" picks a sheet, by name
    or number from 1; the first one otherwise). Rows and columns left wholly empty in
    the sheet are dropped (a sheet often has a blank margin). Lines above the table
    (an instrument's header, a title, notes) are skipped when they do not look like
    the table; *skip_rows* = that many lines (text) or rows (Excel) skipped instead,
    0 = none. The skipped lines are in ``df.attrs["skipped_lines"]``."""
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
            head = "".join(line for _i, line in zip(range(400), f))
        skip, sep, decimal = _text_preamble(head, skip_rows)
        df = pd.read_csv(file, sep=sep, decimal=decimal, encoding="utf-8-sig", skiprows=skip,
                         comment=None)
        # one typo ("abc") in a decimal-comma column keeps the whole column as text,
        # and pandas then leaves its "41,2" unconverted: the group vanished
        df = _numbers_from_text(df, decimal_comma=(decimal == ","))
        df.attrs["skipped_lines"] = [ln for ln in head.splitlines()[:skip] if ln.strip()]
        df.attrs["skip_rows"] = skip
        return df
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
    skipped = []
    if skip_rows not in (None, "auto"):
        skipped = [" ".join(str(v) for v in r if v == v) for r in raw.iloc[:int(skip_rows)].values.tolist()]
        raw = raw.iloc[int(skip_rows):]
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    if raw.empty:
        return pd.DataFrame()
    if skip_rows in (None, "auto") and len(raw) > 2:
        # a title or notes above the table ("Experiment 12", "operator: XX"): rows with
        # far fewer cells than the table's header and rows
        filled = raw.notna().sum(axis=1).tolist()
        width = max(set(filled), key=filled.count)
        top = 0
        while top < len(filled) - 1 and filled[top] < max(2, 0.6 * width):
            top += 1
        if top:
            skipped = [" ".join(str(v) for v in r if v == v) for r in raw.iloc[:top].values.tolist()]
            raw = raw.iloc[top:].dropna(axis=1, how="all")
    header = [str(v).strip() if not (isinstance(v, float) and v != v) else f"column {i + 1}"
              for i, v in enumerate(raw.iloc[0])]
    df = raw.iloc[1:].reset_index(drop=True)
    df.columns = header
    # Excel cells typed as text ("12", "3,5") are numbers to a CSV reader too
    df = _numbers_from_text(df, decimal_comma=True)
    df.attrs["skipped_lines"] = skipped
    return df


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


def select_columns(df, columns, where=None):
    """*df* reduced to the rows matching *where* (a pandas query on any column: "day ==
    14", "sex == 'male' and area > 100"; backquotes around names with spaces) and to
    *columns* (names, any case), in that order: the columns of a wider (tidy) table a
    plot uses, e.g. ["genotype", "sex", "area"] of a table that also has day and
    treatment. None keeps every row / column."""
    if where:
        try:
            df = df.query(str(where), engine="python").reset_index(drop=True)
        except Exception as e:
            raise ValueError(f"cannot keep the rows where {where!r} ({type(e).__name__}: {e}); the "
                             f"columns are {', '.join(map(str, df.columns))}") from None
        if df.empty:
            raise ValueError(f"no row where {where}")
    if not columns:
        return df
    low = {str(c).strip().lower(): c for c in df.columns}
    picked, missing = [], []
    for c in columns:
        key = str(c).strip().lower()
        (picked if key in low else missing).append(low.get(key, c))
    if missing:
        raise ValueError(f"no column {missing}; the columns are {', '.join(map(str, df.columns))}")
    return df[picked].copy()


def _read_csv(path, columns=None, where=None, skip_rows=None):
    return select_columns(read_table(path, skip_rows), columns, where)


def _detect_layout(df, layout: str = "auto") -> str:
    import pandas as pd
    if layout != "auto":
        return layout
    first_numeric = pd.to_numeric(df.iloc[:, 0], errors="coerce").notna().mean()
    return "long" if (df.shape[1] == 2 and first_numeric < 0.5) else "wide"


def _has_label(value) -> bool:
    import pandas as pd
    return not pd.isna(value) and str(value).strip() != ""


def csv_cleaning_report(path: str, layout: str = "auto", columns=None, where=None, skip_rows=None) -> dict:
    """What reading *path* for a plot keeps and leaves out, so it can be told to the
    reader (or to a language model asked "what did you do with the empty cells?").

    Returns ``{"layout", "groups": {group: {"used", "missing", "non_numeric",
    "non_numeric_examples"}}, "rows_without_group", "summary"}``. *missing* counts empty
    cells and NA / nan / n/a markers; *non_numeric* counts text that is not a number
    (often a typo: "4O" for "40", "abc"). Both are left out of the plot and the
    statistics, group by group; nothing is imputed. In a long table a row with no group
    label is left out too (it is not a group of its own)."""
    import pandas as pd
    df = _read_csv(path, skip_rows=skip_rows)
    skipped = df.attrs.get("skipped_lines") or []
    df = select_columns(df, columns, where)
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
    if skipped:
        out["skipped_lines"] = skipped
        out["summary"] += (f" The {len(skipped)} line(s) above the table were skipped ("
                           + ", ".join(repr(str(t)[:40]) for t in skipped[:3]) + ").")
    return out


def table_from_csv(path: str, layout: str = "auto", name: str | None = None, columns=None,
                   where=None, skip_rows=None) -> DataTable:
    """Build a column DataTable from a CSV.

    *columns*: the columns used, of a wider table (e.g. ["genotype", "area"]: long
    layout, group then value). *where*: only the rows matching this pandas query
    ("day == 14", "sex == 'male'"). *skip_rows*: lines above the table to skip (default:
    found automatically).

    *layout*:
      - ``"wide"`` — every column is a group; rows are replicates (Prism-style).
      - ``"long"`` — first column = group label, second column = value.
      - ``"auto"`` — long when there are exactly two columns and the first is
        non-numeric; wide otherwise.
    """
    import pandas as pd
    df = _read_csv(path, columns, where, skip_rows)
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
        t = table_from_columns(columns, name=nm)
        # the axis labels the table gives: "area (um2)" on y, not "Value"
        t.value_name, t.group_name = str(vcol), str(gcol)
        return t

    # wide
    columns = {}
    for c in df.columns:
        columns[str(c)] = pd.to_numeric(df[c], errors="coerce").tolist()
    return table_from_columns(columns, name=nm)


def grouped_table_from_csv(path: str, name: str | None = None, columns=None,
                           where=None, skip_rows=None) -> GroupedDataTable:
    """Build a two-factor GroupedDataTable from a *long* CSV with three columns:
    ``factor_a`` (column group), ``factor_b`` (row label), ``value``. *columns*: the three
    columns used, of a wider (tidy) table, e.g. ["genotype", "sex", "area"]."""
    import pandas as pd
    df = _read_csv(path, columns, where, skip_rows)
    nm = name or os.path.splitext(os.path.basename(str(path)))[0]
    if df.shape[1] < 3:
        raise ValueError("grouped plot needs a long CSV with 3 columns: "
                         "factor_a, factor_b, value")
    fa, fb, val = df.columns[0], df.columns[1], df.columns[2]
    values = pd.to_numeric(df[val], errors="coerce")

    def numeric_share(col):
        # a factor coded by a few numbers (day 7 / 14, a 0 / 1 code) is a factor too
        if df[col].dropna().nunique() <= 12:
            return 0.0
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
    t.factor_names = (str(fa), str(fb))         # shown in the ANOVA note instead of "factor A / B"
    t.value_name = str(val)                      # the y label when none is given
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


def suggest_kind(source, columns=None, where=None, skip_rows=None) -> dict:
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
        df = select_columns(read_table(source, skip_rows), columns, where)
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


def paired_test(table: DataTable, only_significant: bool = True, parametric: bool | None = None) -> dict:
    """Repeated measures (one row per subject, one column per condition: before / after,
    day 1 / 2 / 3), on the subjects measured in every column: 2 conditions -> paired
    t-test (differences normal) or Wilcoxon signed-rank; more -> repeated-measures ANOVA
    or Friedman, then pairwise paired tests (Bonferroni). *parametric* forces the family."""
    names = [c.name for c in table.columns if len(c.valid_values()) > 0]
    rec = {"test": None, "n_subjects": 0, "comparisons": [], "sig_pairs": []}
    if len(names) < 2:
        rec["error"] = "a paired plot needs two columns or more (one per condition)"
        return rec
    arrays = table.paired_arrays(*names)
    n = len(arrays[0])
    rec["n_subjects"] = n
    if n < 3:
        rec["error"] = f"{n} subject(s) measured in every column: too few to test"
        return rec
    diffs = [arrays[j] - arrays[i] for i in range(len(arrays)) for j in range(i + 1, len(arrays))]
    normal = all(_is_normal(d) for d in diffs) if parametric is None else bool(parametric)
    rec["normality"] = ("forced" if parametric is not None else
                        ("differences normal (Shapiro)" if normal else "differences not normal (Shapiro)"))

    def pair_p(a, b):
        if normal:
            return float(_sp.ttest_rel(a, b).pvalue)
        try:
            return float(_sp.wilcoxon(a, b).pvalue)
        except ValueError:                      # all differences zero
            return 1.0
    if len(names) == 2:
        rec["test"] = "Paired t-test" if normal else "Wilcoxon signed-rank test"
        rec["comparisons"] = [_cmp(names[0], names[1], pair_p(arrays[0], arrays[1]))]
        shown = True
    else:
        if normal:
            import pandas as pd
            from statsmodels.stats.anova import AnovaRM
            long = pd.DataFrame([(s_i, names[k], arrays[k][s_i]) for k in range(len(names)) for s_i in range(n)],
                                columns=["subject", "condition", "value"])
            res = AnovaRM(long, "value", "subject", within=["condition"]).fit()
            p_omni = float(res.anova_table["Pr > F"].iloc[0])
            rec["test"] = "Repeated-measures ANOVA + paired t-tests (Bonferroni)"
            rec["omnibus"] = {"stat_name": "F", "stat": float(res.anova_table["F Value"].iloc[0]), "p": p_omni}
        else:
            stat, p_omni = _sp.friedmanchisquare(*arrays)
            rec["test"] = "Friedman + Wilcoxon signed-rank (Bonferroni)"
            rec["omnibus"] = {"stat_name": "chi2", "stat": float(stat), "p": float(p_omni)}
        k = len(names) * (len(names) - 1) // 2
        rec["comparisons"] = [_cmp(names[i], names[j], min(1.0, pair_p(arrays[i], arrays[j]) * k))
                              for i in range(len(names)) for j in range(i + 1, len(names))]
        shown = bool(np.isfinite(rec["omnibus"]["p"]) and rec["omnibus"]["p"] < 0.05)
        rec["posthoc_shown"] = shown
    rec["sig_pairs"] = [(c["a"], c["b"], c["p"]) for c in rec["comparisons"]
                        if shown and np.isfinite(c["p"]) and (c["p"] < 0.05 or not only_significant)]
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


def _annotate_two_way(fig, rec, names=None, ax=None):
    parts = []
    a, b = names if names else ("factor A", "factor B")
    shown = {"factor A": a, "factor B": b, "A x B": f"{a} × {b}"}
    for e in rec.get("effects", []):
        if e["p"] is None:
            continue
        tail = "" if e["stars"] in ("ns", "n/a") else " " + e["stars"]
        parts.append(f"{shown.get(e['effect'], e['effect'])}: p={e['p']:.3g}{tail}")
    if parts:
        # one effect per line: on one line it was wider than the plot, and a panel sized
        # to it squeezed the bars (PyFigures, 2026-11-18)
        if ax is not None:
            # a facet: the note under its own axes
            ax.annotate("Two-way ANOVA\n" + "\n".join(parts), xy=(0.5, 0), xycoords=ax.xaxis.label,
                        xytext=(0, -3), textcoords="offset points", ha="center", va="top", fontsize=6,
                        color="#444444", linespacing=1.2, annotation_clip=False)
            return
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
         show_points: bool = True, violin_inner: str = "lines", violin_side: str = "both",
         element_colors: dict | None = None, ax=None, show_n: bool = False, show_mean: bool = False,
         p_format: str = "stars", hlines=None):
    """Render *table* to a matplotlib Figure and return ``(fig, stats_record)``.

    *kind*: ``"bar"`` | ``"box"`` | ``"violin"`` | ``"scatter"`` | ``"grouped"`` |
    ``"split_violin"``, or ``"auto"``: the one that shows the data best (:func:`suggest_kind`).
    ``split_violin`` takes a grouped (two-factor) table where one factor has two levels
    (sex, treatment...): one violin per level of the other factor, its left half the
    first condition and its right half the second, a legend naming the halves; each
    pair is compared (t-test or Mann-Whitney, as :func:`auto_test` picks) and the
    two-way ANOVA is printed under the axes.
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
      * *violin_side* — ``"left"`` / ``"right"``: half violins (the points on the
        other side, a "raincloud"); ``"both"`` the whole violin;
      * *legend* — ``"inside"``, ``"outside"`` (right of the axes) or ``"none"``;
      * *element_colors* — the colour of parts of the plot: ``{"points": "black",
        "median": "red", "brackets": "#555555", "edges": "none"}`` (:data:`PLOT_ELEMENTS`
        lists them; the groups' fill colours are *colors*);
      * *show_ylabel* / *show_xlabel* — False: no axis label at all (an empty
        *ylabel* means the automatic one, "Value" or the column's name).
    *ax*: draw into these axes (one facet, :func:`plot_facets`) instead of a new figure.
    Journal details: *show_n* writes each group's n under its name ("WT\n(n=12)");
    *show_mean* marks each group's mean (a white diamond) on box / violin plots;
    *p_format* "stars" (default) or "p" (the p-value itself, "p=0.003") on the brackets;
    *hlines*: reference lines, ``[1.0]`` or ``[{"y": 100, "label": "control", "color":
    "grey", "style": "--"}]``.
    """
    extras = {"show_n": show_n, "show_mean": show_mean, "p_format": p_format, "hlines": hlines}
    if ax is not None:
        with _plots.styles.drawing_into(ax):
            return _untagged(_details(_plot_one(table, kind, error=error, annotate=annotate, only_significant=only_significant,
                             stats_mode=stats_mode, title=title, ylabel=ylabel, xlabel=xlabel, colors=colors,
                             show_title=show_title, order=order, show_ylabel=show_ylabel,
                             show_xlabel=show_xlabel, rename=rename, show_points=show_points,
                             violin_inner=violin_inner, violin_side=violin_side, facet_ax=ax),
                table, kind, ax, element_colors, **extras))
    return _untagged(_details(_plot_one(table, kind, error=error, annotate=annotate, only_significant=only_significant,
                     stats_mode=stats_mode, title=title, ylabel=ylabel, xlabel=xlabel, colors=colors,
                     show_title=show_title, aspect=aspect, width=width, height=height,
                     font_scale=font_scale, order=order, ylim=ylim, log_y=log_y,
                     xtick_rotation=xtick_rotation, legend=legend, show_ylabel=show_ylabel,
                     show_xlabel=show_xlabel, rename=rename, show_points=show_points,
                     violin_inner=violin_inner, violin_side=violin_side, element_colors=element_colors),
                     table, kind, None, element_colors, **extras))


P_FORMATS = ("stars", "p")


def _details(result, table, kind, ax=None, element_colors=None, *, show_n=False, show_mean=False,
             p_format="stars", hlines=None):
    """The journal details of a drawn plot: n under the group names, mean markers,
    p-values written out, reference lines."""
    fig, rec = result
    if p_format not in P_FORMATS:
        raise ValueError("p_format must be 'stars' or 'p' (the p-value itself)")
    axes = [ax] if ax is not None else fig.axes[:1]
    groups = table.group_arrays() if isinstance(table, DataTable) else None
    for a in axes:
        if show_n and groups and kind in ("bar", "box", "violin", "paired"):
            n = {str(k): int(np.isfinite(np.asarray(v, float)).sum()) for k, v in groups.items()}
            labels = [t.get_text() for t in a.get_xticklabels()]
            # the shown names may be renamed: n follows the position
            ns = [n.get(lb) for lb in labels]
            if any(v is None for v in ns) and len(n) == len(labels):
                ns = list(n.values())
            a.set_xticks(a.get_xticks(), [f"{lb}\n(n={v})" if v is not None else lb for lb, v in zip(labels, ns)])
        if show_n and kind in ("grouped", "split_violin") and not isinstance(table, DataTable):
            total = sum(1 for g in table.data.values() for cell in g.values() for v in cell
                        if v is not None and np.isfinite(float(v)))
            a.annotate(f"n={total}", xy=(1, 1), xycoords="axes fraction", ha="right", va="bottom",
                       fontsize=7, color="#444444")
        if show_mean and groups and kind in ("box", "violin"):
            for i, (name, vals) in enumerate(groups.items()):
                arr = np.asarray(vals, float)
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    m = a.scatter([i], [arr.mean()], marker="D", s=22, color="white", edgecolors="#222222",
                                  linewidths=0.8, zorder=6)
                    m._staplot_part = "mean"
        if p_format == "p":
            from .stats.descriptive import format_p
            for t in a.texts:
                p = getattr(t, "_staplot_p", None)
                if p is not None:
                    t.set_text(format_p(p))
                    t.set_fontweight("normal")
                    t.set_fontsize(min(t.get_fontsize(), 7))
        for line in (hlines or []):
            spec = line if isinstance(line, dict) else {"y": line}
            try:
                y = float(spec["y"])
            except (KeyError, TypeError, ValueError):
                raise ValueError('hlines: numbers, or {"y": 100, "label": "control", "color": "grey", '
                                 '"style": "--"}') from None
            ref = a.axhline(y, color=spec.get("color", "#888888"), linestyle=spec.get("style", "--"),
                            linewidth=float(spec.get("width", 0.9)), zorder=1)
            ref._staplot_part = "reference"
            if spec.get("label"):
                a.annotate(str(spec["label"]), xy=(1, y), xycoords=("axes fraction", "data"), xytext=(-2, 2),
                           textcoords="offset points", ha="right", va="bottom", fontsize=7,
                           color=spec.get("color", "#666666"))
    if element_colors and (show_mean or hlines):
        apply_element_colors(fig, {k: v for k, v in element_colors.items() if k in ("mean", "reference")})
    if ax is None and (show_n or p_format == "p"):
        fig.tight_layout()
    return fig, rec


def plot_part(artist) -> str | None:
    """The part of a staplot plot an artist draws ("points", "median", "brackets",
    "edges"...: PLOT_ELEMENTS), or None."""
    return getattr(artist, "_staplot_part", None) or (
        artist.get_gid() if artist.get_gid() in _PART_TAGS else None)


def _untagged(result):
    """The plot with its parts named by a private attribute, not by their gid: a gid is
    the element's id in an SVG, and many elements sharing one (every bracket) were drawn
    once by a renderer that draws elements by id (PyFigures)."""
    fig = result[0]
    for artist in fig.findobj():
        gid = artist.get_gid()
        if gid in _PART_TAGS:
            artist._staplot_part = gid
            artist.set_gid(None)
    return result


def _plot_one(table, kind: str = "bar", *, error: str = "sd", annotate: bool = True,
              only_significant: bool = True, stats_mode: str = "auto", title: str = "",
              ylabel: str = "", xlabel: str = "", colors=None, show_title: bool = True,
              aspect=None, width=None, height=None, font_scale=None, order=None, ylim=None,
              log_y: bool = False, xtick_rotation=None, legend=None, show_ylabel: bool = True,
              show_xlabel: bool = True, rename=None, show_points: bool = True,
              violin_inner: str = "lines", violin_side: str = "both", element_colors=None,
              facet_ax=None):
    kind = kind.lower()
    # the y label the table gives (its value column: "area (um2)") when none is asked
    ylabel = ylabel or getattr(table, "value_name", "") or ""
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
    if annotate and kind == "paired":
        stats_record = paired_test(table, only_significant=only_significant, parametric=parametric)
        sig_pairs = stats_record.get("sig_pairs") or None

    if violin_inner not in ("lines", "box", "none"):
        raise ValueError("violin_inner must be 'lines', 'box' or 'none'")
    if violin_side not in VIOLIN_SIDES:
        raise ValueError("violin_side must be 'both', 'left' or 'right'")
    if kind == "split_violin" and isinstance(table, DataTable):
        raise ValueError("split_violin needs a two-factor table (three columns: e.g. genotype, "
                         "sex, value) where one factor has two levels; for half violins of "
                         "one factor use kind='violin' with violin_side='left'")
    if kind == "bar":
        fig = _plots.plot_bar(table, error_type=error, title=title, ylabel=ylabel,
                              colors=colors, sig_pairs=sig_pairs, show_points=show_points)
    elif kind in ("box", "violin"):
        fig = _plots.plot_box(table, plot_type=kind, title=title, ylabel=ylabel,
                              colors=colors, sig_pairs=sig_pairs, show_points=show_points,
                              violin_inner=violin_inner, violin_side=violin_side)
    elif kind == "paired":
        if not isinstance(table, DataTable):
            raise ValueError("a paired plot needs a table with one column per condition and one row per "
                             "subject (before, after)")
        fig = _plots.plot_before_after(table, error_type=error if error != "none" else "sd",
                                       show_mean=error != "none", title=title, ylabel=ylabel,
                                       colors=colors, sig_pairs=sig_pairs)
    elif kind == "scatter":
        fig = _plots.plot_scatter(table, title=title, xlabel=xlabel, ylabel=ylabel,
                                  colors=colors)
        if annotate:
            stats_record = correlation_test(table)
    elif kind == "split_violin":
        x_levels, halves, arrays = split_parts(table, order=order)
        pair_p, pairs = None, None
        if annotate:
            pairs = split_pair_tests(x_levels, halves, arrays, parametric=parametric)
            pair_p = {c["x"]: c["p"] for c in pairs
                      if c["p"] is not None and (c["p"] < 0.05 or not only_significant)}
        names = getattr(table, "factor_names", None)
        if names:
            h_name, x_name = (names[0], names[1]) if list(halves) == list(table.col_groups) else (names[1], names[0])
        else:
            h_name = x_name = ""
        fig = _plots.plot_split_violin(x_levels, halves, arrays, pair_p=pair_p,
                                       show_points=show_points, title=title, ylabel=ylabel,
                                       xlabel=xlabel or x_name, colors=colors, violin_inner=violin_inner,
                                       legend_title=h_name)
        if annotate:
            stats_record = two_way_test(table)
            stats_record["split_comparisons"] = pairs
            if "error" not in stats_record:
                _annotate_two_way(fig, stats_record, getattr(table, "factor_names", None), ax=facet_ax)
    elif kind == "grouped":
        fig = _plots.plot_grouped_bar(table, error_type=error, title=title,
                                      ylabel=ylabel, colors=colors)
        names = getattr(table, "factor_names", None)
        if names:
            ax = facet_ax if facet_ax is not None else fig.axes[0]
            if ax.get_legend() is not None and ax.get_legend().get_title().get_text() in ("", "Group"):
                ax.get_legend().set_title(names[0])          # what the colours are
            if xlabel or not ax.get_xlabel():
                ax.set_xlabel(xlabel or names[1])
        if annotate:
            stats_record = two_way_test(table)
            if "error" not in stats_record:
                _annotate_two_way(fig, stats_record, getattr(table, "factor_names", None), ax=facet_ax)
    if facet_ax is not None:
        # one facet: the figure's size, layout and colours are set once, by plot_facets
        if not show_ylabel:
            facet_ax.set_ylabel("")
        if not show_xlabel:
            facet_ax.set_xlabel("")
        if rename and not isinstance(table, DataTable):
            names = [t.get_text() for t in facet_ax.get_xticklabels()]
            facet_ax.set_xticks(facet_ax.get_xticks(), [str(rename.get(n, n)) for n in names])
            for t in (facet_ax.get_legend().get_texts() if facet_ax.get_legend() else []):
                if t.get_text() in rename:
                    t.set_text(str(rename[t.get_text()]))
        return fig, stats_record
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
    if element_colors:
        apply_element_colors(fig, element_colors)
    if kind in ("grouped", "split_violin") and annotate and stats_record and "error" not in stats_record:
        # room for the ANOVA line under the axis labels (they ran into each other)
        lines = getattr(fig, "_staplot_note_lines", 1)
        fig.tight_layout(rect=(0, min(0.4, 0.012 + 0.026 * lines * 4 / max(fig.get_size_inches()[1], 1)), 1, 1))
    return fig, stats_record


def facet_levels(path: str, facet: str, where=None, skip_rows=None) -> tuple[str, list]:
    """(the facet column's name as in the table, its levels in order: numbers sorted,
    text in the order of the table)."""
    import pandas as pd
    df = select_columns(read_table(path, skip_rows), None, where)
    low = {str(c).strip().lower(): c for c in df.columns}
    col = low.get(str(facet).strip().lower())
    if col is None:
        raise ValueError(f"no column {facet!r} to facet by; the columns are {', '.join(map(str, df.columns))}")
    values = df[col].dropna()
    if pd.api.types.is_numeric_dtype(values):
        levels = sorted(values.unique().tolist())
    else:
        levels = list(dict.fromkeys(values.astype(str).str.strip()))
    if len(levels) > 24:
        raise ValueError(f"{col} has {len(levels)} values: too many facets (24 at most)")
    return str(col), levels


def facet_tables(path: str, kind: str, facet: str, facet_row: str | None = None, *, columns=None,
                 where=None, skip_rows=None):
    """[(label, table or None, error)] of a faceted plot: one table per value of *facet*
    (and of *facet_row*), "day = 7" ..."""
    col, levels = facet_levels(path, facet, where, skip_rows)
    row_col, row_levels = (facet_levels(path, facet_row, where, skip_rows) if facet_row else (None, [None]))
    two = kind in ("grouped", "split_violin")
    if columns is None:
        columns = [str(c) for c in read_table(path, skip_rows).columns if str(c) not in (col, row_col)]
    out = []
    for r_level in row_levels:
        for level in levels:
            parts = [f"({where})"] if where else []
            parts.append(f"`{col}` == {level!r}")
            if facet_row:
                parts.append(f"`{row_col}` == {r_level!r}")
            sub = " and ".join(parts)
            label = f"{col} = {level}" + (f" | {row_col} = {r_level}" if facet_row else "")
            try:
                table = (grouped_table_from_csv(path, columns=columns, where=sub, skip_rows=skip_rows) if two
                         else table_from_csv(path, columns=columns, where=sub, skip_rows=skip_rows))
                out.append((label, table, ""))
            except ValueError as e:
                out.append((label, None, str(e)))
    return out


def facet_stats(path: str, kind: str, facet: str, facet_row: str | None = None, *, columns=None,
                where=None, skip_rows=None, only_significant: bool = True, stats_mode: str = "auto") -> dict:
    """{facet label: the statistics of that facet} (the tests plot_facets draws)."""
    parametric = _STATS_MODE.get(stats_mode)
    out = {}
    for label, table, err in facet_tables(path, kind, facet, facet_row, columns=columns, where=where,
                                          skip_rows=skip_rows):
        if table is None:
            out[label] = {"error": err}
        elif kind in ("bar", "box", "violin"):
            out[label] = auto_test(table, only_significant=only_significant, parametric=parametric)
        elif kind == "paired":
            out[label] = paired_test(table, only_significant=only_significant, parametric=parametric)
        elif kind in ("grouped", "split_violin"):
            rec = two_way_test(table)
            if kind == "split_violin" and "error" not in rec:
                try:
                    rec["split_comparisons"] = split_pair_tests(*split_parts(table), parametric=parametric)
                except ValueError as e:
                    rec["split_error"] = str(e)
            out[label] = rec
        elif kind == "scatter":
            out[label] = correlation_test(table)
    return out


@_plots.styles.styled
def plot_facets(path: str, kind: str = "box", facet: str = "", *, columns=None, where=None,
                skip_rows=None, ncols: int | None = None, sharey: bool = True,
                facet_row: str | None = None, **kwargs):
    """One plot per value of the *facet* column (as ggplot2's facet_wrap), side by side,
    each with its own statistics: e.g. box plots of area per genotype, one facet per
    day: ``plot_facets("cells.csv", "box", "day", columns=["genotype", "area"])``.
    *facet_row*: a second column, one row of facets per value (facet_grid).
    *columns*, *where*, *skip_rows*: as for :func:`table_from_csv` (the columns of the
    plot itself, without the facet column). *ncols*: facets per row (default: up to 4).
    *sharey*: the same y axis for every facet. Other keywords go to :func:`plot`
    (kind, error, colors, violin_inner, element_colors, width / height of the whole
    figure...). Returns ``(fig, {"facets": {level: stats_record}})``."""
    import matplotlib.pyplot as plt
    kind = kind.lower()
    col, levels = facet_levels(path, facet, where, skip_rows)
    row_col, row_levels = (facet_levels(path, facet_row, where, skip_rows) if facet_row else (None, [None]))
    if columns is None:
        columns = [str(c) for c in read_table(path, skip_rows).columns if str(c) not in (col, row_col)]
    if kind == "auto":
        kind = suggest_kind(path, columns=columns, where=where, skip_rows=skip_rows)["kind"] or "box"
    n = len(levels)
    nc = n if facet_row else min(n, int(ncols) if ncols else 4)
    nr = len(row_levels) if facet_row else -(-n // nc)
    per_w = 2.2 if kind in ("bar", "box", "violin") else 3.0
    # own y axes while drawing (each facet's brackets need their room), one range after
    fig, axes = plt.subplots(nr, nc, figsize=(per_w * nc + 0.6, 3.0 * nr + 0.4), squeeze=False)
    layout_keys = ("aspect", "width", "height", "font_scale", "ylim", "log_y", "xtick_rotation", "legend",
                   "element_colors")
    layout = {k: kwargs.pop(k) for k in layout_keys if k in kwargs}
    title = kwargs.pop("title", "")          # over the whole figure; each facet shows its value
    kwargs.pop("show_title", None)
    records = {}
    cells = facet_tables(path, kind, facet, facet_row, columns=columns, where=where, skip_rows=skip_rows)
    for i, (label, table, err) in enumerate(cells):
        ax = axes.flat[i]
        rec = {"error": err}
        if table is not None:
            try:
                _f, rec = plot(table, kind, ax=ax, **kwargs)
            except ValueError as e:
                table, rec = None, {"error": str(e)}
        if table is None:
            ax.text(0.5, 0.5, f"no data\n({rec['error'][:60]})", ha="center", va="center", fontsize=7,
                    transform=ax.transAxes)
        ax.set_title(label, fontsize=9)
        records[label] = rec
    for ax in list(axes.flat)[len(cells):]:
        ax.set_visible(False)
    shown = [a for a in axes.flat if a.get_visible()]
    if sharey and shown:
        lo = min(a.get_ylim()[0] for a in shown)
        hi = max(a.get_ylim()[1] for a in shown)
        for a in shown:
            a.set_ylim(lo, hi)
            if a not in axes[:, 0]:
                a.set_ylabel("")
                a.tick_params(labelleft=False)
    # one legend for the figure, above the facets (on the first one it covered its title)
    top = 1.0
    first = next((a for a in shown if a.get_legend() is not None), None)
    if first is not None:
        handles, labels = first.get_legend_handles_labels()
        title = first.get_legend().get_title().get_text()
        for a in shown:
            if a.get_legend() is not None:
                a.get_legend().remove()
        fig.legend(handles, labels, title=title or None, loc="upper center", ncol=max(1, len(labels)),
                   frameon=False, fontsize=8, title_fontsize=8, bbox_to_anchor=(0.5, 1.0))
        top = 1 - 0.55 / fig.get_size_inches()[1]
    if layout.get("aspect"):
        w, h = fig.get_size_inches()
        fig.set_size_inches(max(0.5, float(layout.pop("aspect")) * h), h)
    _apply_layout(fig, **{k: v for k, v in layout.items() if k != "element_colors"})
    if layout.get("element_colors"):
        apply_element_colors(fig, layout["element_colors"])
    if title:
        fig.suptitle(title)
        top = min(top, 1 - 0.35 / fig.get_size_inches()[1])
    fig.tight_layout(rect=(0, 0, 1, top))
    return fig, {"facets": records, "facet": col, "facet_row": row_col}


def apply_element_colors(fig, element_colors: dict) -> None:
    """Colour parts of a plot drawn by staplot: {"points": "black", "median": "red",
    "edges": "none", ...} (PLOT_ELEMENTS); an unknown part raises."""
    from matplotlib.colors import to_rgba
    unknown = [k for k in element_colors if k not in PLOT_ELEMENTS]
    if unknown:
        raise ValueError(f"no plot element {unknown}; the elements are {', '.join(PLOT_ELEMENTS)}")
    colors = {}
    for k, v in element_colors.items():
        try:
            colors[k] = "none" if str(v).lower() == "none" else to_rgba(v)
        except ValueError:
            raise ValueError(f"unknown colour {v!r} for {k}") from None
    edge_gids = {"bar", "box", "violin"}
    for ax in fig.axes:
        for artist in ax.findobj():
            gid = plot_part(artist)
            if not gid:
                continue
            if gid in edge_gids and "edges" in colors:
                if hasattr(artist, "set_edgecolor"):
                    artist.set_edgecolor(colors["edges"])
                elif hasattr(artist, "set_color"):          # a box drawn as lines
                    artist.set_color(colors["edges"])
            if gid not in colors:
                continue
            c = colors[gid]
            if gid in ("points", "mean"):
                artist.set_facecolor(c)
            elif gid == "outliers" and hasattr(artist, "set_markeredgecolor"):
                artist.set_markeredgecolor(c)
                artist.set_markerfacecolor(c)
            elif type(artist).__name__ == "Rectangle":      # the mini box of a violin
                artist.set_edgecolor(c)
            elif type(artist).__name__ == "PathCollection":  # the violin's median dot
                artist.set_edgecolor(c)
            elif hasattr(artist, "set_color"):
                artist.set_color(c)
        if "axes" in colors:
            c = colors["axes"]
            for spine in ax.spines.values():
                spine.set_edgecolor(c)
            ax.tick_params(colors=c)
            ax.xaxis.label.set_color(c)
            ax.yaxis.label.set_color(c)
        if "background" in colors:
            ax.set_facecolor(colors["background"])
            fig.patch.set_facecolor(colors["background"])


def split_parts(gtable, order=None):
    """(x_levels, halves, arrays) of a two-factor table for split violins: the factor
    with two levels gives the halves (the second column's if both have two: a table
    "genotype, sex, area" reads x = genotype, split by sex), the other the x positions;
    arrays[(x_level, half)] = the values."""
    a, b = list(gtable.col_groups), list(gtable.row_labels)
    if len(b) == 2:
        halves, x_levels, key = b, a, (lambda x, h: (x, h))
    elif len(a) == 2:
        halves, x_levels, key = a, b, (lambda x, h: (h, x))
    else:
        raise ValueError(f"split violins need a factor with two levels (the two halves); "
                         f"{gtable.name} has {len(a)} ({', '.join(a)}) and {len(b)} "
                         f"({', '.join(b)}): use kind='grouped', or a seaborn violinplot")
    if order:
        low = {str(x).lower(): x for x in x_levels}
        x_levels = [low[str(o).lower()] for o in order if str(o).lower() in low] or x_levels
    arrays = {}
    for x in x_levels:
        for h in halves:
            cg, rl = key(x, h)
            arrays[(x, h)] = [float(v) for v in gtable.data.get(cg, {}).get(rl, [])
                              if v is not None and np.isfinite(float(v))]
    return x_levels, halves, arrays


def split_pair_tests(x_levels, halves, arrays, parametric=None) -> list[dict]:
    """At each x position, the two halves compared as :func:`auto_test` compares two
    groups (Welch t-test or Mann-Whitney)."""
    out = []
    for x in x_levels:
        cols = {str(h): arrays[(x, h)] for h in halves}
        rec = {"x": str(x), "a": str(halves[0]), "b": str(halves[1]), "test": None, "p": None}
        if all(len(v) >= 2 for v in cols.values()):
            r = auto_test(table_from_columns(cols, name=str(x)), only_significant=False,
                          parametric=parametric)
            if r.get("comparisons"):
                rec["test"], rec["p"] = r.get("test"), r["comparisons"][0]["p"]
        out.append(rec)
    return out


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
              show_points: bool = True, violin_inner: str = "lines", violin_side: str = "both",
              columns=None, where=None, skip_rows=None, element_colors: dict | None = None,
              facet: str | None = None, facet_row: str | None = None, ncols: int | None = None,
              show_n: bool = False, show_mean: bool = False, p_format: str = "stars", hlines=None) -> str:
    """The Python code of a plot (user: "can staplot print the code it used to generate
    the fig"): a runnable script whose first part redraws the figure with staplot, and
    whose second part reads the table with pandas as staplot does (separator, decimal
    comma, Excel sheet, blank margin, numbers typed as text, empty / NA / typo cells
    left out) and runs the same statistics with scipy / statsmodels alone, so a
    reader or reviewer can check every step without staplot."""
    from . import __version__
    if kind == "auto":
        auto = suggest_kind(path, columns=columns, where=where, skip_rows=skip_rows)
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
                                 ("show_points", show_points, True), ("violin_inner", violin_inner, "lines"),
                                 ("violin_side", violin_side, "both"),
                                 ("element_colors", dict(element_colors) if element_colors else None, None),
                                 ("show_n", show_n, False), ("show_mean", show_mean, False),
                                 ("p_format", p_format, "stars"), ("hlines", list(hlines) if hlines else None, None)):
        if value != default:
            args.append(f"{name}={value!r}")
    if facet:
        # a faceted plot: one plot per value of the facet column, each with its statistics
        fargs = [f"kind={kind!r}", f"facet={facet!r}"] + args[1:]
        for name, value in (("facet_row", facet_row), ("ncols", ncols),
                            ("columns", list(columns) if columns else None), ("where", where),
                            ("skip_rows", None if skip_rows in (None, "auto") else int(skip_rows))):
            if value is not None:
                fargs.append(f"{name}={value!r}")
        return "\n".join([
            f"# Made by staplot {__version__} from {os.path.basename(split_sheet(path)[0])}"
            + (f" (sheet {split_sheet(path)[1]})" if split_sheet(path)[1] else "") + ".",
            "# One plot per value of " + repr(facet) + (f" and of {facet_row!r}" if facet_row else "")
            + "; each facet's statistics are in stats_record['facets'].",
            "from staplot import api",
            "",
            f"fig, stats_record = api.plot_facets(r{str(path)!r}, " + ", ".join(fargs) + ")",
            'fig.savefig("plot.pdf")             # vector; or .png / .svg',
            "for facet, rec in stats_record['facets'].items():",
            "    print(facet, rec.get('test'), [(c['a'], c['b'], c['p']) for c in rec.get('comparisons') or []])",
            ""])
    builder = "grouped_table_from_csv" if kind in ("grouped", "split_violin") else "table_from_csv"
    out = [f"# Made by staplot {__version__} from {os.path.basename(split_sheet(path)[0])}"
           + (f" (sheet {split_sheet(path)[1]})" if split_sheet(path)[1] else "") + ".",
           "# Part 1 redraws the figure; part 2 runs the statistics without staplot.",
           "",
           "# --- 1. the figure " + "-" * 60,
           "from staplot import api",
           "",
           f"table = api.{builder}(r{str(path)!r}" + (f", columns={list(columns)!r}" if columns else "")
           + (f", where={where!r}" if where else "")
           + (f", skip_rows={int(skip_rows)}" if skip_rows not in (None, "auto") else "") + ")",
           "fig, stats_record = api.plot(table, " + ", ".join(args) + ")",
           'fig.savefig("plot.pdf")             # vector; or .png / .svg',
           "print(stats_record)",
           ""]
    if not annotate:
        return "\n".join(out)
    out += ["# --- 2. the statistics with scipy / statsmodels alone " + "-" * 24]
    if kind in ("grouped", "split_violin"):
        out += _grouped_code(path, columns, where, skip_rows)
        if kind == "split_violin":
            out += _split_code(path, order=order, parametric=_STATS_MODE.get(stats_mode), columns=columns,
                               where=where, skip_rows=skip_rows)
        return "\n".join(out)
    if kind == "paired":
        table = table_from_csv(path, columns=columns, where=where, skip_rows=skip_rows)
        rec = paired_test(table, only_significant=only_significant, parametric=_STATS_MODE.get(stats_mode))
        out += ["import numpy as np", "import pandas as pd", "from scipy import stats", ""]
        out += _loading_code(path, columns, where, skip_rows)
        out += ["# one row per subject, one column per condition: the subjects measured in every column",
                "data = df.apply(pd.to_numeric, errors='coerce').dropna()",
                "names = list(data.columns)",
                "arrays = [data[n].to_numpy(float) for n in names]",
                "print(len(data), 'subjects')",
                f"# staplot chose: {rec.get('test')} ({rec.get('normality')})"]
        test = str(rec.get("test") or "")
        pair = ("stats.ttest_rel(arrays[i], arrays[j])" if "t-test" in test
                else "stats.wilcoxon(arrays[i], arrays[j])")
        if test.startswith("Repeated"):
            out += ["from statsmodels.stats.anova import AnovaRM",
                    "long = data.reset_index().melt(id_vars='index', var_name='condition', value_name='value')",
                    "print(AnovaRM(long, 'value', 'index', within=['condition']).fit())"]
        elif test.startswith("Friedman"):
            out += ["print(stats.friedmanchisquare(*arrays))"]
        k = "len(names) * (len(names) - 1) // 2" if len(rec.get("comparisons") or []) > 1 else "1"
        out += ["for i in range(len(names)):",
                "    for j in range(i + 1, len(names)):",
                f"        r = {pair}",
                f"        print(names[i], 'vs', names[j], r, 'p (Bonferroni) =', min(1.0, r.pvalue * {k}))"]
        return "\n".join(out)
    table = table_from_csv(path, columns=columns, where=where, skip_rows=skip_rows)
    # the data come from the table itself, read and cleaned in the script as staplot does
    out +=["import numpy as np", "import pandas as pd", "from scipy import stats", ""]
    out += _loading_code(path, columns, where, skip_rows)
    layout = _detect_layout(select_columns(read_table(path, skip_rows), columns, where))
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


def _loading_code(path: str, columns=None, where=None, skip_rows=None) -> list[str]:
    lines = _loading_code_all(path, skip_rows)
    if where:
        lines = lines[:-1] + [f"df = df.query({str(where)!r}, engine='python').reset_index(drop=True)"
                              "   # the rows the plot uses", ""]
    if columns:
        cols = [str(c) for c in select_columns(read_table(path, skip_rows), columns).columns]
        lines = lines[:-1] + [f"df = df[{cols!r}]                  # the columns the plot uses", ""]
    return lines


def _loading_code_all(path: str, skip_rows=None) -> list[str]:
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
            head = "".join(line for _i, line in zip(range(400), f))
        skip, sep, decimal = _text_preamble(head, skip_rows)
        lines = ["# the table, as staplot reads it" + (f" (the {skip} line(s) above it skipped)" if skip else ""),
                 f"df = pd.read_csv(r{file!r}, sep={sep!r}, decimal={decimal!r}, encoding='utf-8-sig'"
                 + (f", skiprows={skip}" if skip else "") + ")"]
        if decimal == ",":
            lines += ["# a typo in a column keeps it as text: its decimal commas are read here",
                      "for c in df.columns:",
                      "    if df[c].dtype == object or str(df[c].dtype).startswith('str'):",
                      "        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors='coerce')"
                      ".where(lambda s: s.notna(), df[c])"]
        return lines + [""]
    engine = "xlrd" if file.lower().endswith(".xls") else "openpyxl"
    sheet_arg = (repr(int(sheet) - 1) if sheet and sheet.strip().isdigit() else repr(sheet)) if sheet else "0"
    out = ["# the table: the Excel sheet, as staplot reads it",
            f"raw = pd.read_excel(r{file!r}, sheet_name={sheet_arg}, header=None, engine={engine!r})"]
    if skip_rows not in (None, "auto") and int(skip_rows):
        out.append(f"raw = raw.iloc[{int(skip_rows)}:]                         # the rows above the table")
    out.append("raw = raw.dropna(how='all').dropna(axis=1, how='all')      # blank margin rows / columns")
    if skip_rows in (None, "auto"):
        n = len(read_table(path).attrs.get("skipped_lines") or [])
        if n:
            out.append(f"raw = raw.iloc[{n}:].dropna(axis=1, how='all')           # the title / notes above the table")
    return out + [
            "df = raw.iloc[1:].reset_index(drop=True)",
            "df.columns = [str(v).strip() for v in raw.iloc[0]]          # first row = headers",
            "for c in df.columns:                                        # numbers typed as text ('5,5')",
            "    as_number = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors='coerce')",
            "    if as_number.notna().sum() >= max(1, df[c].notna().sum() // 2):",
            "        df[c] = as_number.where(as_number.notna(), df[c])",
            ""]


def _grouped_code(path: str, columns=None, where=None, skip_rows=None) -> list[str]:
    df = select_columns(read_table(path, skip_rows), columns, where)
    if df.shape[1] < 3:
        return ["# a grouped plot needs 3 columns (factor A, factor B, value)"]
    lines = ["import pandas as pd", "import statsmodels.formula.api as smf",
             "from statsmodels.stats.anova import anova_lm", ""]
    lines += _loading_code(path, columns, where, skip_rows)
    lines += ["# long table: factor A, factor B, value (rows with a missing cell are left out)",
              "data = df.iloc[:, :3].copy()",
              "data.columns = ['A', 'B', 'value']",
              "data['value'] = pd.to_numeric(data['value'], errors='coerce')",
              "data = data.dropna()",
              "data[['A', 'B']] = data[['A', 'B']].astype(str)"]
    lines += ["model = smf.ols('value ~ C(A) * C(B)', data=data).fit()",
              "print(anova_lm(model, typ=2))      # factor A, factor B and A x B rows"]
    return lines


def _split_code(path: str, order=None, parametric=None, columns=None, where=None,
                skip_rows=None) -> list[str]:
    """The halves of each split violin compared with scipy, as staplot did."""
    gtable = grouped_table_from_csv(path, columns=columns, where=where, skip_rows=skip_rows)
    x_levels, halves, arrays = split_parts(gtable, order=order)
    pairs = split_pair_tests(x_levels, halves, arrays, parametric=parametric)
    h_col = "A" if list(halves) == list(gtable.col_groups) else "B"
    x_col = "B" if h_col == "A" else "A"
    lines = ["", f"# split violins: at each {x_col} level, the two halves ({h_col} = "
                 f"{halves[0]!r} vs {halves[1]!r}) compared",
             "from scipy import stats"]
    for c in pairs:
        if not c["test"]:
            continue
        sel = "data[(data." + x_col + " == " + repr(c["x"]) + ") & (data." + h_col + " == {h})].value"
        lines.append("left, right = " + sel.format(h=repr(c["a"])) + ", " + sel.format(h=repr(c["b"])))
        if str(c["test"]).startswith("Welch"):
            lines.append(f"print({c['x']!r}, stats.ttest_ind(left, right, equal_var=False))   # Welch t-test")
        elif str(c["test"]).startswith("Mann"):
            lines.append(f"print({c['x']!r}, stats.mannwhitneyu(left, right, alternative='two-sided'))")
        else:
            lines.append(f"# {c['x']}: {c['test']}")
    return lines
