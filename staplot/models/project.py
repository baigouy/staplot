from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


def unique_name(name: str, taken) -> str:
    """*name*, or *name* (2), (3)… if already in *taken*."""
    taken = set(taken)
    if name not in taken:
        return name
    k = 2
    while f"{name} ({k})" in taken:
        k += 1
    return f"{name} ({k})"


class DataColumn:
    def __init__(self, name: str, values: list | None = None):
        self.name = name
        self.values: list = values if values is not None else []
        self.excluded: set[int] = set()  # row indices excluded by the user

    @staticmethod
    def _num(v) -> float:
        """Cell → float; empty or non-numeric (e.g. a text label column) → NaN."""
        if v is None:
            return float("nan")
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    def as_array(self) -> np.ndarray:
        return np.array([self._num(v) for v in self.values], dtype=float)

    def valid_values(self) -> np.ndarray:
        arr = self.as_array()
        mask = ~np.isnan(arr)
        if self.excluded:
            for idx in self.excluded:
                if 0 <= idx < len(mask):
                    mask[idx] = False
        return arr[mask]

    def valid_values_with_indices(self) -> tuple[np.ndarray, list[int]]:
        """Return (values, original_row_indices) skipping NaN and excluded rows."""
        result_vals: list[float] = []
        result_idxs: list[int] = []
        for i, v in enumerate(self.values):
            if i in self.excluded:
                continue
            f = self._num(v)
            if not np.isnan(f):
                result_vals.append(f)
                result_idxs.append(i)
        return np.array(result_vals, dtype=float), result_idxs

    def toggle_excluded(self, row: int):
        if row in self.excluded:
            self.excluded.discard(row)
        else:
            self.excluded.add(row)


class DataTable:
    """Spreadsheet-like data container matching Prism's Column format."""

    def __init__(self, name: str, table_type: str = "column"):
        self.name = name
        self.table_type = table_type  # "column" | "xy"
        self.columns: list[DataColumn] = []

    # ------------------------------------------------------------------
    def add_column(self, name: str | None = None) -> DataColumn:
        """Column names are the keys analyses use, so they are kept unique:
        a taken name gets a ' (2)', ' (3)'… suffix."""
        if name is None:
            name = f"Group {len(self.columns) + 1}"
        col = DataColumn(name=unique_name(name, [c.name for c in self.columns]))
        self.columns.append(col)
        return col

    def remove_column(self, index: int):
        if 0 <= index < len(self.columns):
            self.columns.pop(index)

    def rename_column(self, index: int, new_name: str):
        if 0 <= index < len(self.columns):
            if any(c.name == new_name for i, c in enumerate(self.columns) if i != index):
                raise ValueError(f"A column named '{new_name}' already exists.")
            self.columns[index].name = new_name

    def num_rows(self) -> int:
        if not self.columns:
            return 0
        return max(len(c.values) for c in self.columns)

    def ensure_row(self, row: int):
        for col in self.columns:
            while len(col.values) <= row:
                col.values.append(None)

    def set_value(self, row: int, col_index: int, value):
        self.ensure_row(row)
        col = self.columns[col_index]
        col.values[row] = value

    def to_dataframe(self) -> pd.DataFrame:
        if not self.columns:
            return pd.DataFrame()
        n = self.num_rows()
        data = {}
        for col in self.columns:
            vals = list(col.values) + [np.nan] * (n - len(col.values))
            data[col.name] = vals
        return pd.DataFrame(data)

    def group_arrays(self) -> dict[str, np.ndarray]:
        """Each column's valid values, filtered independently — for UNPAIRED
        comparisons only. Paired analyses must use paired_arrays()."""
        return {col.name: col.valid_values() for col in self.columns}

    def get_column(self, name: str) -> DataColumn | None:
        return next((c for c in self.columns if c.name == name), None)

    def paired_arrays(self, *names: str) -> list[np.ndarray]:
        """Values of the named columns restricted to the rows that are valid
        (present, numeric, not excluded) in ALL of them, so element i of every
        array comes from the same original row (subject / x-y pair)."""
        cols = []
        for nm in names:
            col = self.get_column(nm)
            if col is None:
                raise ValueError(f"Column '{nm}' not found.")
            cols.append(col)
        rows = None
        per_col = []
        for col in cols:
            vals, idx = col.valid_values_with_indices()
            per_col.append(dict(zip(idx, vals)))
            rows = set(idx) if rows is None else rows & set(idx)
        rows = sorted(rows or [])
        return [np.array([d[r] for r in rows], dtype=float) for d in per_col]

    def analysis_dataframe(self) -> pd.DataFrame:
        """to_dataframe() with excluded cells masked to NaN (row alignment kept),
        for analyses; to_dataframe() stays the raw data for export."""
        df = self.to_dataframe()
        for col in self.columns:
            for idx in col.excluded:
                if 0 <= idx < len(df):
                    df.loc[idx, col.name] = np.nan
        return df

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "table_type": self.table_type,
            "columns": [
                {"name": c.name, "values": c.values, "excluded": list(c.excluded)}
                for c in self.columns
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> DataTable:
        t = cls(d["name"], d.get("table_type", "column"))
        for cd in d.get("columns", []):
            col = DataColumn(name=cd["name"], values=cd["values"])
            col.excluded = set(cd.get("excluded", []))
            t.columns.append(col)
        return t


class GroupedDataTable:
    """
    Prism-style Grouped data table.
    Factor A = column groups (col_groups).
    Factor B = row labels (row_labels).
    Each cell contains n_reps replicates.
    """

    def __init__(self, name: str, n_reps: int = 3):
        self.name = name
        self.table_type = "grouped"
        self.col_groups: list[str] = []
        self.row_labels: list[str] = []
        self.n_reps: int = n_reps
        self.data: dict[str, dict[str, list]] = {}
        # data[col_group][row_label] = [val0, val1, …]

    def add_col_group(self, name: str):
        if name not in self.col_groups:
            self.col_groups.append(name)
            self.data[name] = {rl: [None] * self.n_reps for rl in self.row_labels}

    def add_row_label(self, name: str):
        if name not in self.row_labels:
            self.row_labels.append(name)
            for cg in self.col_groups:
                self.data.setdefault(cg, {})[name] = [None] * self.n_reps

    def set_value(self, col_group: str, row_label: str, rep: int, value):
        cell = self.data.setdefault(col_group, {}).setdefault(row_label, [None] * self.n_reps)
        while len(cell) <= rep:
            cell.append(None)
        cell[rep] = value

    def to_long_dataframe(self) -> pd.DataFrame:
        rows = []
        for cg in self.col_groups:
            for rl in self.row_labels:
                for v in self.data.get(cg, {}).get(rl, []):
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        rows.append({"value": float(v), "factor_a": cg, "factor_b": rl})
        return pd.DataFrame(rows)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "table_type": "grouped",
            "col_groups": self.col_groups,
            "row_labels": self.row_labels,
            "n_reps": self.n_reps,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GroupedDataTable:
        t = cls(d["name"], d.get("n_reps", 3))
        t.col_groups = d.get("col_groups", [])
        t.row_labels = d.get("row_labels", [])
        t.data = d.get("data", {})
        return t


class ContingencyTable:
    """
    r×c contingency table for chi-square / Fisher's exact tests.
    self.counts is a list-of-lists [row][col] aligned with row_labels × col_labels.
    """

    def __init__(self, name: str):
        self.name = name
        self.table_type = "contingency"
        self.row_labels: list[str] = []
        self.col_labels: list[str] = []
        self.counts: list[list[int]] = []

    # ------------------------------------------------------------------
    def _ensure_shape(self):
        nrows, ncols = len(self.row_labels), len(self.col_labels)
        while len(self.counts) < nrows:
            self.counts.append([0] * ncols)
        for row in self.counts:
            while len(row) < ncols:
                row.append(0)

    def add_row(self, label: str):
        if label not in self.row_labels:
            self.row_labels.append(label)
            self.counts.append([0] * len(self.col_labels))

    def add_col(self, label: str):
        if label not in self.col_labels:
            self.col_labels.append(label)
            for row in self.counts:
                row.append(0)

    def remove_row(self, label: str):
        if label in self.row_labels:
            idx = self.row_labels.index(label)
            self.row_labels.pop(idx)
            if idx < len(self.counts):
                self.counts.pop(idx)

    def remove_col(self, label: str):
        if label in self.col_labels:
            idx = self.col_labels.index(label)
            self.col_labels.pop(idx)
            for row in self.counts:
                if idx < len(row):
                    row.pop(idx)

    def set_count(self, row_label: str, col_label: str, value: int):
        self._ensure_shape()
        r = self.row_labels.index(row_label)
        c = self.col_labels.index(col_label)
        self.counts[r][c] = int(value)

    def get_count(self, row_label: str, col_label: str) -> int:
        self._ensure_shape()
        r = self.row_labels.index(row_label)
        c = self.col_labels.index(col_label)
        return self.counts[r][c]

    def get_matrix(self) -> np.ndarray:
        self._ensure_shape()
        return np.array(self.counts, dtype=int)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "table_type": "contingency",
            "row_labels": self.row_labels,
            "col_labels": self.col_labels,
            "counts": self.counts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ContingencyTable:
        t = cls(d["name"])
        t.row_labels = d.get("row_labels", [])
        t.col_labels = d.get("col_labels", [])
        t.counts = d.get("counts", [])
        return t


@dataclass
class AnalysisResult:
    name: str
    method: str
    source_table: str
    summary_text: str
    details: dict = field(default_factory=dict)
    run_params: dict = field(default_factory=dict)  # inputs for live re-run

    def to_dict(self) -> dict:
        safe_details = {}
        for k, v in self.details.items():
            safe_details[k] = v.tolist() if hasattr(v, "tolist") else v
        return {
            "name": self.name,
            "method": self.method,
            "source_table": self.source_table,
            "summary_text": self.summary_text,
            "details": safe_details,
            "run_params": self.run_params,
        }


@dataclass
class GraphConfig:
    name: str
    source_table: str
    plot_type: str  # "bar" | "scatter" | "box" | "violin" | "survival"
    options: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "source_table": self.source_table,
            "plot_type": self.plot_type,
            "options": self.options,
        }


class Project:
    def __init__(self, name: str = "Untitled Project"):
        self.name = name
        self.filepath: Optional[str] = None
        self.tables: list[DataTable] = []
        self.analyses: list[AnalysisResult] = []
        self.graphs: list[GraphConfig] = []

    def _table_name(self, name):
        # graphs / analyses find their table by name: keep names unique
        return unique_name(name, [t.name for t in self.tables])

    def add_table(self, name: str | None = None) -> DataTable:
        if name is None:
            name = f"Data {len(self.tables) + 1}"
        t = DataTable(name=self._table_name(name))
        self.tables.append(t)
        return t

    def add_grouped_table(self, name: str | None = None, n_reps: int = 3) -> GroupedDataTable:
        if name is None:
            name = f"Grouped {len(self.tables) + 1}"
        t = GroupedDataTable(name=self._table_name(name), n_reps=n_reps)
        self.tables.append(t)
        return t

    def add_contingency_table(self, name: str | None = None) -> ContingencyTable:
        if name is None:
            name = f"Contingency {len(self.tables) + 1}"
        t = ContingencyTable(name=self._table_name(name))
        self.tables.append(t)
        return t

    def get_table(self, name: str) -> Optional[DataTable | GroupedDataTable | ContingencyTable]:
        return next((t for t in self.tables if t.name == name), None)

    def add_graph(
        self, name: str, source_table: str, plot_type: str, options: dict | None = None
    ) -> GraphConfig:
        g = GraphConfig(name=name, source_table=source_table, plot_type=plot_type, options=options or {})
        self.graphs.append(g)
        return g

    def save(self, filepath: str):
        """Write atomically: a failed save (full disk, crash, serialisation
        error) leaves the previous file intact."""
        import os
        import tempfile
        data = {
            "name": self.name,
            "tables": [t.to_dict() for t in self.tables],
            "analyses": [a.to_dict() for a in self.analyses],
            "graphs": [g.to_dict() for g in self.graphs],
        }
        folder = os.path.dirname(os.path.abspath(filepath))
        fd, tmp = tempfile.mkstemp(prefix=".staplot_", suffix=".tmp", dir=folder)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, filepath)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        self.filepath = filepath

    @classmethod
    def load(cls, filepath: str) -> Project:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        p = cls(data.get("name", "Untitled"))
        p.filepath = filepath
        for td in data.get("tables", []):
            tt = td.get("table_type", "column")
            if tt == "grouped":
                p.tables.append(GroupedDataTable.from_dict(td))
            elif tt == "contingency":
                p.tables.append(ContingencyTable.from_dict(td))
            else:
                p.tables.append(DataTable.from_dict(td))
        for ad in data.get("analyses", []):
            r = AnalysisResult(
                name=ad["name"],
                method=ad["method"],
                source_table=ad["source_table"],
                summary_text=ad["summary_text"],
                details=ad.get("details", {}),
                run_params=ad.get("run_params", {}),
            )
            p.analyses.append(r)
        for gd in data.get("graphs", []):
            p.graphs.append(GraphConfig(**gd))
        return p
