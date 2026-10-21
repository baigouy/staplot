"""Computed-column dialog — add a column defined by a formula."""

import ast

import numpy as np
from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QComboBox, QLineEdit,
    QPushButton, QTextEdit, QDialogButtonBox, QLabel, QHBoxLayout,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont

from ..models.project import DataTable

# Safe function namespace exposed in formulas
_SAFE_NS: dict = {
    "log":   np.log,
    "log2":  np.log2,
    "log10": np.log10,
    "ln":    np.log,
    "sqrt":  np.sqrt,
    "abs":   np.abs,
    "exp":   np.exp,
    "sin":   np.sin,
    "cos":   np.cos,
    "tan":   np.tan,
    "floor": np.floor,
    "ceil":  np.ceil,
    "round": np.round,
    "pi":    np.pi,
    "e":     np.e,
    "nan":   np.nan,
    "inf":   np.inf,
}


def _sanitize_name(name: str) -> str:
    """Turn column names into valid Python identifiers."""
    import re
    s = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if s and s[0].isdigit():
        s = "_" + s
    return s


_CONSTANTS = {k: v for k, v in _SAFE_NS.items() if not callable(v)}
_FUNCTIONS = {k: v for k, v in _SAFE_NS.items() if callable(v)}


def _formula_namespace(col_data: dict[str, np.ndarray]) -> tuple[dict, int]:
    """Columns by sanitised name (padded with NaN to a common length), plus
    the constants. Refuses names that collide with each other or shadow a
    function / constant, rather than silently picking one."""
    n = max((len(a) for a in col_data.values()), default=0)
    ns: dict = dict(_CONSTANTS)
    origin: dict = {}
    for raw_name, arr in col_data.items():
        s = _sanitize_name(raw_name)
        if not s:
            raise ValueError(f"column name {raw_name!r} cannot be used in a formula")
        if s in _SAFE_NS:
            raise ValueError(f"column {raw_name!r} shadows the function/constant '{s}'; "
                             "rename the column")
        if s in origin:
            raise ValueError(f"columns {origin[s]!r} and {raw_name!r} both become '{s}'; "
                             "rename one of them")
        origin[s] = raw_name
        a = np.full(n, np.nan)
        a[:len(arr)] = np.asarray(arr, dtype=float)
        ns[s] = a
    return ns, n


_BINOPS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
           ast.Div: np.true_divide, ast.Pow: np.power, ast.Mod: np.mod,
           ast.FloorDiv: np.floor_divide}
_UNARY = {ast.USub: np.negative, ast.UAdd: np.positive}


def _eval_node(node, ns):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, ns)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in ns:
            raise ValueError(f"unknown name '{node.id}'")
        return ns[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left, ns), _eval_node(node.right, ns))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval_node(node.operand, ns))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _FUNCTIONS and not node.keywords:
        return _FUNCTIONS[node.func.id](*[_eval_node(a, ns) for a in node.args])
    raise ValueError(f"not allowed in a formula: {ast.unparse(node) if hasattr(ast, 'unparse') else type(node).__name__}")


def eval_formula(formula: str, col_data: dict[str, np.ndarray]) -> np.ndarray:
    """
    Evaluate *formula* with each column available as a numpy array (names
    sanitised to identifiers). Only arithmetic, numbers, column names and the
    listed functions/constants are accepted (AST allow-list — no Python eval).
    Returns a float array with one value per table row.
    """
    ns, n = _formula_namespace(col_data)
    try:
        tree = ast.parse(formula.strip(), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"invalid formula: {e.msg}") from None
    with np.errstate(all="ignore"):
        result = np.asarray(_eval_node(tree, ns), dtype=float)
    if result.ndim == 0:
        result = np.full(n, float(result))
    if result.ndim != 1 or len(result) != n:
        raise ValueError(f"the formula must give one value per row ({n}), got shape {result.shape}")
    return result


class ComputedColumnDialog(QDialog):
    """
    Add a new column to an existing DataTable using a numpy formula.

    Examples
    --------
    ``A / B``          divide column A by column B
    ``log2(A / B)``    log2-fold change
    ``sqrt(A)``        square root
    ``A * 100``        percentage scale
    ``(A - B) / B * 100``   % change
    """

    def __init__(self, tables: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Computed Column")
        self.setMinimumWidth(480)

        self._tables = [t for t in tables if isinstance(t, DataTable)]
        self._new_col = None   # result DataTable ref (unused — col added in place)

        layout = QVBoxLayout(self)

        # ── Table / column name selectors ────────────────────────────
        form = QFormLayout()

        self._table_cb = QComboBox()
        self._table_cb.addItems([t.name for t in self._tables])
        self._table_cb.currentIndexChanged.connect(self._refresh_hint)
        form.addRow("Target table:", self._table_cb)

        self._col_name = QLineEdit("Result")
        form.addRow("New column name:", self._col_name)

        self._formula = QLineEdit()
        self._formula.setPlaceholderText("e.g.  A / B  or  log2(A)")
        self._formula.textChanged.connect(self._preview)
        form.addRow("Formula:", self._formula)
        layout.addLayout(form)

        # ── Available columns hint ────────────────────────────────────
        self._hint_lbl = QLabel()
        self._hint_lbl.setStyleSheet("color: #555; font-size: 10px;")
        self._hint_lbl.setWordWrap(True)
        layout.addWidget(self._hint_lbl)

        # ── Functions hint ────────────────────────────────────────────
        funcs_lbl = QLabel(
            "Functions: log, log2, log10, sqrt, abs, exp, sin, cos, tan, "
            "floor, ceil, round\n"
            "Constants: pi, e, nan, inf"
        )
        funcs_lbl.setStyleSheet("color: #777; font-size: 10px;")
        funcs_lbl.setWordWrap(True)
        layout.addWidget(funcs_lbl)

        # ── Preview ───────────────────────────────────────────────────
        preview_btn = QPushButton("Preview (first 8 values)")
        preview_btn.clicked.connect(self._preview)
        layout.addWidget(preview_btn)

        self._preview_box = QTextEdit()
        self._preview_box.setReadOnly(True)
        mono = QFont("Courier New", 8)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self._preview_box.setFont(mono)
        self._preview_box.setMaximumHeight(100)
        layout.addWidget(self._preview_box)

        # ── Buttons ───────────────────────────────────────────────────
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.button(QDialogButtonBox.StandardButton.Ok).setText("Add Column")
        btns.accepted.connect(self._add_column)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self._refresh_hint()

    # ------------------------------------------------------------------
    def _current_table(self) -> DataTable | None:
        idx = self._table_cb.currentIndex()
        return self._tables[idx] if 0 <= idx < len(self._tables) else None

    def _col_data(self) -> dict[str, np.ndarray]:
        t = self._current_table()
        if t is None:
            return {}
        return {c.name: c.as_array() for c in t.columns}

    def _refresh_hint(self):
        t = self._current_table()
        if t is None:
            self._hint_lbl.setText("")
            return
        names = [c.name for c in t.columns]
        safe  = [_sanitize_name(n) for n in names]
        pairs = "  |  ".join(
            f"{raw} → {s}" if raw != s else raw
            for raw, s in zip(names, safe)
        )
        self._hint_lbl.setText(f"Available columns:  {pairs}")

    def _preview(self):
        formula = self._formula.text().strip()
        if not formula:
            self._preview_box.clear()
            return
        try:
            result = eval_formula(formula, self._col_data())
            preview = result[:8]
            lines = [f"[{i}]  {v:.6g}" for i, v in enumerate(preview)]
            self._preview_box.setPlainText(
                "\n".join(lines) + (f"\n… ({len(result)} values total)" if len(result) > 8 else "")
            )
        except Exception as exc:
            self._preview_box.setPlainText(f"Error: {exc}")

    def _add_column(self):
        t = self._current_table()
        if t is None:
            return
        formula   = self._formula.text().strip()
        col_name  = self._col_name.text().strip() or "Result"
        if not formula:
            self._preview_box.setPlainText("Enter a formula.")
            return
        try:
            result = eval_formula(formula, self._col_data())
        except Exception as exc:
            self._preview_box.setPlainText(f"Error: {exc}")
            return

        new_col = t.add_column(col_name)
        new_col.values = [None if not np.isfinite(v) else float(v) for v in result]
        self.accept()

    def get_table(self) -> DataTable | None:
        return self._current_table()
