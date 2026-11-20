"""Visual editor of one staplot plot (staplot.api.plot): the data on the left, the plot on
the right. Click anything in the plot - a group's bar / box / violin, the points, the
median, the brackets, the title, an axis label, the tick labels, the legend, the empty
plot area - and its settings show under the plot: colour, text, size, font, axis range.

Everything is kept as the arguments of staplot.api.plot (``kwargs()``) plus the size and
font of each part of the text (``text_parts()``: title, axis_labels, ticks, legend), so
the caller (PyFigures, a script) stores them as it stores any plot setting.

    from staplot.ui.plot_editor import PlotEditor
    ed = PlotEditor("counts.csv", "box", {"error": "sem"})
    if ed.exec():
        print(ed.kind(), ed.changed_kwargs(), ed.text_parts())

Run alone: ``python -m staplot.ui.plot_editor table.csv [kind]``.
"""
from __future__ import annotations

import copy
import os
import sys
import tempfile

from qtpy.QtCore import QAbstractTableModel, QModelIndex, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QFont, QIcon, QKeySequence
from qtpy.QtWidgets import (
    QApplication, QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFontComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QSplitter, QTableView, QToolBar, QVBoxLayout, QWidget,
)

try:
    from qtpy.QtGui import QAction, QShortcut
except ImportError:                                    # PyQt5
    from qtpy.QtWidgets import QAction, QShortcut      # type: ignore[no-redef]

PLOT_KINDS = ("bar", "box", "violin", "scatter", "grouped", "split_violin", "paired", "line")
TWO_FACTOR_KINDS = ("grouped", "split_violin", "line")
ERRORS = ("sd", "sem", "ci95", "none")
TEXT_PARTS = ("title", "axis_labels", "ticks", "legend")
PART_NAMES = {"title": "titles", "axis_labels": "axis labels", "ticks": "tick labels",
              "legend": "legend texts"}
GROUP_PARTS = ("bar", "box", "violin")


def icon(name: str) -> QIcon:
    """A qtawesome icon, or none when qtawesome is missing (it is optional)."""
    try:
        import qtawesome
        return qtawesome.icon(name)
    except Exception:
        return QIcon()


# ---------------------------------------------------------------------------
# the plot's texts, by part (the same parts as PyFigures' publication_text_parts)
# ---------------------------------------------------------------------------
def text_parts(fig) -> list:
    """[(Text, part)] of every visible text of *fig*: titles, axis labels, tick labels,
    legend texts."""
    out = []
    for name, part in (("_suptitle", "title"), ("_supxlabel", "axis_labels"), ("_supylabel", "axis_labels")):
        t = getattr(fig, name, None)
        if t is not None:
            out.append((t, part))
    legends = list(getattr(fig, "legends", []))
    for ax in fig.axes:
        for t in (ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None)):
            if t is not None:
                out.append((t, "title"))
        out += [(ax.xaxis.label, "axis_labels"), (ax.yaxis.label, "axis_labels")]
        out += [(t, "ticks") for t in ax.get_xticklabels() + ax.get_yticklabels()]
        if ax.get_legend() is not None:
            legends.append(ax.get_legend())
    for lg in legends:
        out += [(t, "legend") for t in lg.get_texts()]
        out.append((lg.get_title(), "legend"))
    return [(t, p) for t, p in out if t.get_visible() and t.get_text().strip()]


def apply_text_parts(fig, parts: dict) -> None:
    """Size ('pt') and font ('font') of each part of the text, as PyFigures draws them."""
    if not parts:
        return
    for ax in fig.axes:
        pt = (parts.get("ticks") or {}).get("pt")
        if pt:
            ax.tick_params(labelsize=pt)
    for t, part in text_parts(fig):
        style = parts.get(part) or {}
        if style.get("pt"):
            t.set_fontsize(style["pt"])
        if style.get("font"):
            t.set_fontfamily(style["font"])


def _first_color(artist):
    """The fill colour of *artist* as an RGB tuple (rounded), or None."""
    from matplotlib.colors import to_rgb
    try:
        fc = artist.get_facecolor()
    except Exception:
        return None
    try:
        if hasattr(fc, "ndim") and fc.ndim == 2:
            if not len(fc):
                return None
            fc = fc[0]
        return tuple(round(v, 3) for v in to_rgb(fc))
    except (ValueError, TypeError):
        return None


def _hex(color) -> str:
    from matplotlib.colors import to_hex
    try:
        return to_hex(color)
    except (ValueError, TypeError):
        return "#808080"


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------
class DataFrameModel(QAbstractTableModel):
    """A pandas DataFrame in a QTableView, editable in place (numbers stay numbers)."""
    edited = Signal()

    def __init__(self, df, editable: bool = True, parent=None):
        super().__init__(parent)
        self.df = df
        self.editable = editable

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.df)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.df.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return None
        v = self.df.iat[index.row(), index.column()]
        if v is None or (isinstance(v, float) and v != v):
            return ""
        if isinstance(v, float) and role == Qt.ItemDataRole.DisplayRole:
            return f"{v:.6g}"
        return str(v)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self.df.columns[section])
        return str(section + 1)

    def flags(self, index):
        f = super().flags(index)
        return f | Qt.ItemFlag.ItemIsEditable if self.editable else f

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if role != Qt.ItemDataRole.EditRole or not self.editable:
            return False
        text = str(value).strip()
        if text == "":
            new = float("nan")
        else:
            try:
                new = float(text.replace(",", "."))
            except ValueError:
                new = text
        col = self.df.columns[index.column()]
        if isinstance(new, str) and self.df[col].dtype.kind in "fiub":
            self.df[col] = self.df[col].astype(object)
        self.df.iat[index.row(), index.column()] = new
        self.dataChanged.emit(index, index)
        self.edited.emit()
        return True


class ColorButton(QPushButton):
    """A button showing a colour; click to pick another."""
    picked = Signal(str)

    def __init__(self, color: str, parent=None):
        super().__init__(parent)
        self.setFixedWidth(64)
        self.set_color(color)
        self.clicked.connect(self._pick)

    def set_color(self, color):
        self.color = _hex(color)
        self.setStyleSheet(f"background-color: {self.color}; border: 1px solid #888; min-height: 18px;")

    def _pick(self):
        c = QColorDialog.getColor(QColor(self.color), self, "Colour")
        if c.isValid():
            self.set_color(c.name())
            self.picked.emit(c.name())


# ---------------------------------------------------------------------------
# the editor
# ---------------------------------------------------------------------------
class PlotEditor(QDialog):
    """The data (left) and the plot (right) of one staplot plot; click a part of the plot
    to change it. *kwargs*: staplot.api.plot's arguments (colors, element_colors, title,
    ylim...); *table_kwargs*: how the table is read (columns, where, skip_rows); *facet*:
    plot_facets' arguments (facet, facet_row, ncols) or None; *parts*: the size / font of
    each part of the text {part: {'pt', 'font'}}."""

    def __init__(self, path: str, kind: str = "bar", kwargs: dict | None = None,
                 table_kwargs: dict | None = None, facet: dict | None = None,
                 parts: dict | None = None, parent=None, title: str = "Edit the plot"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1250, 720)
        self.path = path
        self._kind = kind if kind in PLOT_KINDS else "bar"
        self._kwargs = dict(kwargs or {})
        self._table_kwargs = {k: v for k, v in (table_kwargs or {}).items() if v not in (None, "", [])}
        self._facet = {k: v for k, v in (facet or {}).items() if v}
        self._parts = {p: dict(s) for p, s in (parts or {}).items() if p in TEXT_PARTS and s}
        self._start = self._state()
        self._history, self._pos = [self._state()], 0
        self.fig = None
        self.canvas = None
        self._selection = None
        self._highlight = None
        self._temp_csv = None
        self.data_saved = False
        self._data_dirty = False
        self._render_timer = QTimer(self, singleShot=True, interval=400, timeout=self.render)
        self._build()
        self._load_table()
        self.render()

    # -- state ------------------------------------------------------------------------
    def _state(self):
        return copy.deepcopy((self._kind, self._kwargs, self._parts))

    def kind(self) -> str:
        return self._kind

    def kwargs(self) -> dict:
        return dict(self._kwargs)

    def text_parts(self) -> dict:
        return {p: dict(s) for p, s in self._parts.items() if s}

    def changed_kwargs(self) -> dict:
        """The plot arguments changed since the editor opened (None: back to the default)."""
        start = self._start[1]
        out = {k: v for k, v in self._kwargs.items() if start.get(k) != v}
        out.update({k: None for k in start if k not in self._kwargs})
        return out

    def changed_text_parts(self) -> dict:
        """{part: {'pt', 'font'}} of the parts changed (None in a field: back to the default)."""
        start, out = self._start[2], {}
        for p in TEXT_PARTS:
            a, b = start.get(p) or {}, self._parts.get(p) or {}
            diff = {k: b.get(k) for k in ("pt", "font") if a.get(k) != b.get(k)}
            if diff:
                out[p] = diff
        return out

    def _change(self, fn):
        """Apply *fn* to the settings, keep the step for undo, redraw."""
        fn()
        self._history = self._history[:self._pos + 1] + [self._state()]
        self._pos = len(self._history) - 1
        self._sync_toolbar()
        self.render()

    def set(self, key, value):
        """One plot argument (None or "" removes it: staplot's default)."""
        def fn():
            if value is None or value == "":
                self._kwargs.pop(key, None)
            else:
                self._kwargs[key] = value
        self._change(fn)

    def set_in(self, key, sub, value):
        """One entry of a dict argument (colors, element_colors, rename)."""
        def fn():
            d = self._kwargs.get(key)
            d = dict(d) if isinstance(d, dict) else ({} if not d else self._colors_as_dict(d))
            if value in (None, ""):
                d.pop(sub, None)
            else:
                d[sub] = value
            if d:
                self._kwargs[key] = d
            else:
                self._kwargs.pop(key, None)
        self._change(fn)

    def set_part(self, part, field, value):
        def fn():
            s = dict(self._parts.get(part) or {})
            if value in (None, "", 0):
                s.pop(field, None)
            else:
                s[field] = value
            if s:
                self._parts[part] = s
            else:
                self._parts.pop(part, None)
        self._change(fn)

    def _colors_as_dict(self, colors):
        """A list of colours as {group: colour}, the groups as the plot shows them."""
        names = self._group_names()
        return {n: c for n, c in zip(names, colors)}

    def undo(self):
        if self._pos > 0:
            self._pos -= 1
            self._kind, self._kwargs, self._parts = copy.deepcopy(self._history[self._pos])
            self._sync_toolbar()
            self.render()

    def redo(self):
        if self._pos < len(self._history) - 1:
            self._pos += 1
            self._kind, self._kwargs, self._parts = copy.deepcopy(self._history[self._pos])
            self._sync_toolbar()
            self.render()

    # -- layout -----------------------------------------------------------------------
    def _build(self):
        root = QVBoxLayout(self)
        tb = QToolBar(self)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        root.addWidget(tb)
        self.undo_act = QAction(icon("fa5s.undo"), "Undo", self, triggered=self.undo)
        self.redo_act = QAction(icon("fa5s.redo"), "Redo", self, triggered=self.redo)
        self.undo_act.setShortcut(QKeySequence(QKeySequence.StandardKey.Undo))
        self.redo_act.setShortcut(QKeySequence(QKeySequence.StandardKey.Redo))
        tb.addAction(self.undo_act)
        tb.addAction(self.redo_act)
        tb.addSeparator()
        tb.addWidget(QLabel(" Plot: "))
        self.kind_combo = QComboBox(self)
        self.kind_combo.addItems(PLOT_KINDS)
        self.kind_combo.activated.connect(lambda _i: self._set_kind(self.kind_combo.currentText()))
        tb.addWidget(self.kind_combo)
        tb.addWidget(QLabel("  Error bars: "))
        self.error_combo = QComboBox(self)
        self.error_combo.addItems(ERRORS)
        self.error_combo.activated.connect(lambda _i: self.set("error", self.error_combo.currentText()))
        tb.addWidget(self.error_combo)
        tb.addSeparator()
        self.stats_check = QCheckBox("Statistics", self)
        self.stats_check.clicked.connect(lambda on: self.set("annotate", None if on else False))
        tb.addWidget(self.stats_check)
        self.points_check = QCheckBox("Points", self)
        self.points_check.clicked.connect(lambda on: self.set("show_points", None if on else False))
        tb.addWidget(self.points_check)
        tb.addSeparator()
        self.save_data_act = QAction(icon("fa5s.save"), "Save the data", self, triggered=self.save_data)
        tb.addAction(self.save_data_act)
        self.reset_act = QAction(icon("fa5s.history"), "Back to the start", self, triggered=self.reset)
        tb.addAction(self.reset_act)

        split = QSplitter(Qt.Orientation.Horizontal, self)
        root.addWidget(split, 1)
        left = QWidget(self)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.table_note = QLabel(self)
        self.table_note.setWordWrap(True)
        lv.addWidget(self.table_note)
        self.table = QTableView(self)
        self.table.setAlternatingRowColors(True)
        lv.addWidget(self.table, 1)
        split.addWidget(left)

        right = QWidget(self)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.canvas_box = QVBoxLayout()
        rv.addLayout(self.canvas_box, 1)
        self.message = QLabel(self)
        self.message.setWordWrap(True)
        self.message.setStyleSheet("color: #b00;")
        rv.addWidget(self.message)
        self.inspector = QGroupBox("Click a part of the plot to change it", self)
        self.inspector_form = QFormLayout(self.inspector)
        rv.addWidget(self.inspector)
        split.addWidget(right)
        split.setSizes([500, 750])

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                        | QDialogButtonBox.StandardButton.Cancel, self)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setIcon(icon("fa5s.check"))
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setIcon(icon("fa5s.times"))
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)
        self._sync_toolbar()

    def add_button(self, text: str, slot, icon_name: str = "") -> QPushButton:
        """An extra button next to OK / Cancel (PyFigures: its annotations and script)."""
        b = self.buttons.addButton(text, QDialogButtonBox.ButtonRole.ActionRole)
        if icon_name:
            b.setIcon(icon(icon_name))
        b.clicked.connect(slot)
        return b

    def _sync_toolbar(self):
        self.kind_combo.setCurrentText(self._kind)
        self.error_combo.setCurrentText(self._kwargs.get("error") or "sd")
        self.stats_check.setChecked(self._kwargs.get("annotate", True) is not False)
        self.points_check.setChecked(self._kwargs.get("show_points", True) is not False)
        self.undo_act.setEnabled(self._pos > 0)
        self.redo_act.setEnabled(self._pos < len(self._history) - 1)

    def _set_kind(self, kind):
        def fn():
            self._kind = kind
        self._change(fn)

    def reset(self):
        def fn():
            self._kind, self._kwargs, self._parts = copy.deepcopy(self._start)
        self._change(fn)

    # -- data -------------------------------------------------------------------------
    def _load_table(self):
        from staplot import api
        file, _sheet = api.split_sheet(self.path)
        self._editable = not (api.is_excel(file) or api.is_sqlite(file))
        try:
            self.df = api.read_table(self.path, self._table_kwargs.get("skip_rows"))
        except Exception as e:
            import pandas as pd
            self.df = pd.DataFrame()
            self.table_note.setText(f"Cannot read {os.path.basename(file)}: {e}")
            self._editable = False
        else:
            self.table_note.setText(
                f"<b>{os.path.basename(self.path)}</b> - " + (
                    "double-click a cell to change it; the plot follows. <i>Save the data</i> writes the file."
                    if self._editable else "a workbook or database: shown, not edited here."))
        self.model = DataFrameModel(self.df, self._editable, self)
        self.model.edited.connect(self._on_data_edited)
        self.table.setModel(self.model)
        self.save_data_act.setEnabled(False)

    def _on_data_edited(self):
        self._data_dirty = True
        self.save_data_act.setEnabled(True)
        self._render_timer.start()

    def _source(self) -> tuple[str, dict]:
        """The file the plot reads and how: the edited table (a temporary copy) or the file."""
        if not self._data_dirty:
            return self.path, dict(self._table_kwargs)
        if self._temp_csv is None:
            fd, self._temp_csv = tempfile.mkstemp(suffix=".csv", prefix="staplot_edit_")
            os.close(fd)
        self.df.to_csv(self._temp_csv, index=False)
        tk = dict(self._table_kwargs)
        tk["skip_rows"] = 0
        return self._temp_csv, tk

    def save_data(self, confirm: bool = True):
        """Write the edited table to its file (lines above the table and the separator kept)."""
        from staplot import api
        if not (self._editable and self._data_dirty):
            return
        if confirm and QMessageBox.question(self, "Save the data", f"Overwrite {self.path} with the edited table?") \
                != QMessageBox.StandardButton.Yes:
            return
        with open(self.path, encoding="utf-8-sig", errors="replace", newline="") as f:
            head = "".join(line for _i, line in zip(range(400), f))
        _skip, sep, decimal = api._text_preamble(head, self._table_kwargs.get("skip_rows"))
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            for line in self.df.attrs.get("skipped_lines") or []:
                f.write(line.rstrip("\r\n") + "\n")
            self.df.to_csv(f, index=False, sep=sep, decimal=decimal, lineterminator="\n")
        self._data_dirty = False
        self.data_saved = True
        self.save_data_act.setEnabled(False)
        self.render()

    # -- drawing ----------------------------------------------------------------------
    def make_figure(self):
        """The plot with the current settings: (figure, statistics record)."""
        from staplot import api
        import matplotlib.pyplot as plt
        path, tk = self._source()
        try:
            if self._facet:
                fig, rec = api.plot_facets(path, self._kind, **self._facet, **tk, **self._kwargs)
            else:
                builder = api.grouped_table_from_csv if self._kind in TWO_FACTOR_KINDS else api.table_from_csv
                fig, rec = api.plot(builder(path, **tk), kind=self._kind, **self._kwargs)
        finally:
            pass
        plt.close(fig)                  # out of pyplot's list; the canvas below shows it
        return fig, rec

    def render(self):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        try:
            fig, _rec = self.make_figure()
        except Exception as e:
            self.message.setText(f"The plot cannot be drawn with these settings: {e}")
            return
        self.message.setText("")
        apply_text_parts(fig, self._parts)
        if self.canvas is not None:
            self.canvas_box.removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.canvas.deleteLater()
        self.fig = fig
        self.canvas = FigureCanvasQTAgg(fig)
        self.canvas.setToolTip("Click a part of the plot to change it")
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas_box.addWidget(self.canvas)
        self._highlight = None
        self.canvas.draw()
        if self._selection is not None:
            # the same part, found again in the new figure (its text / settings changed)
            self._select(self._selection[0], self._selection[1], None)

    # -- clicking ---------------------------------------------------------------------
    def hit(self, x, y, inaxes=None):
        """What is under the display point (x, y): (kind, key, artist) with kind 'text'
        (key: part), 'group' (key: group name), 'element' (key: PLOT_ELEMENTS name),
        'axes' or 'figure'."""
        from staplot import api
        from matplotlib.backend_bases import MouseEvent
        r = self.canvas.get_renderer()
        for t, part in text_parts(self.fig):
            try:
                if t.get_window_extent(r).contains(x, y):
                    return "text", part, t
            except Exception:
                continue
        ev = MouseEvent("button_press_event", self.canvas, x, y)
        groups = []
        for a in self.fig.findobj():
            part = api.plot_part(a)
            if part is None or not a.get_visible():
                continue
            try:
                inside = a.contains(ev)[0]
            except Exception:
                continue
            if not inside:
                continue
            if part in GROUP_PARTS:
                groups.append(a)
            else:
                return "element", part, a
        for ax in self.fig.axes:
            for p in ax.patches:
                if p.get_visible() and p not in groups and api.plot_part(p) is None:
                    try:
                        if p.contains(ev)[0] and p.get_width() > 0:
                            groups.append(p)
                    except Exception:
                        continue
        if groups:
            a = groups[-1]
            return "group", self._group_of(a), a
        if inaxes is not None:
            return "axes", None, inaxes
        return "figure", None, None

    def _group_of(self, artist) -> str:
        """The group an artist (bar, box, violin) draws: by its colour in the legend
        (two-factor plots: the colour is the group), else by the x tick under it."""
        ax = artist.axes
        legends = ([ax.get_legend()] if ax.get_legend() else []) + list(self.fig.legends)
        color = _first_color(artist)
        if self._kind in TWO_FACTOR_KINDS and color is not None:
            for lg in legends:
                for h, t in zip(getattr(lg, "legend_handles", None) or getattr(lg, "legendHandles", []),
                                lg.get_texts()):
                    if _first_color(h) == color:
                        return t.get_text()
        bb = artist.get_window_extent(self.canvas.get_renderer())
        x = ax.transData.inverted().transform(((bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2))[0]
        ticks = list(ax.get_xticks())
        labels = [t.get_text() for t in ax.get_xticklabels()]
        if not ticks or len(labels) != len(ticks):
            return ""
        i = min(range(len(ticks)), key=lambda k: abs(ticks[k] - x))
        return labels[i].split("\n")[0]

    def _group_names(self) -> list:
        if self.fig is None or not self.fig.axes:
            return []
        return [t.get_text().split("\n")[0] for t in self.fig.axes[0].get_xticklabels()]

    def _on_click(self, event):
        if event.button != 1 or self.fig is None:
            return
        kind, key, artist = self.hit(event.x, event.y, event.inaxes)
        self._select(kind, key, artist)

    def select_at(self, x, y):
        """Select what is at display point (x, y) (tests, scripts)."""
        ax = next((a for a in self.fig.axes if a.bbox.contains(x, y)), None)
        kind, key, artist = self.hit(x, y, ax)
        self._select(kind, key, artist)
        return kind, key

    def _select(self, kind, key, artist):
        self._selection = (kind, key)
        self._draw_highlight(artist)
        while self.inspector_form.rowCount():
            self.inspector_form.removeRow(0)
        getattr(self, f"_inspect_{kind}")(key, artist)

    def _draw_highlight(self, artist):
        from matplotlib.patches import Rectangle
        if self._highlight is not None:
            try:
                self._highlight.remove()
            except Exception:
                pass
            self._highlight = None
        if artist is None or self.canvas is None:
            self.canvas.draw_idle() if self.canvas is not None else None
            return
        try:
            bb = artist.get_window_extent(self.canvas.get_renderer()).transformed(self.fig.transFigure.inverted())
        except Exception:
            return
        pad = 0.004
        self._highlight = Rectangle((bb.x0 - pad, bb.y0 - pad), bb.width + 2 * pad, bb.height + 2 * pad,
                                    transform=self.fig.transFigure, fill=False, edgecolor="#1e88e5",
                                    linewidth=1.5, linestyle="--", zorder=1000)
        self.fig.add_artist(self._highlight)
        self.canvas.draw_idle()

    # -- the settings of what is selected -----------------------------------------------
    def _title_row(self, text):
        self.inspector.setTitle(text)

    def _color_row(self, label, current, on_pick, default_slot=None):
        row = QWidget(self)
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        b = ColorButton(current, row)
        b.picked.connect(on_pick)
        h.addWidget(b)
        if default_slot is not None:
            d = QPushButton(icon("fa5s.eraser"), "Default", row)
            d.clicked.connect(default_slot)
            h.addWidget(d)
        h.addStretch(1)
        self.inspector_form.addRow(label, row)
        return b

    def _line_row(self, label, text, on_done, hint=""):
        w = QLineEdit(text or "", self)
        w.setPlaceholderText(hint)
        w.editingFinished.connect(lambda: on_done(w.text().strip()) if w.text().strip() != (text or "") else None)
        self.inspector_form.addRow(label, w)
        return w

    def _inspect_text(self, part, t):
        self._title_row(f"Text: the {PART_NAMES[part]}")
        # an axis label's .axes is not always set: its axes is the one owning it
        ax = next((a for a in self.fig.axes if t is not None and (
            t in (a.xaxis.label, a.yaxis.label, a.title) or t in a.get_xticklabels())), None)
        if t is not None and ax is not None and t is ax.yaxis.label:
            self._line_row("Y-axis label:", self._kwargs.get("ylabel") or t.get_text(),
                           lambda v: self.set("ylabel", v))
            show = QCheckBox("Show it", self)
            show.setChecked(self._kwargs.get("show_ylabel", True) is not False)
            show.clicked.connect(lambda on: self.set("show_ylabel", None if on else False))
            self.inspector_form.addRow("", show)
        elif t is not None and ax is not None and t is ax.xaxis.label:
            self._line_row("X-axis label:", self._kwargs.get("xlabel") or t.get_text(),
                           lambda v: self.set("xlabel", v))
            show = QCheckBox("Show it", self)
            show.setChecked(self._kwargs.get("show_xlabel", True) is not False)
            show.clicked.connect(lambda on: self.set("show_xlabel", None if on else False))
            self.inspector_form.addRow("", show)
        elif part == "title":
            self._line_row("Title:", self._kwargs.get("title") or (t.get_text() if t is not None else ""),
                           lambda v: self._set_title(v), "no title")
        elif part == "ticks" and t is not None and ax is not None and t in ax.get_xticklabels():
            name = t.get_text().split("\n")[0]
            original = self._original_name(name)
            self._line_row(f"Name of group {original!r}:", name,
                           lambda v, o=original: self.set_in("rename", o, v if v != o else None))
            rot = QDoubleSpinBox(self)
            rot.setRange(-90, 90)
            rot.setValue(float(self._kwargs.get("xtick_rotation") or 0))
            rot.editingFinished.connect(lambda: self.set("xtick_rotation", rot.value() or None))
            self.inspector_form.addRow("Tilt of the group names (°):", rot)
        elif part == "legend":
            combo = QComboBox(self)
            combo.addItems(["inside", "outside", "none"])
            combo.setCurrentText(self._kwargs.get("legend") or "inside")
            combo.activated.connect(lambda _i: self.set("legend", combo.currentText()))
            self.inspector_form.addRow("Legend:", combo)
        style = self._parts.get(part) or {}
        size = QDoubleSpinBox(self)
        size.setRange(0, 72)
        size.setDecimals(1)
        size.setSpecialValueText("as drawn")
        size.setValue(float(style.get("pt") or 0))
        size.editingFinished.connect(lambda: self.set_part(part, "pt", size.value() or None))
        self.inspector_form.addRow(f"Size of all {PART_NAMES[part]} (pt):", size)
        fonts = QWidget(self)
        h = QHBoxLayout(fonts)
        h.setContentsMargins(0, 0, 0, 0)
        font = QFontComboBox(fonts)
        if style.get("font"):
            font.setCurrentFont(QFont(style["font"]))
        else:
            font.setCurrentIndex(-1)
            font.setEditText("as drawn")
        font.activated.connect(lambda _i: self.set_part(part, "font", font.currentFont().family()))
        h.addWidget(font, 1)
        d = QPushButton(icon("fa5s.eraser"), "Default", fonts)
        d.clicked.connect(lambda: self.set_part(part, "font", None))
        h.addWidget(d)
        self.inspector_form.addRow(f"Font of all {PART_NAMES[part]}:", fonts)
        if part in ("axis_labels", "ticks"):
            self._element_color_row("axes", "Colour (axes, ticks, labels):")
        self.inspector_form.addRow(QLabel("<i>Size and font apply to every text of this kind, "
                                          "as PyFigures' graph text settings.</i>"))

    def _set_title(self, v):
        def fn():
            if v:
                self._kwargs["title"] = v
                self._kwargs["show_title"] = True
            else:
                self._kwargs.pop("title", None)
                self._kwargs["show_title"] = False
        self._change(fn)

    def _original_name(self, shown):
        for orig, new in (self._kwargs.get("rename") or {}).items():
            if new == shown:
                return orig
        return shown

    def _inspect_group(self, name, artist):
        self._title_row(f"Group: {name or '?'}")
        current = _hex(artist.get_facecolor()[0] if hasattr(artist.get_facecolor(), "ndim")
                       and artist.get_facecolor().ndim == 2 else artist.get_facecolor()) \
            if artist is not None else "#808080"
        self._color_row("Colour of this group:", current,
                        lambda c: self.set_in("colors", name, c),
                        lambda: self.set_in("colors", name, None))
        original = self._original_name(name)
        self._line_row("Name shown:", name, lambda v: self.set_in("rename", original, v if v != original else None))
        self._element_color_row("edges", "Outlines (every group):", allow_none=True)
        if self._kind == "violin":
            for key, items, label in (("violin_side", ["both", "left", "right"], "Violins:"),
                                      ("violin_inner", ["lines", "box", "none"], "Inside:")):
                combo = QComboBox(self)
                combo.addItems(items)
                combo.setCurrentText(self._kwargs.get(key) or items[0])
                combo.activated.connect(lambda _i, k=key, c=combo, d=items[0]:
                                        self.set(k, None if c.currentText() == d else c.currentText()))
                self.inspector_form.addRow(label, combo)

    def _element_color_row(self, element, label, allow_none=False):
        current = (self._kwargs.get("element_colors") or {}).get(element) or "#333333"
        self._color_row(label, current if current != "none" else "#ffffff",
                        lambda c: self.set_in("element_colors", element, c),
                        lambda: self.set_in("element_colors", element, None))
        if allow_none:
            none = QCheckBox("None", self)
            none.setChecked(current == "none")
            none.clicked.connect(lambda on: self.set_in("element_colors", element, "none" if on else None))
            self.inspector_form.addRow("", none)

    def _inspect_element(self, element, artist):
        from staplot import api
        self._title_row(f"{element.replace('_', ' ').capitalize()}: {api.PLOT_ELEMENTS.get(element, '')}")
        self._element_color_row(element, "Colour:", allow_none=element in ("edges", "points"))
        if element == "points":
            combo = QComboBox(self)
            combo.addItems(["jitter", "swarm", "line"])
            combo.setCurrentText(self._kwargs.get("point_style") or "jitter")
            combo.activated.connect(lambda _i: self.set("point_style", None if combo.currentText() == "jitter"
                                                        else combo.currentText()))
            self.inspector_form.addRow("Points drawn as:", combo)
        if element == "brackets":
            combo = QComboBox(self)
            combo.addItems(["stars", "p"])
            combo.setCurrentText(self._kwargs.get("p_format") or "stars")
            combo.activated.connect(lambda _i: self.set("p_format", None if combo.currentText() == "stars"
                                                        else combo.currentText()))
            self.inspector_form.addRow("Show:", combo)
            only = QCheckBox("Only the significant comparisons", self)
            only.setChecked(self._kwargs.get("only_significant", True) is not False)
            only.clicked.connect(lambda on: self.set("only_significant", None if on else False))
            self.inspector_form.addRow("", only)

    def _inspect_axes(self, _key, ax):
        self._title_row("The plot area")
        lim = list(self._kwargs.get("ylim") or [None, None])

        def num(v):
            try:
                return float(v.replace(",", "."))
            except ValueError:
                return None
        low = self._line_row("Y axis from:", "" if lim[0] is None else str(lim[0]),
                             lambda v: self._set_ylim(num(v), 0), "automatic")
        high = self._line_row("Y axis to:", "" if lim[1] is None else str(lim[1]),
                              lambda v: self._set_ylim(num(v), 1), "automatic")
        del low, high
        log = QCheckBox("Logarithmic y axis", self)
        log.setChecked(bool(self._kwargs.get("log_y")))
        log.clicked.connect(lambda on: self.set("log_y", True if on else None))
        self.inspector_form.addRow("", log)
        self._element_color_row("background", "Background:")
        self._element_color_row("axes", "Axes, ticks, labels:")

    def _set_ylim(self, value, i):
        lim = list(self._kwargs.get("ylim") or [None, None])
        lim[i] = value
        self.set("ylim", None if lim == [None, None] else lim)

    def _inspect_figure(self, _key, _artist):
        self._title_row("The figure")
        scale = QDoubleSpinBox(self)
        scale.setRange(0, 4)
        scale.setSingleStep(0.1)
        scale.setSpecialValueText("as drawn")
        scale.setValue(float(self._kwargs.get("font_scale") or 0))
        scale.editingFinished.connect(lambda: self.set("font_scale", scale.value() or None))
        self.inspector_form.addRow("Every text times:", scale)
        aspect = QDoubleSpinBox(self)
        aspect.setRange(0, 6)
        aspect.setSingleStep(0.1)
        aspect.setSpecialValueText("as drawn")
        aspect.setValue(float(self._kwargs.get("aspect") or 0))
        aspect.editingFinished.connect(lambda: self.set("aspect", aspect.value() or None))
        self.inspector_form.addRow("Width / height:", aspect)

    def done(self, r):
        if self._data_dirty and r == QDialog.DialogCode.Accepted and self._editable:
            ans = QMessageBox.question(self, "Edited data",
                                       "Save the edited data to its file? (No: the plot keeps reading the file as it is)",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                                       | QMessageBox.StandardButton.Cancel)
            if ans == QMessageBox.StandardButton.Cancel:
                return
            if ans == QMessageBox.StandardButton.Yes:
                self.save_data(confirm=False)
        if self._temp_csv and os.path.exists(self._temp_csv):
            try:
                os.remove(self._temp_csv)
            except OSError:
                pass
        super().done(r)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m staplot.ui.plot_editor table.csv [kind]")
        return 2
    os.environ.setdefault("QT_API", "pyqt6")
    app = QApplication.instance() or QApplication(sys.argv)
    ed = PlotEditor(argv[0], argv[1] if len(argv) > 1 else "bar")
    if ed.exec():
        print("kind:", ed.kind())
        print("arguments:", ed.kwargs())
        print("text parts:", ed.text_parts())
    del app
    return 0


if __name__ == "__main__":
    sys.exit(main())
