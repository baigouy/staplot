import io

import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QInputDialog, QApplication,
)
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QImage

from ..plots import styles


class StyledCanvas(FigureCanvasQTAgg):
    """Qt canvas that (re)draws inside the staplot style: some settings (e.g.
    the tick label size the number of ticks is chosen from) are read when the
    figure is drawn, not when it is built."""

    def draw(self):
        with styles.context():
            super().draw()


def apply_axis_settings(fig: Figure, opts: dict):
    """Apply axis options dict to every Axes in a Figure."""
    if not opts:
        return
    for ax in fig.get_axes():
        x_scale = opts.get("x_scale", "linear")
        y_scale = opts.get("y_scale", "linear")
        if x_scale == "log":
            ax.set_xscale("log")
        if y_scale == "log":
            ax.set_yscale("log")

        x_min, x_max = opts.get("x_min"), opts.get("x_max")
        if x_min is not None or x_max is not None:
            lo, hi = ax.get_xlim()
            ax.set_xlim(
                x_min if x_min is not None else lo,
                x_max if x_max is not None else hi,
            )

        y_min, y_max = opts.get("y_min"), opts.get("y_max")
        y_zero = opts.get("y_zero", False)
        lo_y, hi_y = ax.get_ylim()
        new_lo = 0 if y_zero else (y_min if y_min is not None else lo_y)
        new_hi = y_max if y_max is not None else hi_y
        ax.set_ylim(new_lo, new_hi)

        if opts.get("x_label"):
            ax.set_xlabel(opts["x_label"])
        if opts.get("y_label"):
            ax.set_ylabel(opts["y_label"])

    fig.tight_layout()


class PlotCanvas(QWidget):
    """Embeds a matplotlib figure with toolbar, export, clipboard, annotation, and point exclusion."""

    axes_changed = Signal(dict)         # emitted when user saves new axis options
    point_excluded = Signal(str, int, bool)  # (column_name, row_index, exclude_all)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._fig = Figure(figsize=(6, 4))
        self._canvas = StyledCanvas(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)

        # ── Button bar ────────────────────────────────────────────────
        btn_bar = QHBoxLayout()
        btn_bar.addWidget(self._toolbar)
        btn_bar.addStretch()

        axes_btn = QPushButton("Axes…")
        axes_btn.setFixedWidth(64)
        axes_btn.setToolTip("Edit axis scales and ranges")
        axes_btn.clicked.connect(self._edit_axes)
        btn_bar.addWidget(axes_btn)

        self._annotate_btn = QPushButton("Annotate")
        self._annotate_btn.setFixedWidth(76)
        self._annotate_btn.setCheckable(True)
        self._annotate_btn.setToolTip(
            "Toggle annotation mode: click on the plot to add a text label"
        )
        self._annotate_btn.toggled.connect(self._toggle_annotate)
        btn_bar.addWidget(self._annotate_btn)

        self._exclude_btn = QPushButton("Exclude")
        self._exclude_btn.setFixedWidth(70)
        self._exclude_btn.setCheckable(True)
        self._exclude_btn.setToolTip(
            "Toggle point-exclusion mode: click a data point to exclude/include it"
        )
        self._exclude_btn.toggled.connect(self._toggle_exclude_mode)
        btn_bar.addWidget(self._exclude_btn)

        copy_btn = QPushButton("Copy")
        copy_btn.setFixedWidth(56)
        copy_btn.setToolTip("Copy graph to clipboard as PNG")
        copy_btn.clicked.connect(self.copy_to_clipboard)
        btn_bar.addWidget(copy_btn)

        export_btn = QPushButton("Export…")
        export_btn.setFixedWidth(72)
        export_btn.clicked.connect(self.export_dialog)
        btn_bar.addWidget(export_btn)

        layout.addLayout(btn_bar)
        layout.addWidget(self._canvas)

        self._current_axis_opts: dict = {}
        self._annotate_mode = False
        self._exclude_mode = False
        self._cid_click = self._canvas.mpl_connect("button_press_event", self._on_mpl_click)
        self._cid_pick  = self._canvas.mpl_connect("pick_event",         self._on_pick_event)
        self._show_placeholder()

    # ------------------------------------------------------------------
    def _show_placeholder(self):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.text(0.5, 0.5, "No graph selected", ha="center", va="center",
                transform=ax.transAxes, color="#aaaaaa", fontsize=12)
        ax.axis("off")
        self._canvas.draw()

    @styles.styled
    def show_figure(self, fig: Figure, axis_opts: dict | None = None):
        # Disconnect old handlers from previous figure, reconnect to new
        try:
            self._canvas.mpl_disconnect(self._cid_click)
            self._canvas.mpl_disconnect(self._cid_pick)
        except Exception:
            pass
        self._fig = fig
        if axis_opts:
            self._current_axis_opts = axis_opts
            apply_axis_settings(fig, axis_opts)
        self._canvas.figure = fig
        fig.set_canvas(self._canvas)
        self._cid_click = self._canvas.mpl_connect("button_press_event", self._on_mpl_click)
        self._cid_pick  = self._canvas.mpl_connect("pick_event",         self._on_pick_event)
        self._canvas.draw()

    # ------------------------------------------------------------------
    @styles.styled
    def _edit_axes(self):
        from .axis_dialog import AxisDialog
        dlg = AxisDialog(self._current_axis_opts, parent=self)
        if dlg.exec():
            self._current_axis_opts = dlg.get_options()
            apply_axis_settings(self._fig, self._current_axis_opts)
            self._canvas.draw()
            self.axes_changed.emit(self._current_axis_opts)

    # ------------------------------------------------------------------
    # Annotation mode
    # ------------------------------------------------------------------
    def _toggle_annotate(self, checked: bool):
        self._annotate_mode = checked
        if checked:
            self._exclude_btn.setChecked(False)
        cursor = Qt.CursorShape.CrossCursor if checked else Qt.CursorShape.ArrowCursor
        self._canvas.setCursor(cursor)

    # ------------------------------------------------------------------
    # Point exclusion mode
    # ------------------------------------------------------------------
    def _toggle_exclude_mode(self, checked: bool):
        self._exclude_mode = checked
        if checked:
            self._annotate_btn.setChecked(False)
            self._exclude_btn.setStyleSheet("background-color: #ff9999;")
        else:
            self._exclude_btn.setStyleSheet("")
        cursor = Qt.CursorShape.PointingHandCursor if checked else Qt.CursorShape.ArrowCursor
        self._canvas.setCursor(cursor)

    def _on_pick_event(self, event):
        """Handle matplotlib pick_event: emit point_excluded for the clicked artist."""
        if not self._exclude_mode:
            return
        artist = event.artist
        meta = getattr(artist, "_staplot_meta", None)
        if meta is None:
            return
        col_name  = meta["column"]
        row_list  = meta["rows"]
        excl_all  = meta.get("exclude_all", False)
        for ind in event.ind:
            if ind < len(row_list):
                self.point_excluded.emit(col_name, int(row_list[ind]), excl_all)

    @styles.styled
    def _on_mpl_click(self, event):
        if not self._annotate_mode:
            return
        if event.inaxes is None or event.xdata is None:
            return
        # Only left-clicks
        if event.button != 1:
            return
        text, ok = QInputDialog.getText(self, "Add Annotation", "Annotation text:")
        if not ok or not text.strip():
            return
        ax = event.inaxes
        ax.text(
            event.xdata, event.ydata, text.strip(),
            fontsize=9, ha="center", va="bottom", color="#222222",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#ffffcc", edgecolor="#888888", alpha=0.85),
            zorder=10,
        )
        self._fig.tight_layout()
        self._canvas.draw()

    # ------------------------------------------------------------------
    # Clipboard / export
    # ------------------------------------------------------------------
    @styles.styled
    def copy_to_clipboard(self):
        """Render current figure to PNG and place on system clipboard."""
        buf = io.BytesIO()
        try:
            self._fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        except Exception:
            return
        buf.seek(0)
        image = QImage()
        image.loadFromData(buf.read())
        QApplication.clipboard().setImage(image)

    @styles.styled
    def export_dialog(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Graph", "",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;TIFF (*.tiff)",
        )
        if path:
            self._fig.savefig(path, dpi=300, bbox_inches="tight")
