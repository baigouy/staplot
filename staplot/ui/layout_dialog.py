"""Figure Layout / Panel builder — arrange multiple graphs into one export-ready figure."""

from io import BytesIO

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.figure

from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QAbstractItemView,
    QPushButton, QLabel, QSpinBox, QFileDialog, QGroupBox, QFormLayout,
    QComboBox, QDoubleSpinBox,
)
from qtpy.QtCore import Qt

from ..plots import styles


# ── helpers ──────────────────────────────────────────────────────────


@styles.styled
def _fig_to_img(fig: matplotlib.figure.Figure, dpi: int = 120) -> "numpy array":
    """Render a Figure to an RGBA numpy array via PNG."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return mpimg.imread(buf)


@styles.styled
def build_panel_figure(
    gcs,
    render_fn,
    nrows: int,
    ncols: int,
    dpi: int = 120,
    fig_width: float | None = None,
    fig_height: float | None = None,
    label_panels: bool = True,
) -> matplotlib.figure.Figure:
    """
    Render each GraphConfig in *gcs* into a nrows×ncols panel.

    render_fn(gc) → matplotlib Figure | None

    Panels are rasterised at *dpi* and composited via imshow — fast and reliable
    across all plot types.  For print-quality output, use dpi >= 200.
    """
    import numpy as np

    n_panels = min(len(gcs), nrows * ncols)
    w = fig_width  or ncols * 4.2
    h = fig_height or nrows * 3.6
    panel_fig = plt.figure(figsize=(w, h))

    for idx in range(n_panels):
        gc = gcs[idx]
        ax = panel_fig.add_subplot(nrows, ncols, idx + 1)
        try:
            sub = render_fn(gc)
            if sub is None:
                raise RuntimeError("render returned None")
            img = _fig_to_img(sub, dpi=dpi)
            ax.imshow(img, aspect="equal", interpolation="lanczos")
        except Exception as exc:
            ax.text(0.5, 0.5, f"[error]\n{exc}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="red")
        ax.axis("off")
        if label_panels:
            letter = chr(ord("A") + idx)
            ax.text(-0.02, 1.02, letter, transform=ax.transAxes,
                    fontsize=14, fontweight="bold", va="bottom")

    panel_fig.tight_layout(pad=0.4)
    return panel_fig


# ── Dialog ───────────────────────────────────────────────────────────


class LayoutDialog(QDialog):
    """GUI for composing a multi-panel publication figure."""

    def __init__(self, project, render_fn, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Figure Layout")
        self.setMinimumSize(520, 480)
        self._project   = project
        self._render_fn = render_fn
        self._panel_fig = None

        layout = QVBoxLayout(self)

        # ── Graph selector ────────────────────────────────────────────
        grp_sel = QGroupBox("Select graphs to include (Ctrl/Shift to multi-select)")
        v = QVBoxLayout(grp_sel)
        self._graph_list = QListWidget()
        self._graph_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        for gc in project.graphs:
            self._graph_list.addItem(gc.name)
        if self._graph_list.count():
            self._graph_list.selectAll()
        v.addWidget(self._graph_list)
        layout.addWidget(grp_sel)

        # ── Layout controls ───────────────────────────────────────────
        grp_ctrl = QGroupBox("Panel layout")
        form = QFormLayout(grp_ctrl)

        self._rows_spin = QSpinBox()
        self._rows_spin.setRange(1, 8)
        self._rows_spin.setValue(1)
        form.addRow("Rows:", self._rows_spin)

        self._cols_spin = QSpinBox()
        self._cols_spin.setRange(1, 8)
        self._cols_spin.setValue(min(len(project.graphs), 3) or 1)
        form.addRow("Columns:", self._cols_spin)

        self._label_chk_cb = QComboBox()
        self._label_chk_cb.addItems(["A, B, C … labels", "No panel labels"])
        form.addRow("Panel labels:", self._label_chk_cb)

        self._dpi_spin = QSpinBox()
        self._dpi_spin.setRange(72, 600)
        self._dpi_spin.setValue(150)
        self._dpi_spin.setSuffix(" dpi")
        form.addRow("Preview DPI:", self._dpi_spin)

        layout.addWidget(grp_ctrl)

        # ── Action buttons ────────────────────────────────────────────
        btn_row = QHBoxLayout()

        preview_btn = QPushButton("Preview Panel")
        preview_btn.setDefault(True)
        preview_btn.clicked.connect(self._preview)
        btn_row.addWidget(preview_btn)

        btn_row.addStretch()

        export_btn = QPushButton("Export…")
        export_btn.clicked.connect(self._export)
        btn_row.addWidget(export_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)

        layout.addLayout(btn_row)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

    # ------------------------------------------------------------------
    def _selected_gcs(self):
        names = {item.text() for item in self._graph_list.selectedItems()}
        return [gc for gc in self._project.graphs if gc.name in names]

    def _preview(self):
        gcs = self._selected_gcs()
        if not gcs:
            self._status.setText("Select at least one graph.")
            return
        nrows = self._rows_spin.value()
        ncols = self._cols_spin.value()
        dpi   = self._dpi_spin.value()
        label = self._label_chk_cb.currentIndex() == 0

        self._status.setText("Rendering…")
        self.repaint()
        try:
            self._panel_fig = build_panel_figure(
                gcs, self._render_fn, nrows, ncols,
                dpi=dpi, label_panels=label,
            )
            # Show in a new matplotlib window
            self._panel_fig.show()
            self._status.setText(
                f"Panel: {nrows} x {ncols}  ({len(gcs)} graph(s)) — rendered OK"
            )
        except Exception as exc:
            self._status.setText(f"Error: {exc}")

    @styles.styled
    def _export(self):
        gcs = self._selected_gcs()
        if not gcs:
            self._status.setText("Select at least one graph.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Panel Figure", "panel_figure",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;TIFF (*.tiff)",
        )
        if not path:
            return

        nrows = self._rows_spin.value()
        ncols = self._cols_spin.value()
        label = self._label_chk_cb.currentIndex() == 0
        export_dpi = 300 if path.lower().endswith((".tiff", ".tif")) else 200

        self._status.setText("Exporting…")
        self.repaint()
        try:
            fig = build_panel_figure(
                gcs, self._render_fn, nrows, ncols,
                dpi=export_dpi, label_panels=label,
            )
            fig.savefig(path, dpi=export_dpi, bbox_inches="tight")
            import os
            self._status.setText(f"Saved: {os.path.basename(path)}")
        except Exception as exc:
            self._status.setText(f"Export error: {exc}")
