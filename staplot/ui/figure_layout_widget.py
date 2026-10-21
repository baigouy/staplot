"""
Multi-panel figure layout dialog.
Lets the user pick N graphs from the project and arrange them in a grid,
then export as a single publication-quality figure.
"""

import math

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox,
    QSpinBox, QListWidget, QListWidgetItem, QPushButton, QLabel,
    QFileDialog, QAbstractItemView, QWidget, QSplitter, QMessageBox,
)
from qtpy.QtCore import Qt

from ..models.project import GraphConfig, Project
from .. import plots as plot_lib
from ..plots import styles


class FigureLayoutDialog(QDialog):
    """Compose a multi-panel figure from the project's existing graphs."""

    def __init__(self, project: Project, render_fn, parent=None):
        """render_fn(gc) -> matplotlib Figure  (same logic as MainWindow._render_graph)."""
        super().__init__(parent)
        self.setWindowTitle("Multi-Panel Figure Layout")
        self.resize(1000, 680)
        self._project = project
        self._render_fn = render_fn

        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # --- Left panel: graph picker + layout controls ---
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 6, 6)

        left_layout.addWidget(QLabel("Available graphs (select to add):"))
        self._graph_list = QListWidget()
        self._graph_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        for gc in project.graphs:
            item = QListWidgetItem(gc.name)
            item.setData(Qt.ItemDataRole.UserRole, gc)
            self._graph_list.addItem(item)
        left_layout.addWidget(self._graph_list)

        form = QFormLayout()
        self._cols_spin = QSpinBox(); self._cols_spin.setRange(1, 6); self._cols_spin.setValue(2)
        form.addRow("Columns:", self._cols_spin)
        self._figw_spin = QSpinBox(); self._figw_spin.setRange(4, 24); self._figw_spin.setValue(10)
        form.addRow("Figure width (in):", self._figw_spin)
        self._figh_spin = QSpinBox(); self._figh_spin.setRange(3, 20); self._figh_spin.setValue(8)
        form.addRow("Figure height (in):", self._figh_spin)
        left_layout.addLayout(form)

        btn_row = QHBoxLayout()
        preview_btn = QPushButton("Preview")
        preview_btn.clicked.connect(self._preview)
        export_btn = QPushButton("Export…")
        export_btn.clicked.connect(self._export)
        btn_row.addWidget(preview_btn)
        btn_row.addWidget(export_btn)
        left_layout.addLayout(btn_row)

        splitter.addWidget(left)

        # --- Right panel: preview canvas ---
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self._preview_fig = Figure(figsize=(8, 6))
        self._canvas = FigureCanvasQTAgg(self._preview_fig)
        right_layout.addWidget(self._canvas)
        splitter.addWidget(right)
        splitter.setSizes([260, 740])

    # ------------------------------------------------------------------
    def _selected_graphs(self) -> list[GraphConfig]:
        return [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self._graph_list.selectedItems()
        ]

    @styles.styled
    def _build_figure(self, figsize: tuple | None = None) -> Figure | None:
        graphs = self._selected_graphs()
        if not graphs:
            QMessageBox.information(self, "No graphs", "Select at least one graph from the list.")
            return None

        n = len(graphs)
        ncols = min(self._cols_spin.value(), n)
        nrows = math.ceil(n / ncols)
        fw = self._figw_spin.value() if figsize is None else figsize[0]
        fh = self._figh_spin.value() if figsize is None else figsize[1]

        fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh), squeeze=False)
        plt.close(fig)  # we'll redraw manually

        fig = Figure(figsize=(fw, fh))
        for idx, gc in enumerate(graphs):
            row, col = divmod(idx, ncols)
            ax_main = fig.add_subplot(nrows, ncols, idx + 1)
            try:
                src_fig = self._render_fn(gc)
                if src_fig is not None and src_fig.get_axes():
                    src_ax = src_fig.get_axes()[0]
                    # Copy lines, collections, patches into ax_main
                    import matplotlib
                    for line in src_ax.get_lines():
                        ax_main.add_line(
                            matplotlib.lines.Line2D(
                                line.get_xdata(), line.get_ydata(),
                                color=line.get_color(), linewidth=line.get_linewidth(),
                                linestyle=line.get_linestyle(), alpha=line.get_alpha() or 1,
                            )
                        )
                    # Simpler: use the title and axis labels
                    ax_main.set_title(src_ax.get_title(), fontsize=9)
                    ax_main.set_xlabel(src_ax.get_xlabel(), fontsize=8)
                    ax_main.set_ylabel(src_ax.get_ylabel(), fontsize=8)
                    ax_main.autoscale()
                    plt.close(src_fig)
            except Exception as e:
                ax_main.text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center",
                             transform=ax_main.transAxes, fontsize=7)

        # Hide unused axes
        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            fig.add_subplot(nrows, ncols, idx + 1).set_visible(False)

        fig.tight_layout(pad=1.5)
        return fig

    @styles.styled
    def _preview(self):
        """Render a live preview by drawing each sub-figure into subplots."""
        graphs = self._selected_graphs()
        if not graphs:
            QMessageBox.information(self, "No graphs", "Select at least one graph.")
            return

        n = len(graphs)
        ncols = min(self._cols_spin.value(), n)
        nrows = math.ceil(n / ncols)
        fw = self._figw_spin.value()
        fh = self._figh_spin.value()

        self._preview_fig.clear()
        self._preview_fig.set_size_inches(fw, fh)

        for idx, gc in enumerate(graphs):
            ax = self._preview_fig.add_subplot(nrows, ncols, idx + 1)
            try:
                src_fig = self._render_fn(gc)
                if src_fig and src_fig.get_axes():
                    src_ax = src_fig.get_axes()[0]
                    # Efficient preview: copy rendered image via buffer
                    import io
                    buf = io.BytesIO()
                    src_fig.savefig(buf, format="png", dpi=72, bbox_inches="tight")
                    buf.seek(0)
                    import matplotlib.image as mpimg
                    img = mpimg.imread(buf)
                    ax.imshow(img, aspect="auto")
                    ax.set_title(gc.name, fontsize=8)
                    ax.axis("off")
                    plt.close(src_fig)
            except Exception as exc:
                ax.text(0.5, 0.5, str(exc), ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, wrap=True)
                ax.axis("off")

        self._preview_fig.tight_layout(pad=0.5)
        self._canvas.figure = self._preview_fig
        self._preview_fig.set_canvas(self._canvas)
        self._canvas.draw()

    @styles.styled
    def _export(self):
        graphs = self._selected_graphs()
        if not graphs:
            QMessageBox.information(self, "No graphs", "Select at least one graph.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Figure", "",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;TIFF (*.tiff)",
        )
        if not path:
            return

        n = len(graphs)
        ncols = min(self._cols_spin.value(), n)
        nrows = math.ceil(n / ncols)
        fw, fh = self._figw_spin.value(), self._figh_spin.value()

        fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh), squeeze=False)
        for idx, gc in enumerate(graphs):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            try:
                src_fig = self._render_fn(gc)
                if src_fig and src_fig.get_axes():
                    import io
                    buf = io.BytesIO()
                    src_fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                    buf.seek(0)
                    import matplotlib.image as mpimg
                    img = mpimg.imread(buf)
                    ax.imshow(img, aspect="auto")
                    ax.set_title(gc.name, fontsize=9)
                    ax.axis("off")
                    plt.close(src_fig)
            except Exception as exc:
                ax.text(0.5, 0.5, str(exc), ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
                ax.axis("off")

        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        fig.tight_layout(pad=1.0)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        QMessageBox.information(self, "Exported", f"Figure saved to:\n{path}")
