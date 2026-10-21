"""Graph gallery — thumbnail grid of all graphs in the project."""

from io import BytesIO

from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QGridLayout, QLabel, QWidget,
    QScrollArea, QPushButton, QFrame,
)
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QPixmap, QFont

from ..plots import styles


@styles.styled
def _fig_to_pixmap(fig, w: int = 220, h: int = 165) -> QPixmap:
    """Render a matplotlib Figure to a QPixmap at thumbnail size."""
    try:
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
        buf.seek(0)
        px = QPixmap()
        px.loadFromData(buf.read())
        return px.scaled(w, h,
                         Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
    except Exception:
        px = QPixmap(w, h)
        px.fill(Qt.GlobalColor.lightGray)
        return px


class _Thumbnail(QFrame):
    """Single graph thumbnail with name label and Open button."""

    clicked = Signal(str)   # graph name

    def __init__(self, name: str, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self._name = name
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFixedSize(240, 230)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        img_lbl = QLabel()
        img_lbl.setPixmap(pixmap)
        img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(img_lbl)

        name_lbl = QLabel(name)
        name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_lbl.setWordWrap(True)
        f = QFont()
        f.setPointSize(8)
        name_lbl.setFont(f)
        layout.addWidget(name_lbl)

        open_btn = QPushButton("Open")
        open_btn.setFixedHeight(24)
        open_btn.clicked.connect(lambda: self.clicked.emit(self._name))
        layout.addWidget(open_btn)


class GraphGallery(QDialog):
    """Thumbnail gallery of every graph in the project."""

    graph_selected = Signal(str)   # emitted with the graph name

    def __init__(self, project, render_fn, parent=None):
        """
        project   — the Project instance
        render_fn — callable(GraphConfig) → matplotlib.figure.Figure | None
        """
        super().__init__(parent)
        self.setWindowTitle("Graph Gallery")
        self.setMinimumSize(740, 520)

        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setSpacing(12)
        grid.setContentsMargins(8, 8, 8, 8)

        cols = 3
        visible = 0
        for gc in project.graphs:
            try:
                fig = render_fn(gc)
            except Exception:
                fig = None
            if fig is None:
                continue
            px = _fig_to_pixmap(fig)
            thumb = _Thumbnail(gc.name, px)
            thumb.clicked.connect(self._on_click)
            grid.addWidget(thumb, visible // cols, visible % cols)
            visible += 1

        if visible == 0:
            grid.addWidget(QLabel("No graphs in this project yet."), 0, 0)

        scroll.setWidget(inner)
        layout.addWidget(scroll)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)

    def _on_click(self, name: str):
        self.graph_selected.emit(name)
        self.accept()
