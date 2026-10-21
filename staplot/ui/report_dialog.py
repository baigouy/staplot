"""HTML / PDF report generator."""

import base64
import datetime
import io
import os
import webbrowser
import tempfile

from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextBrowser, QFileDialog, QLabel, QProgressBar,
)
from qtpy.QtCore import Qt

from ..plots import styles


@styles.styled
def _fig_to_b64(fig) -> str:
    """Render a matplotlib Figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def generate_report_html(project, render_fn) -> str:
    """
    Build a self-contained HTML report from a Project.
    render_fn(gc) must return a matplotlib Figure or None.
    """
    date_str = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M")

    # ── Graphs ────────────────────────────────────────────────────────
    graphs_html = []
    for gc in project.graphs:
        try:
            fig = render_fn(gc)
        except Exception:
            fig = None
        if fig is None:
            continue
        b64 = _fig_to_b64(fig)
        graphs_html.append(
            f'<div class="graph">'
            f'<h3>{_esc(gc.name)}</h3>'
            f'<img src="data:image/png;base64,{b64}" alt="{_esc(gc.name)}"/>'
            f'</div>'
        )

    # ── Analyses ──────────────────────────────────────────────────────
    analyses_html = []
    for a in project.analyses:
        analyses_html.append(
            f'<div class="analysis">'
            f'<h3>{_esc(a.name)}</h3>'
            f'<p class="method">Method: {_esc(a.method)}'
            + (f'  |  Table: {_esc(a.source_table)}' if a.source_table else '')
            + f'</p>'
            f'<pre>{_esc(a.summary_text)}</pre>'
            f'</div>'
        )

    # ── Data tables ───────────────────────────────────────────────────
    tables_html = []
    for t in project.tables:
        if hasattr(t, "to_dataframe"):
            df = t.to_dataframe()
            if df.empty:
                continue
            # Limit display to first 200 rows
            df_disp = df.head(200)
            table_html = df_disp.to_html(index=True, classes="data-table",
                                          border=0, na_rep="")
            note = (f'<p class="note">Showing first 200 of {len(df)} rows.</p>'
                    if len(df) > 200 else "")
            tables_html.append(
                f'<div class="table-section">'
                f'<h3>{_esc(t.name)}</h3>'
                f'{note}{table_html}'
                f'</div>'
            )
        elif hasattr(t, "get_matrix"):
            import numpy as np
            mat = t.get_matrix()
            rows = ""
            for i, rl in enumerate(t.row_labels):
                cells = "".join(f"<td>{mat[i,j]}</td>"
                                for j in range(len(t.col_labels)))
                rows += f"<tr><th>{_esc(rl)}</th>{cells}</tr>"
            col_heads = "".join(f"<th>{_esc(cl)}</th>"
                                for cl in t.col_labels)
            tables_html.append(
                f'<div class="table-section">'
                f'<h3>{_esc(t.name)}  <span class="badge">Contingency</span></h3>'
                f'<table class="data-table"><thead>'
                f'<tr><th></th>{col_heads}</tr></thead>'
                f'<tbody>{rows}</tbody></table>'
                f'</div>'
            )

    # ── Assemble ──────────────────────────────────────────────────────
    def section(heading: str, items: list, empty_msg: str) -> str:
        body = "\n".join(items) if items else f"<p>{empty_msg}</p>"
        return f"<h2>{heading}  <span class='count'>({len(items)})</span></h2>\n{body}\n"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_esc(project.name)} — Staplot Report</title>
<style>
  body {{
    font-family: Arial, Helvetica, sans-serif;
    max-width: 1080px; margin: 0 auto; padding: 24px 32px;
    color: #222; background: #fff;
  }}
  h1  {{ color: #2d5fa6; border-bottom: 3px solid #2d5fa6; padding-bottom: 6px; }}
  h2  {{ color: #333; border-bottom: 1px solid #ddd; margin-top: 32px; }}
  h3  {{ color: #444; margin-bottom: 4px; }}
  .count   {{ font-size: 0.75em; color: #888; font-weight: normal; }}
  .badge   {{ font-size: 0.7em; background: #e8f0ff; color: #2d5fa6;
              padding: 2px 6px; border-radius: 10px; }}
  .method  {{ color: #666; font-size: 0.85em; margin: 2px 0 8px; }}
  .note    {{ color: #888; font-size: 0.8em; font-style: italic; }}
  .graph   {{ margin: 18px 0; page-break-inside: avoid; }}
  .graph img {{ max-width: 100%; border: 1px solid #e8e8e8;
                border-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  .analysis {{
    background: #f7f9fc; border-left: 4px solid #2d5fa6;
    padding: 12px 16px; margin: 12px 0; border-radius: 0 4px 4px 0;
  }}
  pre {{
    font-family: "Courier New", Courier, monospace;
    font-size: 11px; white-space: pre-wrap; margin: 0;
  }}
  table.data-table {{ border-collapse: collapse; width: 100%;
                      font-size: 12px; margin-top: 8px; }}
  table.data-table th {{
    background: #2d5fa6; color: #fff;
    padding: 5px 8px; text-align: left;
  }}
  table.data-table td {{ border: 1px solid #e0e0e0; padding: 4px 8px; }}
  table.data-table tr:nth-child(even) td {{ background: #f4f4f4; }}
  footer {{ color: #999; font-size: 11px; margin-top: 40px;
            border-top: 1px solid #eee; padding-top: 10px; }}
  @media print {{
    body {{ padding: 0; }}
    .no-print {{ display: none; }}
  }}
</style>
</head>
<body>

<h1>{_esc(project.name)}</h1>
<p>Generated by <strong>Staplot</strong> on {date_str}</p>

{section("Graphs",    graphs_html,   "No graphs in this project.")}
{section("Analyses",  analyses_html, "No analyses in this project.")}
{section("Data Tables", tables_html, "No data tables.")}

<footer>Staplot — Statistical analysis and publication-quality plotting</footer>
</body>
</html>"""


def _esc(text: str) -> str:
    """Minimal HTML escaping."""
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


# ── Dialog ────────────────────────────────────────────────────────────

class ReportDialog(QDialog):
    """Preview and export the project report as HTML (printable to PDF)."""

    def __init__(self, project, render_fn, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate Report")
        self.setMinimumSize(820, 620)

        self._html = ""
        layout = QVBoxLayout(self)

        # Status
        self._status = QLabel("Rendering graphs…")
        layout.addWidget(self._status)

        self._progress = QProgressBar()
        self._progress.setRange(0, max(1, len(project.graphs)))
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        # Preview
        self._browser = QTextBrowser()
        self._browser.setOpenExternalLinks(True)
        layout.addWidget(self._browser)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        save_btn = QPushButton("Save as HTML…")
        save_btn.clicked.connect(self._save_html)
        btn_row.addWidget(save_btn)

        open_btn = QPushButton("Open in Browser  (→ print to PDF)")
        open_btn.clicked.connect(self._open_browser)
        btn_row.addWidget(open_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        # Generate (inline, quick enough for typical projects)
        self._generate(project, render_fn)

    def _generate(self, project, render_fn):
        n = len(project.graphs)

        def tracked_render(gc):
            fig = render_fn(gc)
            self._progress.setValue(self._progress.value() + 1)
            from qtpy.QtWidgets import QApplication
            QApplication.processEvents()
            return fig

        self._html = generate_report_html(project, tracked_render)
        self._progress.setValue(n)
        self._status.setText(
            f"Report ready — {len(project.graphs)} graphs, "
            f"{len(project.analyses)} analyses, "
            f"{len(project.tables)} tables."
        )
        self._browser.setHtml(self._html)

    def _save_html(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", "", "HTML (*.html);;All files (*)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._html)
            self._status.setText(f"Saved: {path}")

    def _open_browser(self):
        tmp = tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w", encoding="utf-8")
        tmp.write(self._html)
        tmp.close()
        webbrowser.open(f"file://{tmp.name}")
