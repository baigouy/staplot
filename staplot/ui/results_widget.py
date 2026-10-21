from qtpy.QtWidgets import QTextEdit
from qtpy.QtGui import QFont


class ResultsWidget(QTextEdit):
    """Read-only monospaced display for analysis output."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        font = QFont("Courier New", 9)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)
        self.setStyleSheet(
            "QTextEdit { background: #fafafa; border: none; padding: 8px; }"
        )

    def show_result(self, result):
        self.setPlainText(result.summary_text)
