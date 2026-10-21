import os
import sys

os.environ.setdefault("QT_API", "pyqt6")

from qtpy.QtWidgets import QApplication
from .ui.main_window import MainWindow


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Staplot")
    app.setOrganizationName("Staplot")
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
