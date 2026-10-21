from qtpy.QtWidgets import QTreeWidget, QTreeWidgetItem
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QFont


class NavigatorPanel(QTreeWidget):
    """Left-side tree navigator mirroring Prism's project navigator."""

    item_activated = Signal(str, str, str)  # (section, name, kind)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setMinimumWidth(170)
        self.setMaximumWidth(280)
        self.setAnimated(True)
        self.setIndentation(14)

        bold = QFont()
        bold.setBold(True)

        self._data_root = QTreeWidgetItem(self, ["Data Tables"])
        self._data_root.setFont(0, bold)
        self._data_root.setExpanded(True)

        self._analyses_root = QTreeWidgetItem(self, ["Analyses"])
        self._analyses_root.setFont(0, bold)
        self._analyses_root.setExpanded(True)

        self._graphs_root = QTreeWidgetItem(self, ["Graphs"])
        self._graphs_root.setFont(0, bold)
        self._graphs_root.setExpanded(True)

        self.itemActivated.connect(self._on_activated)
        self.itemClicked.connect(self._on_activated)

        self.setStyleSheet("""
            QTreeWidget { border: none; background: #f5f5f5; }
            QTreeWidget::item { padding: 3px 2px; }
            QTreeWidget::item:selected { background: #2d7dd2; color: white; border-radius: 3px; }
        """)

    # ------------------------------------------------------------------
    def _on_activated(self, item: QTreeWidgetItem, _col: int = 0):
        kind = item.data(0, Qt.ItemDataRole.UserRole)
        if kind:
            section, name = kind
            self.item_activated.emit(section, name, kind[0])

    def _add_child(self, root: QTreeWidgetItem, section: str, name: str) -> QTreeWidgetItem:
        child = QTreeWidgetItem(root, [name])
        child.setData(0, Qt.ItemDataRole.UserRole, (section, name))
        return child

    # ------------------------------------------------------------------
    def add_table(self, name: str):
        self._add_child(self._data_root, "table", name)
        self._data_root.setExpanded(True)

    def add_analysis(self, name: str):
        self._add_child(self._analyses_root, "analysis", name)
        self._analyses_root.setExpanded(True)

    def add_graph(self, name: str):
        self._add_child(self._graphs_root, "graph", name)
        self._graphs_root.setExpanded(True)

    def rename_last_table(self, old: str, new: str):
        for i in range(self._data_root.childCount()):
            child = self._data_root.child(i)
            if child.text(0) == old:
                child.setText(0, new)
                child.setData(0, Qt.ItemDataRole.UserRole, ("table", new))
                break

    def clear_all(self):
        self._data_root.takeChildren()
        self._analyses_root.takeChildren()
        self._graphs_root.takeChildren()

    def populate(self, project):
        self.clear_all()
        for t in project.tables:
            self.add_table(t.name)
        for a in project.analyses:
            self.add_analysis(a.name)
        for g in project.graphs:
            self.add_graph(g.name)
