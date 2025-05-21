# components/__init__.py
"""
This file makes the 'components' directory a Python package.
It allows modules within this directory to be imported using dot notation
(e.g., from components.kpi_display import KPIClusterDisplay).
"""
# Selectively expose components at the package level for easier imports.
from .kpi_display import KPIClusterDisplay
from .data_table_display import DataTableDisplay
from .notes_viewer import NotesViewerComponent
from .calendar_view import PnLCalendarComponent
from .sidebar_manager import SidebarManager
from .column_mapper_ui import ColumnMapperUI
from .scroll_buttons import ScrollButtons # <<< ADDED NEW COMPONENT

__all__ = [
    "KPIClusterDisplay",
    "DataTableDisplay",
    "NotesViewerComponent",
    "PnLCalendarComponent",
    "SidebarManager",
    "ColumnMapperUI",
    "ScrollButtons" # <<< ADDED NEW COMPONENT TO __all__
]
