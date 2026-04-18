"""Shared data models used by both app.py and ai_agent.py.

Kept in a standalone module so neither file has to import the other —
importing app.py from ai_agent.py would re-execute the Streamlit script
and duplicate every widget, causing 'duplicate key' errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class LineGroup:
    x_col: str
    y_cols: List[str] = field(default_factory=list)
    z_cols: List[str] = field(default_factory=list)
    name_prefix: str = ""


@dataclass
class SubplotConfig:
    title: str
    plot_type: Literal["2D", "3D"]
    groups: List[LineGroup]
    chart_type: Literal["line", "scatter", "bar", "histogram"] = "line"
    mode: Literal["lines", "markers", "lines+markers"] = "lines"
    animated: bool = False
    x_label: str = ""
    y_label: str = ""
    z_label: str = ""
    log_x: bool = False
    log_y: bool = False
    animation_axis_mode: Literal["auto", "fixed"] = "fixed"
