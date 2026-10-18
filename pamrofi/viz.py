"""Visualisation parameters shared by the app and notebooks."""
from __future__ import annotations

from .agro import ADVISORY_COLORS, ADVISORY_LABELS, STAGE_NAMES, STRESS_COLORS, STRESS_LABELS
from .config import Season

STAGE_COLORS = {0: "#d1d5db", 1: "#a3e635", 2: "#15803d", 3: "#b45309"}


def crop_vis(season: Season) -> dict:
    return {"min": 0, "max": len(season.classes) - 1, "palette": season.palette}


def crop_legend(season: Season) -> dict[str, str]:
    return {c.name: c.color for c in season.classes}


STRESS_VIS = {"min": 0, "max": 2, "palette": [STRESS_COLORS[i] for i in range(3)]}
STRESS_LEGEND = {STRESS_LABELS[i]: STRESS_COLORS[i] for i in range(3)}

ADVISORY_VIS = {"min": 0, "max": 2, "palette": [ADVISORY_COLORS[i] for i in range(3)]}
ADVISORY_LEGEND = {ADVISORY_LABELS[i]: ADVISORY_COLORS[i] for i in range(3)}

STAGE_VIS = {"min": 0, "max": 3, "palette": [STAGE_COLORS[i] for i in range(4)]}
STAGE_LEGEND = {STAGE_NAMES[i]: STAGE_COLORS[i] for i in range(4)}

DEFICIT_VIS = {"min": 0, "max": 50, "palette": ["#f0f9ff", "#7dd3fc", "#0284c7", "#0c4a6e"]}
VCI_VIS = {"min": 0, "max": 100, "palette": ["#b91c1c", "#facc15", "#15803d"]}
SMI_VIS = {"min": 0, "max": 1, "palette": ["#92400e", "#fde68a", "#1d4ed8"]}
SOWING_VIS = {"min": 150, "max": 240, "palette": ["#fef3c7", "#f59e0b", "#7c2d12"]}
TRUE_COLOR_VIS = {"bands": ["B4", "B3", "B2"], "min": 0.02, "max": 0.25}
