"""Pure-Python agronomy: FAO-56 Kc curves, PAM stage tokens, Hargreaves ET0,
water balance and advisory messages.

These functions are the single source of truth for the formulas. The Earth
Engine versions in ``pamrofi.ee_agro`` implement exactly the same equations on
images, and the unit tests check both against FAO-56 reference values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import CropClass

# PAM phenology tokens (kept identical to the original project semantics)
STAGE_OUT_OF_SEASON = 0   # before sowing / after harvest
STAGE_INITIAL = 1         # sowing, emergence, transplant establishment
STAGE_CRITICAL = 2        # development + mid-season (most water-sensitive)
STAGE_LATE = 3            # maturity / senescence (natural drawdown)

STAGE_NAMES = {
    STAGE_OUT_OF_SEASON: "Out of season",
    STAGE_INITIAL: "Initial",
    STAGE_CRITICAL: "Development / mid-season",
    STAGE_LATE: "Late season",
}


def stage_boundaries(c: CropClass) -> tuple[int, int, int, int]:
    """Cumulative day boundaries (end of ini, end of dev, end of mid, end of late)."""
    s = c.stages
    b1 = s["ini"]
    b2 = b1 + s["dev"]
    b3 = b2 + s["mid"]
    b4 = b3 + s["late"]
    return b1, b2, b3, b4


def kc_at(das: float, c: CropClass) -> float:
    """FAO-56 piecewise-linear crop coefficient for a given Days-After-Sowing."""
    if not c.irrigable or das < 0:
        return 0.0
    b1, b2, b3, b4 = stage_boundaries(c)
    ini, mid, end = c.kc["ini"], c.kc["mid"], c.kc["end"]
    if das <= b1:
        return ini
    if das <= b2:
        return ini + (das - b1) * (mid - ini) / (b2 - b1)
    if das <= b3:
        return mid
    if das <= b4:
        return mid + (das - b3) * (end - mid) / (b4 - b3)
    return 0.0


def stage_at(das: float, c: CropClass) -> int:
    """PAM token for a given Days-After-Sowing."""
    if not c.irrigable or das < 0:
        return STAGE_OUT_OF_SEASON
    b1, b2, b3, b4 = stage_boundaries(c)
    if das <= b1:
        return STAGE_INITIAL
    if das <= b3:
        return STAGE_CRITICAL
    if das <= b4:
        return STAGE_LATE
    return STAGE_OUT_OF_SEASON


# --------------------------------------------------------------------- ET0
GSC = 0.0820  # solar constant, MJ m-2 min-1


def extraterrestrial_radiation(lat_deg: float, doy: int) -> float:
    """FAO-56 eq. 21 — daily extraterrestrial radiation Ra in MJ m-2 day-1."""
    phi = math.radians(lat_deg)
    dr = 1 + 0.033 * math.cos(2 * math.pi * doy / 365)
    delta = 0.409 * math.sin(2 * math.pi * doy / 365 - 1.39)
    ws = math.acos(max(-1.0, min(1.0, -math.tan(phi) * math.tan(delta))))
    return (24 * 60 / math.pi) * GSC * dr * (
        ws * math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.sin(ws)
    )


def hargreaves_et0(tmax_c: float, tmin_c: float, lat_deg: float, doy: int) -> float:
    """FAO-56 eq. 52 Hargreaves reference ET0 in mm/day."""
    ra = extraterrestrial_radiation(lat_deg, doy)
    tmean = (tmax_c + tmin_c) / 2
    return 0.0023 * (tmean + 17.8) * math.sqrt(max(tmax_c - tmin_c, 0.0)) * ra * 0.408


# ----------------------------------------------------------- water balance
def water_deficit(et0_mm: float, kc: float, rain_mm: float,
                  effective_rain_factor: float = 0.8,
                  percolation_mm: float = 0.0) -> float:
    """Period deficit (mm) = ETc + percolation − effective rain, floored at 0."""
    etc = et0_mm * kc
    return max(0.0, etc + percolation_mm - effective_rain_factor * rain_mm)


def advisory_level(deficit_mm: float, thresholds=(10, 30)) -> int:
    """0 = no irrigation, 1 = routine in 3–4 days, 2 = immediate."""
    low, high = thresholds
    if deficit_mm > high:
        return 2
    if deficit_mm >= low:
        return 1
    return 0


ADVISORY_LABELS = {
    0: "No irrigation needed",
    1: "Routine irrigation within 3–4 days",
    2: "Critical deficit: irrigate immediately",
}
ADVISORY_COLORS = {0: "#16a34a", 1: "#f59e0b", 2: "#dc2626"}
STRESS_LABELS = {0: "No stress", 1: "Mild depletion (watch)", 2: "Severe deficit"}
STRESS_COLORS = {0: "#16a34a", 1: "#f59e0b", 2: "#dc2626"}


@dataclass
class FieldAdvice:
    crop: str
    stage: int
    deficit_mm: float
    level: int
    period_start: str
    period_end: str

    def message(self, lang: str = "en") -> str:
        depth = round(self.deficit_mm)
        if lang == "hi":
            stage_hi = {0: "मौसम से बाहर", 1: "प्रारंभिक", 2: "वृद्धि / मध्य", 3: "पकाव"}[self.stage]
            action = {
                0: "अभी सिंचाई की आवश्यकता नहीं है।",
                1: f"अगले 3–4 दिनों में लगभग {depth} मिमी सिंचाई करें।",
                2: f"तुरंत लगभग {depth} मिमी सिंचाई करें — पानी की गंभीर कमी।",
            }[self.level]
            return (f"फसल: {self.crop} | अवस्था: {stage_hi} | "
                    f"{self.period_start} से {self.period_end}: {action}")
        action = {
            0: "No irrigation needed right now.",
            1: f"Irrigate about {depth} mm within the next 3–4 days.",
            2: f"Irrigate about {depth} mm immediately — critical water deficit.",
        }[self.level]
        return (f"Crop: {self.crop} | Stage: {STAGE_NAMES[self.stage]} | "
                f"{self.period_start} to {self.period_end}: {action}")


# ------------------------------------------------------- moisture stress
def stress_level(stage: int, vci: float, smi: float, flooded: bool = False,
                 vci_severe: float = 35, vci_watch: float = 60,
                 smi_severe: float = 0.35, smi_watch: float = 0.50) -> int | None:
    """Phase-aware VCI × SMI decision matrix. Returns None when not applicable.

    * Initial stage: the canopy is not established yet, so low NDVI is normal and
      VCI is ignored; only radar soil moisture counts. Flooded paddy is never stressed.
    * Development / mid-season (critical): full matrix. Severe if VCI is very low,
      or if soil is very dry while the crop is at its most sensitive stage.
    * Late season: falling greenness is natural senescence, so only flag stress
      when both the canopy AND the soil are in the severe range.
    """
    if stage == STAGE_OUT_OF_SEASON:
        return None
    if stage == STAGE_INITIAL:
        if flooded:
            return 0
        if smi < smi_severe:
            return 2
        if smi < smi_watch:
            return 1
        return 0
    if stage == STAGE_CRITICAL:
        if vci < vci_severe or smi < smi_severe:
            return 2
        if vci < vci_watch and smi < smi_watch:
            return 1
        return 0
    # late
    return 1 if (vci < vci_severe and smi < smi_severe) else 0
