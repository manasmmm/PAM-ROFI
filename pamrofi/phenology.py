"""Per-pixel phenology alignment (PAM) on Earth Engine.

The original notebook gave every pixel the same calendar (month → token, one
sowing date for the whole region). Here every pixel gets its own sowing date,
derived from its own satellite time series, and from that its own
Days-After-Sowing (DAS), PAM stage token and FAO-56 Kc on any date.

The formulas mirror ``pamrofi.agro`` exactly.
"""
from __future__ import annotations

import ee

from .agro import (STAGE_CRITICAL, STAGE_INITIAL, STAGE_LATE, STAGE_OUT_OF_SEASON,
                   stage_boundaries)
from .config import Season
from .features import SeasonCube


class PhenologyModel:
    def __init__(self, season: Season, cube: SeasonCube, crop_map: ee.Image):
        self.season = season
        self.cube = cube
        self.crop_map = crop_map.rename("class")
        self.metrics = cube.phenology_metrics()
        self.sowing_t = self._sowing_image()          # days since season start
        self.active = self._active_mask()             # irrigable and not fallow

    # ----------------------------------------------------------------- sowing
    def _sowing_for_class(self, c) -> ee.Image:
        if c.sowing_method == "fixed":
            t = ee.Date(c.fixed_sowing).difference(self.cube.start, "day")
            return ee.Image.constant(t).float()
        if c.sowing_method == "vh_min":
            s0, s1 = self.season.sowing_search
            return self.cube.vh_minimum_time(s0, s1).subtract(c.emergence_lag_days).float()
        # ndvi_sos
        return self.cube.start_of_season(c.sos_fraction).subtract(c.emergence_lag_days).float()

    def _sowing_image(self) -> ee.Image:
        out = ee.Image.constant(-9999).float()
        for c in self.season.classes:
            if c.irrigable:
                out = out.where(self.crop_map.eq(c.id), self._sowing_for_class(c))
        return out.rename("sowing_t")

    def _active_mask(self) -> ee.Image:
        active = ee.Image.constant(0)
        amp = self.metrics.select("ndvi_amp")
        for c in self.season.classes:
            if not c.irrigable:
                continue
            is_c = self.crop_map.eq(c.id)
            if c.fallow_amplitude is not None:
                is_c = is_c.And(amp.gte(c.fallow_amplitude))
            active = active.where(is_c, 1)
        return active.rename("active")

    def sowing_date_doy(self) -> ee.Image:
        """Sowing date as day-of-year (for maps/QA)."""
        start_doy = ee.Number.parse(self.cube.start.format("D"))
        return self.sowing_t.add(start_doy).updateMask(self.active).rename("sowing_doy")

    # ----------------------------------------------------- per-date products
    def das(self, date) -> ee.Image:
        t = ee.Date(date).difference(self.cube.start, "day")
        return ee.Image.constant(t).float().subtract(self.sowing_t).rename("DAS")

    def stage(self, date) -> ee.Image:
        d = self.das(date)
        out = ee.Image.constant(STAGE_OUT_OF_SEASON)
        for c in self.season.classes:
            if not c.irrigable:
                continue
            b1, b2, b3, b4 = stage_boundaries(c)
            token = (ee.Image.constant(STAGE_OUT_OF_SEASON)
                     .where(d.gte(0).And(d.lte(b1)), STAGE_INITIAL)
                     .where(d.gt(b1).And(d.lte(b3)), STAGE_CRITICAL)
                     .where(d.gt(b3).And(d.lte(b4)), STAGE_LATE))
            out = out.where(self.crop_map.eq(c.id), token)
        return out.where(self.active.Not(), STAGE_OUT_OF_SEASON).rename("PAM_stage").toByte()

    def kc(self, date) -> ee.Image:
        d = self.das(date)
        out = ee.Image.constant(0).float()
        for c in self.season.classes:
            if not c.irrigable:
                continue
            b1, b2, b3, b4 = stage_boundaries(c)
            ini, mid, end = c.kc["ini"], c.kc["mid"], c.kc["end"]
            s_dev = (mid - ini) / (b2 - b1)
            s_late = (end - mid) / (b4 - b3)
            # Constants are inlined: expression() maps take images only.
            kc_c = d.expression(
                f"(D < 0) ? 0"
                f" : (D <= {b1}) ? {ini}"
                f" : (D <= {b2}) ? {ini} + (D - {b1}) * ({s_dev})"
                f" : (D <= {b3}) ? {mid}"
                f" : (D <= {b4}) ? {mid} + (D - {b3}) * ({s_late})"
                f" : 0",
                {"D": d},
            )
            out = out.where(self.crop_map.eq(c.id), kc_c)
        return out.where(self.active.Not(), 0).rename("Kc")

    def percolation(self, date, days: int) -> ee.Image:
        """Extra paddy water need (mm over the period) while the crop is in the field."""
        stage = self.stage(date)
        out = ee.Image.constant(0).float()
        for c in self.season.classes:
            if c.irrigable and c.percolation_mm_day:
                out = out.where(self.crop_map.eq(c.id).And(stage.gt(0)),
                                c.percolation_mm_day * days)
        return out.rename("percolation")
