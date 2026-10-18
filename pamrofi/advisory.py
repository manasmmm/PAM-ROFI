"""8-day crop water balance and irrigation advisory (Objective 3) on Earth Engine.

Fixes vs. the original notebook:
* The original read the MOD16 ``ET`` band (actual ET) and called it ET0.
  Here ET0 is either Hargreaves from ERA5-Land temperatures (FAO-56 eq. 52,
  default) or MOD16 ``PET``.
* Kc comes from each pixel's own DAS (per-pixel sowing), not one regional date.
* Only effective rain is credited, and paddy gets its percolation loss.
* The advisory is for a chosen 8-day period, not a season median.
Formulas mirror ``agro.water_deficit`` / ``agro.advisory_level``.
"""
from __future__ import annotations

import math

import ee

from . import data
from .agro import GSC
from .phenology import PhenologyModel


def _ra_image(date: ee.Date) -> ee.Image:
    """FAO-56 eq. 21 extraterrestrial radiation (MJ m-2 day-1) for every pixel."""
    doy = ee.Number.parse(date.format("D"))
    phi = ee.Image.pixelLonLat().select("latitude").multiply(math.pi / 180)
    ang = doy.multiply(2 * math.pi / 365)
    dr = ang.cos().multiply(0.033).add(1)
    delta = ang.subtract(1.39).sin().multiply(0.409)
    ws = phi.tan().multiply(ee.Image.constant(delta).tan()).multiply(-1).clamp(-1, 1).acos()
    term = (ws.multiply(phi.sin()).multiply(ee.Image.constant(delta).sin())
            .add(phi.cos().multiply(ee.Image.constant(delta).cos()).multiply(ws.sin())))
    return term.multiply(ee.Image.constant(dr)).multiply(24 * 60 / math.pi * GSC).rename("Ra")


def _sum_or_masked(col: ee.ImageCollection, band: str) -> ee.Image:
    """Sum of a collection, or a fully-masked image if no data exists yet
    (e.g. a date newer than the dataset's latency)."""
    empty = ee.Image.constant(0).rename(band).selfMask()
    return ee.Image(ee.Algorithms.If(col.size().gt(0), col.sum().rename(band), empty))


def hargreaves_daily(img: ee.Image) -> ee.Image:
    date = ee.Date(img.get("system:time_start"))
    tmax, tmin = img.select("tmax"), img.select("tmin")
    tmean = tmax.add(tmin).divide(2)
    et0 = (tmean.add(17.8).multiply(tmax.subtract(tmin).max(0).sqrt())
           .multiply(_ra_image(date)).multiply(0.0023 * 0.408))
    return et0.rename("ET0").copyProperties(img, ["system:time_start"])


class AdvisoryModel:
    def __init__(self, pheno: PhenologyModel, cfg: dict):
        self.pheno = pheno
        self.roi = pheno.cube.roi
        self.cfg = cfg
        self.days = int(cfg.get("period_days", 8))
        self.f_eff = float(cfg.get("effective_rain_factor", 0.8))
        self.low, self.high = cfg.get("thresholds_mm", [10, 30])

    def et0(self, start: ee.Date) -> ee.Image:
        end = start.advance(self.days, "day")
        if self.cfg.get("et0_source", "hargreaves_era5") == "modis_pet":
            col = data.modis_pet(self.roi, start.advance(-4, "day"), start.advance(4, "day"))
            # MOD16 periods are 8 days, scale to our period length just in case
            pet = _sum_or_masked(col.limit(1), "ET0")
            return pet.multiply(self.days / 8).rename("ET0")
        daily = data.era5_daily(self.roi, start, end).map(hargreaves_daily)
        return _sum_or_masked(daily, "ET0").resample("bilinear")

    def rain(self, start: ee.Date) -> ee.Image:
        end = start.advance(self.days, "day")
        if self.cfg.get("rain_source", "chirps") == "era5":
            col = data.era5_daily(self.roi, start, end).select("rain")
        else:
            col = data.chirps(self.roi, start, end)
        return _sum_or_masked(col, "rain").resample("bicubic")

    def advisory(self, date) -> ee.Image:
        start = ee.Date(date)
        kc = self.pheno.kc(start.advance(self.days // 2, "day"))
        et0 = self.et0(start)
        rain = self.rain(start)
        perc = self.pheno.percolation(start, self.days)
        etc = et0.multiply(kc).rename("ETc")
        deficit = (etc.add(perc).subtract(rain.multiply(self.f_eff)).max(0)
                   .rename("deficit"))
        level = (ee.Image.constant(0)
                 .where(deficit.gte(self.low), 1)
                 .where(deficit.gt(self.high), 2)
                 .rename("advisory").toByte())
        active = self.pheno.active
        out = ee.Image.cat([level, deficit, etc, et0, rain, kc, perc])
        return out.updateMask(active).clip(self.roi)
