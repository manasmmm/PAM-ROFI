"""Phase-aware crop moisture stress (Objective 2) on Earth Engine.

Changes vs. the original notebook:
* VCI uses a proper multi-year baseline for the same calendar window (Kogan
  VCI), not the min/max of the current season only.
* SMI uses multi-year dry/wet SAR references (5th/95th percentile VV), a
  standard change-detection soil moisture proxy.
* The decision matrix reads each pixel's own PAM stage (from its own sowing
  date) instead of a month-based token. Logic mirrors ``agro.stress_level``.
"""
from __future__ import annotations

import ee

from . import data
from .agro import STAGE_CRITICAL, STAGE_INITIAL, STAGE_LATE
from .phenology import PhenologyModel


class StressModel:
    def __init__(self, pheno: PhenologyModel, cfg: dict):
        self.pheno = pheno
        self.cube = pheno.cube
        self.cfg = cfg
        self.years = int(cfg.get("history_years", 4))
        self.window = int(cfg.get("window_days", 8))
        self.fallback = int(cfg.get("fallback_days", 30))
        self._dry, self._wet = self._sar_references()

    # ------------------------------------------------------------ baselines
    def _ndvi_window(self, center: ee.Date, half_days: int) -> ee.Image:
        col = data.sentinel2(self.cube.roi, center.advance(-half_days, "day"),
                             center.advance(half_days, "day")).select("NDVI")
        return ee.Image(ee.Algorithms.If(col.size().gt(0), col.median(),
                                         ee.Image.constant(0).rename("NDVI").selfMask()))

    def _vci_baseline(self, date: ee.Date):
        half = 15
        imgs = [self._ndvi_window(date.advance(-y, "year"), half) for y in range(1, self.years + 1)]
        imgs.append(self._ndvi_window(date, half))
        hist = ee.ImageCollection(imgs)
        return hist.min(), hist.max()

    def _sar_references(self):
        """Driest / wettest radar response over the same season in past years."""
        cols = []
        for y in range(0, self.years + 1):
            s = self.cube.start.advance(-y, "year")
            e = self.cube.end.advance(-y, "year")
            cols.append(data.sentinel1(self.cube.roi, s, e).select("VV"))
        merged = cols[0]
        for c in cols[1:]:
            merged = merged.merge(c)
        p = merged.reduce(ee.Reducer.percentile([5, 95]))
        return p.select("VV_p5"), p.select("VV_p95")

    # ------------------------------------------------------------ products
    def indices(self, date) -> ee.Image:
        """VCI (0–100), SMI (0–1) and flood flag for the window starting at ``date``."""
        d = ee.Date(date)
        end = d.advance(self.window, "day")
        s2 = self.cube.s2
        win = s2.filterDate(d, end)
        wide = s2.filterDate(d.advance(-(self.fallback - self.window) // 2, "day"),
                             end.advance((self.fallback - self.window) // 2, "day"))
        empty = ee.Image.constant([0, 0]).rename(["NDVI", "LSWI"]).selfMask()
        opt = ee.Image(ee.Algorithms.If(
            win.size().gt(0), win.select(["NDVI", "LSWI"]).median(),
            ee.Algorithms.If(wide.size().gt(0), wide.select(["NDVI", "LSWI"]).median(), empty)))
        # a cloudy 8-day window falls back to the wider window pixel by pixel too
        opt = opt.unmask(ee.Image(ee.Algorithms.If(
            wide.size().gt(0), wide.select(["NDVI", "LSWI"]).median(), empty)))

        sar_win = self.cube.s1.filterDate(d.advance(-6, "day"), end.advance(6, "day")).select("VV")
        vv = ee.Image(ee.Algorithms.If(sar_win.size().gt(0), sar_win.median(),
                                       ee.Image.constant(0).rename("VV").selfMask()))

        nmin, nmax = self._vci_baseline(d.advance(self.window // 2, "day"))
        vci = (opt.select("NDVI").subtract(nmin)
               .divide(nmax.subtract(nmin).max(0.05)).multiply(100).clamp(0, 100).rename("VCI"))
        smi = (vv.subtract(self._dry).divide(self._wet.subtract(self._dry).max(0.5))
               .clamp(0, 1).rename("SMI"))
        flooded = opt.select("LSWI").gt(0.1).rename("flooded")
        return ee.Image.cat([vci, smi, flooded, opt.select("NDVI")]).clip(self.cube.roi)

    def stress(self, date) -> ee.Image:
        c = self.cfg
        vs, vw = c.get("vci_severe", 35), c.get("vci_watch", 60)
        ss, sw = c.get("smi_severe", 0.35), c.get("smi_watch", 0.50)
        idx = self.indices(date)
        vci, smi, flooded = idx.select("VCI"), idx.select("SMI"), idx.select("flooded")
        stage = self.pheno.stage(date)

        initial = (ee.Image.constant(0)
                   .where(smi.lt(sw), 1)
                   .where(smi.lt(ss), 2)
                   .where(flooded, 0))
        critical = (ee.Image.constant(0)
                    .where(vci.lt(vw).And(smi.lt(sw)), 1)
                    .where(vci.lt(vs).Or(smi.lt(ss)), 2))
        late = ee.Image.constant(0).where(vci.lt(vs).And(smi.lt(ss)), 1)

        level = (ee.Image.constant(0)
                 .where(stage.eq(STAGE_INITIAL), initial)
                 .where(stage.eq(STAGE_CRITICAL), critical)
                 .where(stage.eq(STAGE_LATE), late)
                 .updateMask(stage.gt(0))
                 .rename("stress").toByte())
        return ee.Image.cat([level, stage, idx])
