"""Regular multi-sensor composites, temporal feature stack and per-pixel
phenology metrics (the "Tabular PAM" features)."""
from __future__ import annotations

import ee

from . import data

OPTICAL_BANDS = ["NDVI", "EVI", "NDWI", "LSWI", "NDRE"]
SAR_BANDS = ["VV", "VH", "VH_VV"]
STACK_BANDS = ["NDVI", "LSWI", "NDRE", "VV", "VH", "VH_VV"]  # per-date classifier inputs


class SeasonCube:
    """All the per-season Earth Engine objects, built once and reused.

    ``composites`` is an ImageCollection of gap-filled N-day composites. Each
    image carries a constant band ``t`` = days since the season start and a band
    ``obs`` = 1 where the optical window really had a cloud-free observation.
    """

    def __init__(self, roi: ee.Geometry, start: str, end: str, step_days: int = 10,
                 orbit_pass: str | None = None):
        self.roi = roi
        self.start = ee.Date(start)
        self.end = ee.Date(end)
        self.step = step_days
        pad = step_days * 2
        self.s2 = data.sentinel2(roi, self.start.advance(-pad, "day"), self.end.advance(pad, "day"))
        self.s1 = data.sentinel1(roi, self.start.advance(-pad, "day"), self.end.advance(pad, "day"),
                                 orbit_pass=orbit_pass)
        self._season_opt = self.s2.filterDate(self.start, self.end).select(OPTICAL_BANDS).median()
        self._season_sar = self.s1.filterDate(self.start, self.end).select(SAR_BANDS).median()
        self.composites = self._build_composites()

    # -------------------------------------------------------------- composites
    def _composite(self, offset):
        offset = ee.Number(offset)
        t0 = self.start.advance(offset, "day")
        t1 = t0.advance(self.step, "day")
        wide0 = t0.advance(-self.step, "day")
        wide1 = t1.advance(self.step, "day")

        opt_win = self.s2.filterDate(t0, t1).select(OPTICAL_BANDS).median()
        opt_wide = self.s2.filterDate(wide0, wide1).select(OPTICAL_BANDS).median()
        sar_win = self.s1.filterDate(t0, t1).select(SAR_BANDS).median()
        sar_wide = self.s1.filterDate(wide0, wide1).select(SAR_BANDS).median()

        # An empty collection's median() has no bands; constant fallbacks keep band
        # names stable so the stack always has the same shape.
        empty_opt = ee.Image.constant([0] * len(OPTICAL_BANDS)).rename(OPTICAL_BANDS).selfMask()
        empty_sar = ee.Image.constant([0] * len(SAR_BANDS)).rename(SAR_BANDS).selfMask()
        n_opt = self.s2.filterDate(t0, t1).size()
        n_opt_wide = self.s2.filterDate(wide0, wide1).size()
        n_sar = self.s1.filterDate(t0, t1).size()
        n_sar_wide = self.s1.filterDate(wide0, wide1).size()

        opt_win = ee.Image(ee.Algorithms.If(n_opt.gt(0), opt_win, empty_opt))
        opt_wide = ee.Image(ee.Algorithms.If(n_opt_wide.gt(0), opt_wide, empty_opt))
        sar_win = ee.Image(ee.Algorithms.If(n_sar.gt(0), sar_win, empty_sar))
        sar_wide = ee.Image(ee.Algorithms.If(n_sar_wide.gt(0), sar_wide, empty_sar))

        obs = opt_win.select("NDVI").mask().gt(0).unmask(0).rename("obs")
        optical = opt_win.unmask(opt_wide).unmask(self._season_opt)
        sar = sar_win.unmask(sar_wide).unmask(self._season_sar)
        t_band = ee.Image.constant(offset).float().rename("t")

        return (optical.addBands(sar).addBands(obs).addBands(t_band)
                .clip(self.roi)
                .set({"system:time_start": t0.millis(),
                      "date": t0.format("YYYY-MM-dd"),
                      "t": offset}))

    def _build_composites(self) -> ee.ImageCollection:
        n_days = self.end.difference(self.start, "day")
        offsets = ee.List.sequence(0, n_days.subtract(1), self.step)
        # EE inspects the callback signature, so a bound method (with `self`) is
        # rejected; wrap it in a plain one-argument function.
        return ee.ImageCollection(offsets.map(lambda offset: self._composite(offset)))

    # ---------------------------------------------------------- feature stack
    def temporal_stack(self) -> ee.Image:
        """One band per (composite, variable): the classifier sees the whole
        seasonal trajectory, not just a single median."""
        return self.composites.select(STACK_BANDS).toBands().float()

    def phenology_metrics(self, sos_fraction: float = 0.35) -> ee.Image:
        """Per-pixel phenology features — this is the 'Tabular PAM' idea done per
        pixel: the classifier gets *when* things happen, not just *how green*."""
        col = self.composites
        ndvi = col.select("NDVI")
        ndvi_max = ndvi.max().rename("ndvi_max")
        ndvi_min = ndvi.min().rename("ndvi_min")
        amp = ndvi_max.subtract(ndvi_min).rename("ndvi_amp")
        t_peak = col.qualityMosaic("NDVI").select("t").rename("t_peak")
        sos = self.start_of_season(sos_fraction).rename("t_sos")

        vh = col.select("VH")
        vh_min = vh.min().rename("vh_min")
        vh_range = vh.max().subtract(vh_min).rename("vh_range")
        t_vh_min = col.map(lambda i: i.addBands(i.select("VH").multiply(-1).rename("negVH"))) \
                      .qualityMosaic("negVH").select("t").rename("t_vh_min")
        lswi_mean = col.select("LSWI").mean().rename("lswi_mean")
        season_len = col.map(
            lambda i: i.select("NDVI").gt(ndvi_min.add(amp.multiply(0.5))).rename("on")
        ).sum().multiply(self.step).rename("green_days")

        return ee.Image.cat([ndvi_max, ndvi_min, amp, t_peak, sos, vh_min, vh_range,
                             t_vh_min, lswi_mean, season_len]).float()

    def start_of_season(self, fraction: float = 0.35) -> ee.Image:
        """First composite time (days since season start) where NDVI rises above
        min + fraction × amplitude, searching up to the NDVI peak."""
        col = self.composites
        ndvi_min = col.select("NDVI").min()
        thr = ndvi_min.add(col.select("NDVI").max().subtract(ndvi_min).multiply(fraction))
        t_peak = col.qualityMosaic("NDVI").select("t")

        def crossing(img):
            above = img.select("NDVI").gt(thr).And(img.select("t").lte(t_peak))
            return img.select("t").updateMask(above)

        return col.map(crossing).min().unmask(t_peak)

    def vh_minimum_time(self, search_start: str, search_end: str) -> ee.Image:
        """Time of minimum VH inside the transplanting window (paddy flooding)."""
        s0 = ee.Number(ee.Date(search_start).difference(self.start, "day"))
        s1 = ee.Number(ee.Date(search_end).difference(self.start, "day"))
        col = self.composites.filter(ee.Filter.rangeContains("t", s0, s1))
        return (col.map(lambda i: i.addBands(i.select("VH").multiply(-1).rename("negVH")))
                .qualityMosaic("negVH").select("t"))

    def classification_stack(self) -> ee.Image:
        return self.temporal_stack().addBands(self.phenology_metrics())
