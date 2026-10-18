"""End-to-end PAM-ROFI pipeline: one object that the app, the CLI scripts and
notebooks all share."""
from __future__ import annotations

from dataclasses import dataclass

import ee

from . import classify, data
from .advisory import AdvisoryModel
from .config import Config, Season
from .features import SeasonCube
from .phenology import PhenologyModel
from .stress import StressModel


@dataclass
class PipelineOptions:
    season_key: str | None = None
    roi_key: str | None = None
    roi_geometry: ee.Geometry | None = None   # overrides roi_key (uploaded GeoJSON etc.)
    orbit_pass: str | None = None


class Pipeline:
    def __init__(self, config: Config, options: PipelineOptions | None = None):
        options = options or PipelineOptions()
        self.config = config
        self.season: Season = config.season(options.season_key)
        if options.roi_geometry is not None:
            self.roi = options.roi_geometry
        else:
            self.roi = data.roi_from_config(config.rois[options.roi_key or config.raw["default_roi"]])
        self.cube = SeasonCube(self.roi, self.season.start, self.season.end,
                               self.season.composite_days, options.orbit_pass)
        self.train_result = classify.train(self.season, self.cube, config.section("classification"))
        self.crop_map = self.train_result.crop_map
        self.pheno = PhenologyModel(self.season, self.cube, self.crop_map)
        self.stress_model = StressModel(self.pheno, config.section("stress"))
        self.advisory_model = AdvisoryModel(self.pheno, config.section("advisory"))

    # --------------------------------------------------------------- layers
    @property
    def validated(self) -> bool:
        return self.train_result.validated

    def evaluate(self) -> dict:
        return classify.evaluate(self.train_result, self.season.class_names)

    def stress(self, date) -> ee.Image:
        return self.stress_model.stress(date)

    def advisory(self, date) -> ee.Image:
        return self.advisory_model.advisory(date)

    def sowing_doy(self) -> ee.Image:
        return self.pheno.sowing_date_doy()

    # ---------------------------------------------------------- statistics
    def area_by_class(self, scale: int = 20) -> dict[str, float]:
        """Hectares per crop class inside the ROI."""
        area = ee.Image.pixelArea().divide(10000).addBands(self.crop_map)
        res = area.reduceRegion(
            reducer=ee.Reducer.sum().group(groupField=1, groupName="class"),
            geometry=self.roi, scale=scale, maxPixels=1e10, tileScale=4,
        ).getInfo()
        names = self.season.class_names
        return {names.get(int(g["class"]), str(g["class"])): g["sum"] for g in res.get("groups", [])}

    def advisory_summary(self, date, scale: int = 20) -> list[dict]:
        """Hectares per (crop, advisory level) and mean deficit — a table an
        irrigation officer can act on."""
        adv = self.advisory(date)
        key = self.crop_map.multiply(10).add(adv.select("advisory")).rename("key")
        img = ee.Image.pixelArea().divide(10000).rename("ha").addBands(adv.select("deficit")).addBands(key)
        res = img.reduceRegion(
            # inputs: ha → sum, deficit → mean, key → group
            reducer=ee.Reducer.sum().combine(ee.Reducer.mean(), sharedInputs=False)
            .group(groupField=2, groupName="key"),
            geometry=self.roi, scale=scale, maxPixels=1e10, tileScale=4,
        ).getInfo()
        names = self.season.class_names
        rows = []
        for g in res.get("groups", []):
            k = int(g["key"])
            rows.append({"crop": names.get(k // 10, str(k // 10)), "advisory": k % 10,
                         "hectares": g.get("sum"), "mean_deficit_mm": g.get("mean")})
        return rows

    # ------------------------------------------------------- field inspector
    def point_timeseries(self, lon: float, lat: float) -> dict:
        """Everything about one field over the season, in one getInfo call."""
        pt = ee.Geometry.Point([lon, lat])
        adv_days = self.advisory_model.days

        def comp_feature(img):
            img = ee.Image(img)
            date = ee.Date(img.get("system:time_start"))
            st = self.stress_model.stress(date)
            vals = (img.select(["NDVI", "LSWI", "VV", "VH", "obs"])
                    .addBands(st.select(["stress", "PAM_stage", "VCI", "SMI"]))
                    .addBands(self.pheno.das(date))
                    .reduceRegion(ee.Reducer.first(), pt, 10))
            return ee.Feature(None, vals).set("date", date.format("YYYY-MM-dd"))

        n_days = self.cube.end.difference(self.cube.start, "day")
        adv_offsets = ee.List.sequence(0, n_days.subtract(adv_days), adv_days)

        def adv_feature(off):
            d = self.cube.start.advance(off, "day")
            a = self.advisory_model.advisory(d)
            vals = a.unmask(0).reduceRegion(ee.Reducer.first(), pt, 10)
            return ee.Feature(None, vals).set("date", d.format("YYYY-MM-dd"))

        static = (self.crop_map.addBands(self.pheno.sowing_t).addBands(self.pheno.active)
                  .addBands(self.pheno.metrics)
                  .reduceRegion(ee.Reducer.first(), pt, 10))

        out = ee.Dictionary({
            "static": static,
            "composites": self.cube.composites.map(comp_feature).toList(500).map(
                lambda f: ee.Feature(f).toDictionary()),
            "advisory": adv_offsets.map(lambda o: adv_feature(o).toDictionary()),
            "season_start": self.cube.start.format("YYYY-MM-dd"),
        }).getInfo()
        return out

    # --------------------------------------------------------------- export
    def export_image(self, image: ee.Image, description: str, folder: str = "PAM_ROFI",
                     scale: int = 10) -> ee.batch.Task:
        task = ee.batch.Export.image.toDrive(image=image.toFloat(), description=description,
                                             folder=folder, region=self.roi, scale=scale,
                                             maxPixels=1e10, fileFormat="GeoTIFF")
        task.start()
        return task

    def download_url(self, image: ee.Image, scale: int = 10, name: str = "pamrofi") -> str:
        """Direct GeoTIFF link (works for small ROIs, < ~32 MB)."""
        return image.toFloat().getDownloadURL({"region": self.roi, "scale": scale,
                                               "format": "GEO_TIFF", "name": name})
