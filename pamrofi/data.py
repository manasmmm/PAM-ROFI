"""Satellite and climate data loaders (Earth Engine)."""
from __future__ import annotations

import json

import ee

S2_ID = "COPERNICUS/S2_SR_HARMONIZED"
CS_PLUS_ID = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
S1_ID = "COPERNICUS/S1_GRD"
MODIS_PET_ID = "MODIS/061/MOD16A2GF"
CHIRPS_ID = "UCSB-CHG/CHIRPS/DAILY"
ERA5_DAILY_ID = "ECMWF/ERA5_LAND/DAILY_AGGR"
WORLDCOVER_ID = "ESA/WorldCover/v200"
GAUL_L2_ID = "FAO/GAUL/2015/level2"
SMAP_ID = "NASA/SMAP/SPL4SMGP/007"

WORLDCOVER_CROPLAND = 40
WORLDCOVER_NON_AGRI = [10, 50, 60, 80]  # trees, built-up, bare, permanent water


# ------------------------------------------------------------------- ROI
def roi_from_config(spec: dict) -> ee.Geometry:
    kind = spec.get("type")
    if kind == "polygon":
        return ee.Geometry.Polygon([spec["coordinates"]])
    if kind == "buffer":
        return ee.Geometry.Point(spec["center"]).buffer(spec["radius_m"]).bounds()
    if kind == "district":
        return district_geometry(spec["district"], spec.get("state"))
    raise ValueError(f"unknown ROI type: {kind}")


def district_geometry(district: str, state: str | None = None) -> ee.Geometry:
    fc = ee.FeatureCollection(GAUL_L2_ID).filter(ee.Filter.eq("ADM2_NAME", district))
    if state:
        fc = fc.filter(ee.Filter.eq("ADM1_NAME", state))
    return fc.geometry()


def roi_from_geojson(geojson: str | dict) -> ee.Geometry:
    gj = json.loads(geojson) if isinstance(geojson, str) else geojson
    if gj.get("type") == "FeatureCollection":
        return ee.FeatureCollection(gj).geometry()
    if gj.get("type") == "Feature":
        return ee.Geometry(gj["geometry"])
    return ee.Geometry(gj)


# ------------------------------------------------------------ Sentinel-2
def _add_indices(img: ee.Image) -> ee.Image:
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    lswi = img.normalizedDifference(["B8", "B11"]).rename("LSWI")  # canopy water content
    evi = img.expression(
        "2.5 * (N - R) / (N + 6 * R - 7.5 * B + 1)",
        {"N": img.select("B8"), "R": img.select("B4"), "B": img.select("B2")},
    ).rename("EVI")
    ndre = img.normalizedDifference(["B8", "B5"]).rename("NDRE")  # red-edge chlorophyll
    return img.addBands([ndvi, ndwi, lswi, evi, ndre])


def sentinel2(roi: ee.Geometry, start, end, cs_threshold: float = 0.6) -> ee.ImageCollection:
    """Cloud-masked (Cloud Score+) Sentinel-2 SR with reflectance in 0–1 and indices."""
    cs = ee.ImageCollection(CS_PLUS_ID)

    def prep(img):
        clear = img.select("cs_cdf").gte(cs_threshold)
        refl = img.select(["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12"]).divide(10000)
        return _add_indices(refl.updateMask(clear)).copyProperties(img, ["system:time_start"])

    return (ee.ImageCollection(S2_ID)
            .filterBounds(roi)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
            .linkCollection(cs, ["cs_cdf"])
            .map(prep))


# ------------------------------------------------------------ Sentinel-1
def sentinel1(roi: ee.Geometry, start, end, orbit_pass: str | None = None,
              speckle_radius_m: int = 30) -> ee.ImageCollection:
    """Sentinel-1 IW GRD VV+VH in dB, edge-masked and speckle-filtered, plus VH/VV ratio."""
    col = (ee.ImageCollection(S1_ID)
           .filterBounds(roi)
           .filterDate(start, end)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    if orbit_pass:
        col = col.filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))

    def prep(img):
        valid = img.select("VV").gt(-30)
        vvvh = (img.select(["VV", "VH"])
                .updateMask(valid)
                .focalMedian(speckle_radius_m, "circle", "meters"))
        ratio = vvvh.select("VH").subtract(vvvh.select("VV")).rename("VH_VV")  # dB difference = ratio
        return vvvh.addBands(ratio).copyProperties(img, ["system:time_start"])

    return col.map(prep)


# --------------------------------------------------------------- climate
def modis_pet(roi, start, end) -> ee.ImageCollection:
    """MOD16A2GF potential ET, converted to mm per 8-day period (scale 0.1)."""
    return (ee.ImageCollection(MODIS_PET_ID).filterBounds(roi).filterDate(start, end)
            .map(lambda i: i.select("PET").multiply(0.1).rename("ET0")
                 .copyProperties(i, ["system:time_start"])))


def chirps(roi, start, end) -> ee.ImageCollection:
    return (ee.ImageCollection(CHIRPS_ID).filterBounds(roi).filterDate(start, end)
            .select(["precipitation"], ["rain"]))


def era5_daily(roi, start, end) -> ee.ImageCollection:
    """ERA5-Land daily: Tmax/Tmin (°C) and rain (mm)."""
    def prep(i):
        tmax = i.select("temperature_2m_max").subtract(273.15).rename("tmax")
        tmin = i.select("temperature_2m_min").subtract(273.15).rename("tmin")
        rain = i.select("total_precipitation_sum").multiply(1000).rename("rain")
        return tmax.addBands([tmin, rain]).copyProperties(i, ["system:time_start"])

    return ee.ImageCollection(ERA5_DAILY_ID).filterBounds(roi).filterDate(start, end).map(prep)


def worldcover() -> ee.Image:
    return ee.ImageCollection(WORLDCOVER_ID).first().select("Map")


def smap_surface_moisture(roi, start, end) -> ee.ImageCollection:
    """SMAP L4 surface soil moisture (m3/m3) for validating the SAR SMI proxy."""
    return (ee.ImageCollection(SMAP_ID).filterBounds(roi).filterDate(start, end)
            .select("sm_surface"))
