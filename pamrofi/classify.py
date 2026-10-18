"""Crop-type classification (Objective 1) with honest validation.

Key change vs. the original notebook: labels and evaluation are separated.
* Real labels (GEE asset or GeoJSON of digitised polygons / GPS points) are split
  into train/test **by polygon**, so pixels from the same field never appear in
  both sets. That is what makes the reported accuracy meaningful.
* "weak" labels still exist for a quick demo, but the result is flagged as
  UNVALIDATED everywhere, because testing a model on rule-generated labels only
  measures agreement with the rules.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import ee

from . import data
from .config import Season
from .features import SeasonCube


# --------------------------------------------------------------- labels
def load_labels(source: str, class_property: str = "class") -> ee.FeatureCollection:
    if source.endswith((".geojson", ".json")):
        path = Path(source)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / path
        with open(path, encoding="utf-8") as fh:
            gj = json.load(fh)
        fc = ee.FeatureCollection(gj)
    else:
        fc = ee.FeatureCollection(source)  # GEE asset id
    return fc.filter(ee.Filter.notNull([class_property]))


def weak_label_image(season: Season, cube: SeasonCube) -> ee.Image:
    """Rule-based seed labels from agronomic signatures (demo only).

    * Non-agri  – ESA WorldCover trees / built-up / bare / water (independent source)
    * fixed-sowing crop (sugarcane) – already green at season start, long green season
    * vh_min crop (paddy) – strong VH flooding dip in the transplant window + high peak NDVI
    * other ndvi_sos crop without fallow rule (e.g. wheat) – clear single green-up
    * class with a fallow rule – remaining cropland
    """
    m = cube.phenology_metrics()
    wc = data.worldcover()
    cropland = wc.eq(data.WORLDCOVER_CROPLAND)
    non_agri = wc.remap(data.WORLDCOVER_NON_AGRI, [1] * len(data.WORLDCOVER_NON_AGRI), 0)
    first = ee.Image(cube.composites.first())

    labels = ee.Image.constant(-1)
    remainder_id = None
    for c in season.classes:
        if not c.irrigable:
            labels = labels.where(non_agri, c.id)
        elif c.sowing_method == "fixed":
            rule = (cropland.And(first.select("NDVI").gt(0.45))
                    .And(m.select("green_days").gte(120)))
            labels = labels.where(rule.And(labels.eq(-1)), c.id)
        elif c.sowing_method == "vh_min":
            s0, s1 = season.sowing_search
            t_vh = cube.vh_minimum_time(s0, s1)
            vh_dip = cube.composites.select("VH").min()
            rule = (cropland.And(vh_dip.lt(-19)).And(m.select("vh_range").gt(5))
                    .And(m.select("ndvi_max").gt(0.6)).And(t_vh.gte(0)))
            labels = labels.where(rule.And(labels.eq(-1)), c.id)
        elif c.fallow_amplitude is None:
            rule = cropland.And(m.select("ndvi_amp").gt(0.4)).And(m.select("ndvi_max").gt(0.6))
            labels = labels.where(rule.And(labels.eq(-1)), c.id)
        else:
            remainder_id = c.id
    if remainder_id is not None:
        labels = labels.where(cropland.And(labels.eq(-1)), remainder_id)
    return labels.updateMask(labels.gte(0)).rename("class").toInt()


def weak_label_points(season: Season, cube: SeasonCube, per_class: int = 400,
                      seed: int = 42) -> ee.FeatureCollection:
    img = weak_label_image(season, cube)
    pts = img.stratifiedSample(numPoints=per_class, classBand="class", region=cube.roi,
                               scale=10, seed=seed, geometries=True, tileScale=4)
    # Weak labels have no fields, so the split falls back to points.
    return pts.randomColumn("rand", seed).map(
        lambda f: f.set("poly_id", f.get("system:index")))


def label_points(labels: ee.FeatureCollection, class_property: str, points_per_polygon: int,
                 seed: int) -> ee.FeatureCollection:
    """Turn labelled polygons (or points) into sample points that remember which
    polygon they came from."""
    labels = labels.randomColumn("rand", seed)

    def to_points(f):
        f = ee.Feature(f)
        geom = f.geometry()
        is_point = ee.String(geom.type()).compareTo("Point").eq(0)
        cls = f.get(class_property)
        rand = f.get("rand")
        pid = f.get("system:index")
        # Shrink polygons by 10 m so edge pixels (mixed with roads/bunds) are skipped,
        # unless the field is too small to survive the shrink.
        inner = geom.buffer(-10, 1)
        region = ee.Geometry(ee.Algorithms.If(inner.area(1).gt(400), inner, geom))
        pts = ee.FeatureCollection(ee.Algorithms.If(
            is_point,
            ee.FeatureCollection([ee.Feature(geom)]),
            ee.FeatureCollection.randomPoints(region, points_per_polygon, seed),
        ))
        return pts.map(lambda p: p.set({"class": cls, "rand": rand, "poly_id": pid}))

    return labels.map(to_points).flatten()


# ------------------------------------------------------------ training
@dataclass
class TrainResult:
    classifier: ee.Classifier
    crop_map: ee.Image
    validated: bool
    n_train: int | None = None
    n_test: int | None = None
    metrics: dict = field(default_factory=dict)
    importance: dict = field(default_factory=dict)


def train(season: Season, cube: SeasonCube, cfg: dict) -> TrainResult:
    source = cfg.get("labels", "weak")
    seed = int(cfg.get("seed", 42))
    test_fraction = float(cfg.get("test_fraction", 0.3))
    validated = source != "weak"

    if validated:
        labels = load_labels(source, cfg.get("class_property", "class")).filterBounds(cube.roi)
        points = label_points(labels, cfg.get("class_property", "class"),
                              int(cfg.get("points_per_polygon", 25)), seed)
    else:
        points = weak_label_points(season, cube, seed=seed)

    stack = cube.classification_stack()
    samples = stack.sampleRegions(collection=points, properties=["class", "rand", "poly_id"],
                                  scale=int(cfg.get("scale", 10)), tileScale=4,
                                  geometries=False)
    train_fc = samples.filter(ee.Filter.gte("rand", test_fraction))
    test_fc = samples.filter(ee.Filter.lt("rand", test_fraction))

    clf = (ee.Classifier.smileRandomForest(numberOfTrees=int(cfg.get("num_trees", 200)),
                                           seed=seed)
           .train(features=train_fc, classProperty="class",
                  inputProperties=stack.bandNames()))
    crop_map = stack.classify(clf).rename("class").toByte().clip(cube.roi)
    result = TrainResult(classifier=clf, crop_map=crop_map, validated=validated)
    result._test_fc = test_fc  # evaluated lazily (getInfo is slow)
    result._train_fc = train_fc
    return result


def evaluate(result: TrainResult, class_names: dict[int, str]) -> dict:
    """Pull the confusion matrix to the client and compute full metrics."""
    tested = result._test_fc.classify(result.classifier)
    order = sorted(class_names)
    em = tested.errorMatrix("class", "classification", order)
    info = ee.Dictionary({
        "matrix": em.array(),
        "n_train": result._train_fc.size(),
        "n_test": result._test_fc.size(),
        "importance": ee.Dictionary(result.classifier.explain()).get("importance"),
    }).getInfo()
    metrics = confusion_metrics(info["matrix"], [class_names[i] for i in order])
    result.metrics = metrics
    result.n_train = info["n_train"]
    result.n_test = info["n_test"]
    result.importance = info.get("importance") or {}
    return metrics


def confusion_metrics(matrix: list[list[int]], names: list[str]) -> dict:
    """OA, kappa, producer's/user's accuracy and F1 from a confusion matrix
    (rows = reference, columns = predicted). Pure Python, unit-tested."""
    n = len(matrix)
    total = sum(sum(r) for r in matrix)
    if total == 0:
        return {"overall_accuracy": None, "kappa": None, "per_class": {}, "matrix": matrix,
                "names": names}
    diag = sum(matrix[i][i] for i in range(n))
    oa = diag / total
    row = [sum(matrix[i]) for i in range(n)]
    col = [sum(matrix[r][i] for r in range(n)) for i in range(n)]
    pe = sum(row[i] * col[i] for i in range(n)) / (total * total)
    kappa = (oa - pe) / (1 - pe) if pe < 1 else 1.0
    per = {}
    for i, name in enumerate(names):
        pa = matrix[i][i] / row[i] if row[i] else None
        ua = matrix[i][i] / col[i] if col[i] else None
        f1 = 2 * pa * ua / (pa + ua) if pa and ua else 0.0
        per[name] = {"producers_accuracy": pa, "users_accuracy": ua, "f1": f1,
                     "support": row[i]}
    return {"overall_accuracy": oa, "kappa": kappa, "per_class": per, "matrix": matrix,
            "names": names}
