"""Check the Sentinel-1 SMI proxy against NASA SMAP L4 surface soil moisture.

SMAP is ~9 km, so this compares the ROI-average SMI over cropland with the
ROI-average SMAP value for each 8-day window. A clear positive correlation is
evidence (for your report/pitch) that the radar index tracks real soil moisture.

Usage
  python scripts/validate_smi.py --roi meerut_buffer_10km
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ee  # noqa: E402

from pamrofi import data, earthengine  # noqa: E402
from pamrofi.config import Config  # noqa: E402
from pamrofi.pipeline import Pipeline, PipelineOptions  # noqa: E402


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--season", default=None)
    ap.add_argument("--roi", default="meerut_buffer_10km")
    ap.add_argument("--out", default="reports/smi_vs_smap.json")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    earthengine.init(cfg.ee_project)
    pipe = Pipeline(cfg, PipelineOptions(season_key=args.season, roi_key=args.roi))
    crop_mask = data.worldcover().eq(data.WORLDCOVER_CROPLAND)
    days = 8

    n_days = pipe.cube.end.difference(pipe.cube.start, "day")
    offsets = ee.List.sequence(0, n_days.subtract(days), days)

    def one(off):
        d = pipe.cube.start.advance(off, "day")
        smi = pipe.stress_model.indices(d).select("SMI").updateMask(crop_mask)
        smap = data.smap_surface_moisture(pipe.roi, d, d.advance(days, "day")).mean()
        a = smi.reduceRegion(ee.Reducer.mean(), pipe.roi, 100, maxPixels=1e10, tileScale=4)
        b = smap.reduceRegion(ee.Reducer.mean(), pipe.roi, 9000)
        return ee.Dictionary({"date": d.format("YYYY-MM-dd"), "smi": a.get("SMI"),
                              "smap": b.get("sm_surface")})

    rows = [r for r in offsets.map(one).getInfo() if r.get("smi") is not None and r.get("smap") is not None]
    if len(rows) < 3:
        print("Not enough overlapping data.")
        return
    r = pearson([x["smi"] for x in rows], [x["smap"] for x in rows])
    for x in rows:
        print(f"{x['date']}  SMI {x['smi']:.3f}   SMAP {x['smap']:.3f} m³/m³")
    print(f"\nPearson r (SMI vs SMAP), n={len(rows)}: {r:.3f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"pearson_r": r, "rows": rows}, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
