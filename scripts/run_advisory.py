"""Produce the advisory for one 8-day period: summary table, sample farmer
messages, and (optionally) GeoTIFF exports to Google Drive.

Usage
  python scripts/run_advisory.py --date 2025-09-10
  python scripts/run_advisory.py --date 2025-09-10 --roi meerut_district --export
  python scripts/run_advisory.py --date 2025-09-10 --point 77.68 29.117 --lang hi
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pamrofi import earthengine  # noqa: E402
from pamrofi.agro import ADVISORY_LABELS, FieldAdvice  # noqa: E402
from pamrofi.config import Config  # noqa: E402
from pamrofi.pipeline import Pipeline, PipelineOptions  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="first day of the 8-day period (YYYY-MM-DD)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--season", default=None)
    ap.add_argument("--roi", default=None)
    ap.add_argument("--point", nargs=2, type=float, metavar=("LON", "LAT"))
    ap.add_argument("--lang", default="en", choices=["en", "hi"])
    ap.add_argument("--export", action="store_true", help="export GeoTIFFs to Google Drive")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    earthengine.init(cfg.ee_project)
    pipe = Pipeline(cfg, PipelineOptions(season_key=args.season, roi_key=args.roi))
    days = int(cfg.section("advisory").get("period_days", 8))
    end = (date.fromisoformat(args.date) + timedelta(days=days - 1)).isoformat()

    rows = pipe.advisory_summary(args.date)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"advisory_{args.date}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["crop", "advisory", "advisory_label", "hectares",
                                           "mean_deficit_mm"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "advisory_label": ADVISORY_LABELS[r["advisory"]]})
    print(f"Advisory {args.date} → {end}")
    for r in sorted(rows, key=lambda r: (r["crop"], r["advisory"])):
        print(f"  {r['crop']:<30} {ADVISORY_LABELS[r['advisory']]:<40} "
              f"{r['hectares']:10.1f} ha   mean deficit {r['mean_deficit_mm'] or 0:5.1f} mm")
    print(f"Saved {csv_path}")

    if args.point:
        lon, lat = args.point
        ts = pipe.point_timeseries(lon, lat)
        stat = ts["static"] or {}
        if not stat.get("active"):
            print("Point is non-agricultural or fallow — no advisory.")
        else:
            crop = pipe.season.class_names[int(stat["class"])]
            adv = next((a for a in ts["advisory"] if a["date"] == args.date), None)
            comp = [c for c in ts["composites"] if c["date"] <= args.date]
            stage = int(comp[-1].get("PAM_stage") or 0) if comp else 0
            if adv:
                print(FieldAdvice(crop, stage, float(adv["deficit"]), int(adv["advisory"]),
                                  args.date, end).message(args.lang))

    if args.export:
        for name, img in {
            "crop_map": pipe.crop_map,
            "stress": pipe.stress(args.date).select(["stress", "PAM_stage", "VCI", "SMI"]),
            "advisory": pipe.advisory(args.date).select(["advisory", "deficit", "ETc", "ET0",
                                                         "rain", "Kc"]),
        }.items():
            task = pipe.export_image(img, f"pamrofi_{name}_{args.date}")
            print(f"Export started: {name} (task {task.id}) → Drive/PAM_ROFI")


if __name__ == "__main__":
    main()
