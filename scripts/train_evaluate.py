"""Train the crop classifier and print an honest accuracy report.

Usage
  python scripts/train_evaluate.py                      # uses config.yaml
  python scripts/train_evaluate.py --labels labels/meerut_kharif_2025.geojson
  python scripts/train_evaluate.py --labels projects/<proj>/assets/pamrofi_labels_kharif2025 \
         --roi meerut_district --out reports/accuracy.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pamrofi import earthengine  # noqa: E402
from pamrofi.config import Config  # noqa: E402
from pamrofi.pipeline import Pipeline, PipelineOptions  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--season", default=None)
    ap.add_argument("--roi", default=None)
    ap.add_argument("--labels", default=None, help="asset id, GeoJSON path, or 'weak'")
    ap.add_argument("--out", default="reports/accuracy.json")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.labels:
        cfg = cfg.with_overrides(classification={"labels": args.labels})
    earthengine.init(cfg.ee_project)

    pipe = Pipeline(cfg, PipelineOptions(season_key=args.season, roi_key=args.roi))
    print(f"Season: {pipe.season.label}")
    print(f"Labels: {cfg.section('classification')['labels']}")
    m = pipe.evaluate()
    if m["overall_accuracy"] is None:
        print("No test samples — check that your labels overlap the ROI.")
        return

    if not pipe.validated:
        print("\n!!! WEAK LABELS: numbers below are agreement with rules, NOT accuracy !!!\n")
    names = m["names"]
    w = max(len(n) for n in names) + 2
    print("Confusion matrix (rows = reference, cols = predicted)")
    print(" " * w + "".join(f"{n[:10]:>12}" for n in names))
    for n, row in zip(names, m["matrix"]):
        print(f"{n:<{w}}" + "".join(f"{v:>12}" for v in row))
    print(f"\nOverall accuracy: {m['overall_accuracy']*100:.2f}%   Kappa: {m['kappa']:.4f}")
    print(f"Train / test samples: {pipe.train_result.n_train} / {pipe.train_result.n_test}\n")
    print(f"{'class':<{w}}{'producer':>10}{'user':>10}{'F1':>8}{'n':>7}")
    for n, pc in m["per_class"].items():
        fmt = lambda x: f"{x*100:9.1f}%" if x is not None else "       –"  # noqa: E731
        print(f"{n:<{w}}{fmt(pc['producers_accuracy'])}{fmt(pc['users_accuracy'])}"
              f"{pc['f1']:8.3f}{pc['support']:7d}")

    imp = sorted(pipe.train_result.importance.items(), key=lambda kv: -kv[1])[:10]
    if imp:
        print("\nTop features:")
        for k, v in imp:
            print(f"  {k:<28}{v:10.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"validated": pipe.validated, "metrics": m,
                               "importance": pipe.train_result.importance,
                               "n_train": pipe.train_result.n_train,
                               "n_test": pipe.train_result.n_test}, indent=2))
    print(f"\nSaved report → {out}")


if __name__ == "__main__":
    main()
