
from __future__ import annotations

import os, sys
# Ensure repository root on sys.path for direct script invocation
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
from typing import Any, Dict, List, Tuple

import pandas as pd

from utils.yload import load as yload
from attribution.rule_based import attribute_failure
from viz.plots import plot_stack_bars, plot_sensitivity, plot_sankey
from viz.dashboard import export_html


def _load_thresholds(path: str) -> Dict[str, Any]:
    return yload(path)


def _compute_attributions(df: pd.DataFrame, th: Dict[str, Any]) -> pd.DataFrame:
    mods: List[List[str]] = []
    errs: List[List[str]] = []
    roots: List[str] = []
    for _, row in df.iterrows():
        # Reconstruct log dict minimally
        log = {
            "meta": {"scenario": row["scenario"]},
            "perception": {
                "avg_conf": row.get("perception.avg_conf"),
                "detected": bool(row.get("perception.detected", True)),
                "seg_iou": row.get("perception.seg_iou"),
            },
            "geometry": {
                "pnp_success": bool(row.get("geometry.pnp_success", True)),
                "pnp_rmse": row.get("geometry.pnp_rmse"),
            },
            "planning": {
                "success": bool(row.get("planning.success", True)),
                "path_cost": row.get("planning.path_cost"),
                "collisions": int(row.get("planning.collisions", 0)),
            },
            "control": {
                "track_rmse": row.get("control.track_rmse"),
                "overshoot": row.get("control.overshoot"),
                "oscillation": bool(row.get("control.oscillation", False)),
            },
        }
        fe, root = attribute_failure(log, th)
        mods.append([m for m, _ in fe])
        errs.append([e for _, e in fe])
        roots.append(root)
    df = df.copy()
    df["attr_modules"] = mods
    df["attr_errors"] = errs
    df["root_cause"] = roots
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresholds", default="configs/thresholds.yaml")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    th = _load_thresholds(args.thresholds)
    df = _compute_attributions(df, th)

    # Save per-scenario plots
    for scenario, sdf in df.groupby("scenario"):
        scen_dir = os.path.join(args.out, scenario)
        os.makedirs(scen_dir, exist_ok=True)
        plot_stack_bars(sdf, os.path.join(scen_dir, "stacked.png"))
        plot_sensitivity(sdf, os.path.join(scen_dir, "sensitivity.png"))
        plot_sankey(sdf, os.path.join(scen_dir, "sankey.png"))
        export_html(sdf, scen_dir)


if __name__ == "__main__":
    main()
