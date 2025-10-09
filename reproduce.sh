#!/usr/bin/env bash
set -e
python scripts/sweep_perturb.py --cfg configs/robosuite_grasp.yaml \
  --scenario occlusion --levels 0.0 0.2 0.4 0.6 \
  --thresholds configs/thresholds.yaml --runs 3 \
  --merge_csv results/occlusion_sweep.csv
python scripts/export_report.py --csv results/occlusion_sweep.csv --out results/report/occlusion

