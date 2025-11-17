R2: Layered Failure Attribution & Explainability Metrics for Robotics
====================================================================

Overview
--------
This repository implements a complete, offline‑reproducible pipeline for a synthetic robotics task “Lift” with layered failure attribution across modules: Vision → Geometry → Planning → Control. The pipeline supports perturbation sweeps, JSONL logging, rule‑based attribution, metrics aggregation, PNG plots, and an offline Plotly dashboard. Everything runs with pure Python 3.10+, numpy, pandas, matplotlib, and plotly (offline), with deterministic seeds and no Internet or external assets.

Pipeline Diagram (ASCII)
------------------------

  [Synth Env] --render--> [Vision: Detect/Segment] --ctx--> [Geometry: Pose]
          |                                      |                 |
          +---- ctx(noise/obstacles) <-----------+-----------------+
                                                               |
  [Perturbations] --> ctx ----> [Planning: RRT* Fallback] ----> [Control: PD]
                                                               |
          [Logging JSONL] <------ [Aggregation & Attribution] <--+
                                 |           |
                                 v           v
                          [PNG Plots]   [Offline HTML Dashboard]

Installation
------------
- Python 3.10+
- Install dependencies:

  pip install -r requirements.txt

Running the End-to-End Sweep
----------------------------

### Legacy Single-Scenario Run

Use the one‑click script:

  bash reproduce.sh

This runs a perturbation sweep for the occlusion scenario and generates CSV, PNG plots, and an offline HTML dashboard under `results/`.

### 🆕 Publication-Ready Comprehensive Experiment Suite

**NEW**: Run all 5 scenarios with comprehensive statistical analysis, validation, and interactive dashboards:

  # Full suite (100 runs per condition, ~60 minutes)
  ./run_all_experiments.sh

  # Quick test (10 runs per condition, ~10 minutes)
  ./run_all_experiments.sh --quick

**Includes**:
- ✅ 5 comprehensive scenarios (occlusion, module_failure, data_corruption, noise_injection, adversarial_patches)
- ✅ Real 3D models (human, vehicle, traffic sign)
- ✅ Sufficient statistical power (100 runs per condition)
- ✅ Automated validation and quality checks
- ✅ Cross-scenario comparative analysis
- ✅ Module vulnerability analysis
- ✅ Beautiful interactive HTML dashboards

See **[EXPERIMENTS.md](EXPERIMENTS.md)** for complete documentation.

Run a Single Scenario
---------------------

Example: single level at 0.4 for occlusion with 3 runs (uses synthetic stubs):

  python scripts/run_suite.py --cfg configs/robosuite_grasp.yaml \
    --scenario occlusion --level 0.4 \
    --thresholds configs/thresholds.yaml \
    --runs 3 \
    --out results/logs/occlusion_0p4.jsonl

Then export a report from a merged CSV:

  python scripts/export_report.py --csv results/occlusion_sweep.csv --out results/report/occlusion

Replace Stubs with Real Systems
-------------------------------
- Vision (`vision/*`): swap detector/segmenter with real models (e.g., YOLO/Mask)
  keeping the same function interfaces.
- Geometry (`geometry/pose_pnp_stub.py`): replace with real PnP/registration while
  preserving returned keys.
- Planning (`planning/rrt_star_fallback.py`): integrate OMPL or other planners
  but keep the output structure.
- Control (`control/*`): hook up a real IK/trajectory controller; maintain the
  same outputs.

Thresholds & Configs
--------------------
Thresholds live in `configs/thresholds.yaml`. You can tune confidence, IoU, PnP RMSE, collision allowance, tracking RMSE, and system success distance thresholds without changing code. Run configuration (`configs/robosuite_grasp.yaml`) controls task, robot, camera, steps, seeds, and output directory. Perturbation levels are in `configs/perturbations.yaml`.

Schema Example
--------------
Each run logs a single JSON object per line (JSONL):

  {
    "run_id": "2024-01-01T00:00:00Z_0",
    "meta": {"task": "Lift", "robot": "Panda", "seed": 0, "scenario": "occlusion", "level": 0.4, "steps": 150, "utc_time": "2024-01-01T00:00:00Z"},
    "perception": {"avg_conf": 0.72, "detected": true, "seg_iou": 0.55, "bbox": [120, 100, 200, 180]},
    "geometry": {"pnp_success": true, "pnp_rmse": 1.2, "pose_estimate": [0.42, 0.61, 0.0]},
    "planning": {"success": true, "path_cost": 0.85, "collisions": 0, "planner": "RRTstar"},
    "control": {"track_rmse": 0.01, "overshoot": 0.02, "oscillation": false},
    "system": {"success": true, "final_dist_to_goal": 0.02, "stop_reason": null},
    "artifacts": {"rgb_path": "results/artifacts/occlusion_0.4_seed0_rgb.png", "mask_path": "results/artifacts/occlusion_0.4_seed0_mask.png", "path_plot": "results/artifacts/occlusion_0.4_seed0_path.png"}
  }

Tests
-----
- PyTest is optional. You can run tests via Python:

  python -m tests.test_schemas
  python -m tests.test_attribution

Both tests validate schema conformity and the attribution logic with synthetic logs.

Notes
-----
- All modules include type hints and docstrings.
- No prints in libraries; scripts avoid chatty output.
- All plotting is offline (matplotlib PNGs; Plotly HTML via offline embedding).
- Paths are handled cross‑platform via `os.path`.
