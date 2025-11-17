Robo-Oracle: Causal Failure Diagnosis for Robotic Systems
==========================================================

🤖 **NEW: Robo-Oracle Implementation**

This repository now includes **Robo-Oracle**, a novel framework for self-supervised causal failure attribution in opaque end-to-end robotic policies. Robo-Oracle uses the classical pipeline as a "ground-truth oracle" to train diagnostic models that can explain *why* black-box policies fail.

📖 **Quick Links:**
- **Complete Documentation**: [ROBO_ORACLE_README.md](ROBO_ORACLE_README.md) - Full system documentation
- **Quick Start Guide**: [QUICKSTART.md](QUICKSTART.md) - Get started in 30 minutes
- **Run Full Pipeline**: `python run_robo_oracle_pipeline.py --mode quick-test`

---

## Repository Structure

This repository contains two main components:

### 1. **Classical Pipeline (R2)** - The "Oracle"
The original deterministic, white-box robotics pipeline with layered failure attribution.

### 2. **Robo-Oracle** - Self-Supervised Failure Diagnosis
A complete system for training diagnostic models using the classical pipeline as a supervisor.

---

R2: Layered Failure Attribution & Explainability Metrics for Robotics
====================================================================

Overview
--------
This repository implements a complete, offline‑reproducible pipeline for a synthetic robotics task "Lift" with layered failure attribution across modules: Vision → Geometry → Planning → Control. The pipeline supports perturbation sweeps, JSONL logging, rule‑based attribution, metrics aggregation, PNG plots, and an offline Plotly dashboard. Everything runs with pure Python 3.10+, numpy, pandas, matplotlib, and plotly (offline), with deterministic seeds and no Internet or external assets.

**Note**: This classical pipeline serves as the "Oracle" in the Robo-Oracle system, providing ground-truth causal labels for training diagnostic models.

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

Use the one‑click script:

  bash reproduce.sh

This runs a perturbation sweep for the occlusion scenario and generates CSV, PNG plots, and an offline HTML dashboard under `results/`.

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
