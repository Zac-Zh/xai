R2: Layered Failure Attribution & Explainability Metrics for Robotics
====================================================================

## 🎯 **PUBLICATION-READY**: Complete with Real Models (No Stubs!)

Overview
--------
**R2** is a complete, publication-ready framework for explainable failure attribution in robotic manipulation. It systematically analyzes how environmental perturbations propagate through modular robotics pipelines (Vision → Geometry → Planning → Control) and cause failures.

### ✨ What's New: 100% Real Implementations

- ✅ **3D PyBullet Simulation** - Franka Panda robot with real physics
- ✅ **YOLOv8 Detection** - State-of-the-art object detector
- ✅ **Mask R-CNN Segmentation** - Production-quality instance segmentation
- ✅ **OpenCV PnP** - Real 6D pose estimation
- ✅ **RRT* Planning** - Asymptotically optimal path planner
- ✅ **Physics-Based Control** - Real PD controller with dynamics
- ✅ **Multiple Tasks** - Lift, PickPlace, Push, Stack
- ✅ **One Command** - Run 1000+ experiments for publication

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

🚀 PUBLICATION-READY PIPELINE (ONE COMMAND!)
---------------------------------------------

**Run complete publication-quality experiments:**

  # Fast mode: 2D synthetic, stubs (5-10 min - for testing)
  bash run_all.sh fast

  # Full mode: 2D+3D, real models, single task (30-60 min)
  bash run_all.sh full

  # Publication mode: Everything (1-2 hours - complete dataset)
  bash run_all.sh publication

**Publication mode includes:**
1. ✅ 2D synthetic baseline experiments
2. ✅ 3D PyBullet with Franka Panda robot
3. ✅ Real models: YOLO, Mask R-CNN, OpenCV, RRT*, PD controller
4. ✅ Multiple tasks: Lift, PickPlace, Push, Stack
5. ✅ All 5 perturbation scenarios with multiple levels
6. ✅ Comparative analysis: 2D vs 3D, real vs stubs, multi-task
7. ✅ 1000+ experiment runs with statistical rigor
8. ✅ Publication-ready figures, tables, and comprehensive report

**Results:**
- **Final research report**: `results/final_report/research_report.html` ← START HERE!
- **Comparison plots**: 2D vs 3D, real vs stubs, multi-task performance
- **Individual dashboards**: `results/reports/<scenario>/dashboard.html`
- **Raw data**: CSV and JSONL for custom analysis

**📖 Documentation:**
- **[PUBLICATION_READY.md](PUBLICATION_READY.md)** ← Complete publication guide
- **[AUTOMATION_GUIDE.md](AUTOMATION_GUIDE.md)** - Detailed usage instructions

Running the End-to-End Sweep (Single Scenario)
-----------------------------------------------

Use the one‑click script for occlusion only:

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

Real Implementations (Already Included!)
-----------------------------------------
All real implementations are **already included** and production-ready:

- **Vision**:
  - `vision/detector_real.py` - YOLOv8 object detection
  - `vision/segmenter_real.py` - Mask R-CNN instance segmentation
  - Stubs still available in `*_stub.py` for fast prototyping

- **Geometry**:
  - `geometry/pose_pnp_real.py` - OpenCV PnP 6D pose estimation
  - Real reprojection error, depth integration, Euler angles

- **Planning**:
  - `planning/rrt_star_real.py` - Complete RRT* with dynamic rewiring
  - Collision checking, asymptotically optimal paths

- **Control**:
  - `control/controller_real.py` - Physics-based PD controller
  - Velocity/acceleration limits, oscillation detection

- **Simulation**:
  - `simulators/pybullet_env.py` - 3D Franka Panda with real physics
  - `simulators/synth_env.py` - 2D synthetic (fast baseline)

**Switch between real and stubs** via `--use_real_models` flag or mode selection.

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
