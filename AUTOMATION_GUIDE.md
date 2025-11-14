# R2: Complete Automation Pipeline Guide

## 🚀 Quick Start - One Command!

Run the complete research pipeline (validation → experiments → analysis → visualization → report):

```bash
bash run_all.sh
```

That's it! This single command will:
1. ✅ Validate environment and run tests
2. 📊 Generate synthetic data automatically
3. 🔬 Run all perturbation experiments (5 scenarios × multiple levels)
4. 📈 Generate visualizations (PNG plots + HTML dashboards)
5. 📝 Create comprehensive final research report with discussion

**Total time:** ~5-15 minutes (depending on `--runs` parameter)

---

## 📋 What Gets Automated?

### Complete Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────┐
│                   R2 AUTOMATION PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. VALIDATION                                                  │
│     • Python version check                                      │
│     • Dependencies verification                                 │
│     • Schema & attribution tests                                │
│                                                                 │
│  2. DATA GENERATION (Automatic - Synthetic)                     │
│     • 2D "Lift" task environment                                │
│     • Deterministic seeds for reproducibility                   │
│     • No manual data downloading needed                         │
│                                                                 │
│  3. EXPERIMENTATION                                             │
│     • Occlusion sweep       [0.0, 0.2, 0.4, 0.6]               │
│     • Lighting sweep        [0.0, 0.3, 0.6]                    │
│     • Motion blur sweep     [0.0, 0.5, 1.0]                    │
│     • Camera jitter sweep   [0.0, 0.3, 0.6]                    │
│     • Overlap sweep         [0.0, 0.5]                         │
│                                                                 │
│  4. ANALYSIS                                                    │
│     • Per-run attribution (rule-based)                          │
│     • Module failure tracking                                   │
│     • Root cause identification                                 │
│     • Cross-scenario comparison                                 │
│                                                                 │
│  5. VISUALIZATION                                               │
│     • Stacked bar charts (module failures)                      │
│     • Sensitivity curves (degradation vs perturbation)          │
│     • Sankey diagrams (failure flow)                            │
│     • Interactive HTML dashboards                               │
│                                                                 │
│  6. FINAL REPORT                                                │
│     • Executive summary                                         │
│     • Key findings                                              │
│     • Discussion & interpretation                               │
│     • Recommendations                                           │
│     • Future work                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Usage

### Basic Usage (Default Settings)

```bash
bash run_all.sh
```

**Defaults:**
- Runs per level: 3
- Results directory: `results/`

### Custom Configuration

```bash
# More runs for statistical significance
bash run_all.sh 10

# Custom results directory
bash run_all.sh 3 my_experiment_results

# Advanced: Python script directly with all options
python scripts/run_all.py \
    --runs 5 \
    --results_dir custom_results \
    --cfg configs/robosuite_grasp.yaml \
    --thresholds configs/thresholds.yaml \
    --perturbations configs/perturbations.yaml
```

### Skip Validation (Faster, Not Recommended)

```bash
python scripts/run_all.py --skip_validation
```

---

## 📊 Output Structure

After running `bash run_all.sh`, you'll get:

```
results/
├── final_report/
│   └── research_report.html          ← START HERE! Comprehensive analysis
│
├── reports/
│   ├── occlusion/
│   │   ├── dashboard.html             ← Interactive Plotly dashboard
│   │   ├── stacked.png                ← Module failure distribution
│   │   ├── sensitivity.png            ← Performance degradation curve
│   │   └── sankey.png                 ← Failure flow diagram
│   ├── lighting/
│   │   └── [same structure]
│   ├── motion_blur/
│   │   └── [same structure]
│   ├── camera_jitter/
│   │   └── [same structure]
│   └── overlap/
│       └── [same structure]
│
├── occlusion_sweep.csv                ← Raw data
├── lighting_sweep.csv
├── motion_blur_sweep.csv
├── camera_jitter_sweep.csv
├── overlap_sweep.csv
│
├── logs/                              ← JSONL logs per run
│   └── [individual .jsonl files]
│
└── artifacts/                         ← Images, masks, path plots
    └── [RGB, masks, path visualizations]
```

---

## 🔍 Understanding the Results

### 1. Final Research Report (START HERE!)

Open in browser:
```bash
# Linux/Mac
open results/final_report/research_report.html

# Or manually navigate to:
file:///path/to/xai/results/final_report/research_report.html
```

**Contents:**
- **Executive Summary**: Overall statistics
- **Key Findings**: Top insights (success rates, module vulnerabilities, root causes, sensitivity)
- **Detailed Results**: Per-scenario breakdowns with tables
- **Discussion**: Interpretation, methodology, limitations
- **Recommendations**: Actionable next steps
- **Future Work**: Research directions

### 2. Interactive Dashboards

Each scenario has an interactive Plotly dashboard:
```bash
open results/reports/occlusion/dashboard.html
```

Explore:
- Hover over charts for details
- Filter data dynamically
- Zoom into specific perturbation levels

### 3. PNG Visualizations

Three types of plots per scenario:

1. **Stacked Bar Chart** (`stacked.png`)
   - Shows which modules fail at each perturbation level
   - Color-coded by module (Vision, Geometry, Planning, Control)

2. **Sensitivity Curve** (`sensitivity.png`)
   - X-axis: Perturbation level
   - Y-axis: Success rate
   - Shows performance degradation

3. **Sankey Diagram** (`sankey.png`)
   - Flow from perturbation → module failures → root causes
   - Width indicates frequency

### 4. Raw Data (CSV)

For custom analysis:
```python
import pandas as pd
df = pd.read_csv("results/occlusion_sweep.csv")
print(df.head())
```

**Columns:**
- Metadata: `run_id`, `scenario`, `level`, `seed`
- Perception: `perception.avg_conf`, `perception.detected`, `perception.seg_iou`
- Geometry: `geometry.pnp_success`, `geometry.pnp_rmse`
- Planning: `planning.success`, `planning.path_cost`, `planning.collisions`
- Control: `control.track_rmse`, `control.overshoot`, `control.oscillation`
- System: `system.success`, `system.final_dist_to_goal`
- Attribution: `attr_modules`, `attr_errors`, `root_cause`

---

## ⚙️ Configuration

### Modify Perturbation Scenarios

Edit `configs/perturbations.yaml`:

```yaml
scenarios:
  - name: "occlusion"
    levels: [0.0, 0.2, 0.4, 0.6, 0.8]  # Add more levels
  - name: "lighting"
    levels: [0.0, 0.3, 0.6, 0.9]       # Increase range
  # Add new scenarios...
```

### Adjust Success Thresholds

Edit `configs/thresholds.yaml`:

```yaml
perception:
  min_confidence: 0.5      # Detection confidence threshold
  min_seg_iou: 0.6         # Segmentation quality threshold

geometry:
  max_pnp_rmse: 2.0        # Pose estimation error threshold

planning:
  max_collisions: 0        # Zero tolerance for collisions
  max_path_cost_ratio: 2.0

control:
  max_track_rmse: 0.05     # Tracking accuracy threshold
  max_overshoot: 0.1

system:
  max_final_distance: 0.05  # Success criterion: within 5cm
```

### Change Task Configuration

Edit `configs/robosuite_grasp.yaml`:

```yaml
task: "Lift"
robot: "Panda"
camera:
  resolution: [640, 480]  # Increase resolution
  fov: 60

simulation:
  max_steps: 150          # Longer episodes
  seeds: [0, 1, 2, 3, 4]  # More seeds for diversity
```

---

## 🧪 Running Individual Components

### Run Tests Only

```bash
python -m tests.test_schemas
python -m tests.test_attribution
```

### Run Single Scenario

```bash
python scripts/sweep_perturb.py \
    --cfg configs/robosuite_grasp.yaml \
    --scenario occlusion \
    --levels 0.0 0.2 0.4 0.6 \
    --thresholds configs/thresholds.yaml \
    --runs 3 \
    --merge_csv results/occlusion_only.csv
```

### Generate Report from Existing Data

```bash
python scripts/export_report.py \
    --csv results/occlusion_sweep.csv \
    --out results/reports/occlusion \
    --thresholds configs/thresholds.yaml
```

### Generate Final Report from Multiple CSVs

```bash
python scripts/generate_final_report.py \
    --csv results/occlusion_sweep.csv \
    --csv results/lighting_sweep.csv \
    --csv results/motion_blur_sweep.csv \
    --results_dir results/reports \
    --output_dir results/final_report
```

---

## 🔬 Extending the Pipeline

### Add New Perturbation Type

1. Create perturbation module:
   ```python
   # perturb/my_perturbation.py
   def apply_my_perturbation(img, level):
       # Your perturbation logic
       return perturbed_img
   ```

2. Register in `configs/perturbations.yaml`:
   ```yaml
   scenarios:
     - name: "my_perturbation"
       levels: [0.0, 0.5, 1.0]
   ```

3. Run pipeline:
   ```bash
   bash run_all.sh
   ```

### Replace Stub Modules with Real Implementations

**Example: Replace detector stub with YOLO**

1. Install YOLO:
   ```bash
   pip install ultralytics
   ```

2. Modify `vision/detector_stub.py`:
   ```python
   from ultralytics import YOLO

   def detect(img):
       model = YOLO('yolov8n.pt')
       results = model(img)
       # Convert to R2 format
       return {
           "detected": len(results) > 0,
           "avg_conf": results[0].boxes.conf.mean(),
           "bbox": results[0].boxes.xyxy[0],
       }
   ```

3. Pipeline automatically uses new detector:
   ```bash
   bash run_all.sh
   ```

---

## 📈 Interpreting Key Metrics

### Success Rate
- **100%**: Perfect robustness (no failures at any perturbation level)
- **50-100%**: Degraded but functional
- **0-50%**: Severe impact
- **0%**: Complete failure

### Sensitivity Slope
- **Close to 0**: Graceful degradation (robust)
- **-0.5 to -1.0**: Moderate sensitivity
- **< -1.0**: Rapid failure cascade (vulnerable)

### Module Failure Frequency
- **High Vision failures**: Detection/segmentation issues
- **High Geometry failures**: Pose estimation problems
- **High Planning failures**: Path finding difficulties
- **High Control failures**: Trajectory tracking errors

### Root Causes
- **Occlusion**: Object blocking
- **Lighting**: Illumination variations
- **MotionBlur**: Camera/object motion
- **CameraJitter**: Camera instability
- **Overlap**: Multiple objects interfering

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'pandas'"

Install dependencies:
```bash
pip install -r requirements.txt
```

### "Permission denied" when running bash script

Make executable:
```bash
chmod +x run_all.sh
```

### Tests failing

Check Python version (requires 3.10+):
```bash
python --version
```

### Out of memory

Reduce runs:
```bash
bash run_all.sh 1  # Use 1 run per level instead of 3
```

### Want to re-run specific scenario

Delete CSV and re-run:
```bash
rm results/occlusion_sweep.csv
bash run_all.sh
```

---

## 📚 Additional Resources

- **Main README**: `README.md`
- **Original reproduce script**: `reproduce.sh` (single occlusion scenario)
- **Threshold config**: `configs/thresholds.yaml`
- **Perturbation config**: `configs/perturbations.yaml`
- **Test files**: `tests/`

---

## 🎉 Summary

**One command. Complete automation. Comprehensive results.**

```bash
bash run_all.sh
```

Then open: `results/final_report/research_report.html`

**That's it!** The entire research pipeline runs automatically:
- ✅ No manual data downloading
- ✅ No manual experiment execution
- ✅ No manual analysis
- ✅ No manual visualization
- ✅ Comprehensive final report with discussion

**Perfect for:**
- 🔬 Reproducible research
- 📊 Systematic benchmarking
- 🎓 Educational demonstrations
- 🏭 Robustness testing

---

**Questions or issues?** Check the troubleshooting section above or examine individual scripts for detailed documentation.
