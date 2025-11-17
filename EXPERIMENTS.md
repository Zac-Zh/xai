# XAI Publication-Ready Experiment Suite

## Overview

This repository contains a comprehensive, publication-ready experiment suite for evaluating robustness of vision-based robotic systems under various perturbations.

## Features

### ✅ Complete Implementation

- **5 Comprehensive Scenarios**: Occlusion, module failure, data corruption, noise injection, and adversarial patches
- **Real 3D Models**: Human, vehicle, and traffic sign models for realistic testing
- **Statistical Power**: 100 runs per condition (configurable) for reliable results
- **Automated Validation**: Built-in checks for data quality and consistency
- **Cross-Scenario Analysis**: Comparative robustness metrics across all scenarios
- **Module Vulnerability Analysis**: Identifies weakest links in the pipeline
- **Interactive Dashboards**: Beautiful HTML visualizations for each scenario

## Quick Start

### Run All Experiments

```bash
# Full experiment suite (100 runs per condition, ~30-60 minutes)
./run_all_experiments.sh

# Quick test (10 runs per condition, ~5 minutes)
./run_all_experiments.sh --quick
```

### Run Specific Scenarios

```bash
# Run only certain scenarios
python3 scripts/run_publication_experiments.py \
  --config configs/robosuite_grasp.yaml \
  --perturbations configs/perturbations.yaml \
  --thresholds configs/thresholds.yaml \
  --output results/publication \
  --scenarios occlusion module_failure
```

### Generate Dashboards

```bash
python3 scripts/generate_dashboards.py \
  --logs-dir results/publication/logs \
  --reports-dir results/publication/reports \
  --output-dir results/publication/reports
```

## Scenarios

### 1. Occlusion
Simulates partial and complete object occlusion by dynamic and static occluders.

**Levels**: 0.0 (no occlusion) to 0.8 (80% coverage)
- Dynamic occluders (e.g., pedestrians, vehicles)
- Static occluders (e.g., walls, furniture)
- Intermittent occlusion patterns

### 2. Module Failure
Simulates component failures in the perception-planning-control pipeline.

**Levels**: 0.0 (no failures) to 0.8 (critical failures)
- Detector confidence degradation
- PnP solver failures
- Path planner timeouts
- Controller gain reduction
- Inter-module communication drops

### 3. Data Corruption
Corrupts sensor data and internal state representations.

**Levels**: 0.0 (no corruption) to 0.8 (severe corruption)
- Pixel dropout (dead pixels)
- Bit flip errors
- Salt and pepper noise
- Color channel corruption
- Calibration data corruption

### 4. Noise Injection
Adds various types of noise to sensors and processing pipeline.

**Levels**: 0.0 (no noise) to 0.8 (high noise)
- Gaussian sensor noise
- Photon shot noise
- Motion blur from vibration
- EMI (electromagnetic interference)
- IMU and odometry noise

### 5. Adversarial Patches
Adds adversarial visual patterns designed to fool perception.

**Levels**: 0.0 (no patches) to 0.8 (optimized patches)
- Simple patterns (checkerboard, gradient)
- Random noise patches
- Optimized adversarial patches
- Targeted vs untargeted attacks

## 3D Models

### Human Model
Simplified parametric human model (~1.7m height)
- Torso, head, arms, legs modeled as capsules/spheres
- Realistic proportions for occlusion testing

### Vehicle Model
Standard car model (~4.5m × 1.8m × 1.5m)
- Main body and cabin
- Four wheels
- Used for traffic scenarios

### Traffic Sign Models
- Stop sign (octagonal)
- Yield sign (triangular)
- Customizable dimensions
- Mounted on poles

## Results Structure

```
results/publication/
├── logs/                              # Raw experimental data
│   ├── occlusion_level0.0.jsonl
│   ├── occlusion_level0.2.jsonl
│   ├── ...
│   ├── module_failure_level0.0.jsonl
│   └── ...
├── reports/                           # Analysis and dashboards
│   ├── comprehensive_analysis.json    # Main analysis report
│   ├── occlusion/
│   │   └── dashboard.html
│   ├── module_failure/
│   │   └── dashboard.html
│   └── ...
└── artifacts/                         # Visualizations
    ├── occlusion_0.0_seed0_rgb.png
    ├── occlusion_0.0_seed0_path.png
    └── ...
```

## Analyses Performed

### 1. Degradation Curves
Success rate vs perturbation level for each scenario and module.

### 2. Statistical Validation
- Sample size checks (n ≥ 100 for 80% power)
- Missing data detection
- Outlier detection (IQR method)
- Variance checks

### 3. Effect Size Analysis
Cohen's d between conditions with interpretation:
- Negligible: |d| < 0.2
- Small: 0.2 ≤ |d| < 0.5
- Medium: 0.5 ≤ |d| < 0.8
- Large: |d| ≥ 0.8

### 4. Cross-Scenario Comparison
- Robustness ranking (area under curve)
- Relative vulnerability assessment
- Failure mode comparison

### 5. Module Vulnerability Analysis
- Failure rates by module (perception, geometry, planning, control)
- Cascading failure detection
- Critical module identification

## Configuration

### Perturbations (`configs/perturbations.yaml`)
```yaml
scenarios:
  - name: "occlusion"
    levels: [0.0, 0.2, 0.4, 0.6, 0.8]
    description: "Partial and complete object occlusion"
    runs_per_level: 100

statistical:
  min_samples_per_condition: 100
  confidence_level: 0.95
  effect_size_target: 0.5

models_3d:
  enabled: true
  types:
    - human
    - vehicle
    - traffic_sign_stop
    - traffic_sign_yield
```

### Thresholds (`configs/thresholds.yaml`)
Success criteria for each module and overall system.

### Experiment Config (`configs/robosuite_grasp.yaml`)
Task, robot, and camera configuration.

## Output Files

### JSONL Logs
Each experiment run produces a JSON line with:
```json
{
  "run_id": "unique_id",
  "meta": {"task": "...", "scenario": "...", "level": 0.5},
  "perception": {"avg_conf": 0.85, "detected": true},
  "geometry": {"pnp_success": true, "pnp_rmse": 0.01},
  "planning": {"success": true, "path_cost": 1.5},
  "control": {"track_rmse": 0.02, "oscillation": false},
  "system": {"success": true, "final_dist_to_goal": 0.05}
}
```

### Comprehensive Analysis Report
JSON file containing:
- Validation results for each scenario
- Cross-scenario robustness ranking
- Module vulnerability analysis
- Detailed statistics

### Interactive Dashboards
HTML files with:
- System degradation curves
- Module-level performance plots
- Statistical tables
- Cross-scenario comparisons
- Color-coded status indicators

## Statistical Power

The suite is designed for publication-quality analysis:

- **100 runs per condition** (configurable)
- **80% statistical power** at α=0.05
- **Effect size detection**: Cohen's d ≥ 0.5
- **95% confidence intervals**

## Validation Checks

Automated validation includes:
- ✓ Sample size adequacy (n ≥ 30 warning, n ≥ 100 recommended)
- ✓ Missing data detection (warning if >50%)
- ✓ Extreme success rate detection (0% or 100%)
- ✓ Zero variance checks
- ✓ Duplicate run ID detection

## Customization

### Add New Scenarios

1. Create perturbation module in `perturb/your_scenario.py`:
```python
def apply(ctx: Dict, level: float) -> None:
    """Apply your perturbation."""
    ctx.setdefault("noise", {})["your_scenario"] = float(level)
    # ... your perturbation logic
```

2. Add to `configs/perturbations.yaml`:
```yaml
scenarios:
  - name: "your_scenario"
    levels: [0.0, 0.2, 0.4, 0.6, 0.8]
    description: "Your scenario description"
    runs_per_level: 100
```

3. Run experiments:
```bash
python3 scripts/run_publication_experiments.py \
  --scenarios your_scenario \
  ...
```

### Modify 3D Models

Edit `geometry/models_3d.py` to customize:
- Human proportions and pose
- Vehicle dimensions and shape
- Traffic sign types and sizes

## Performance

Approximate run times (on standard workstation):

| Configuration | Time | Total Runs |
|--------------|------|------------|
| Quick test (all 5 scenarios) | ~10 min | 250 |
| Full suite (all 5 scenarios) | ~60 min | 2,500 |
| Single scenario (100 runs/level) | ~12 min | 500 |

## Citation

If you use this experiment suite in your research, please cite:

```bibtex
@software{xai_experiment_suite,
  title = {XAI Publication-Ready Experiment Suite},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/xai}
}
```

## License

[Your License Here]

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Troubleshooting

### "No module named 'numpy'"
```bash
pip install numpy scipy pandas matplotlib
```

### "Validation FAILED"
Check the comprehensive analysis report for specific warnings/errors:
```bash
cat results/publication/reports/comprehensive_analysis.json
```

### Low success rates across all scenarios
This is expected! The scenarios are designed to stress-test the system.
Check individual module performance to identify bottlenecks.

## Support

For questions or issues:
- Open an issue on GitHub
- Review the comprehensive analysis report
- Check dashboard visualizations for insights

---

**Status**: ✅ All 5 scenarios implemented with comprehensive analysis
