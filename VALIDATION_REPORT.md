# Code Validation Report

## Critical Issues Found

### ❌ ISSUE #1: Argument Mismatch in run_all.sh

**File**: `run_all.sh` lines 64-70
**Problem**: Calls `run_all.py` with non-existent arguments

```bash
python scripts/run_all.py \
    --cfg configs/publication_experiments.yaml \
    --thresholds configs/thresholds.yaml \
    --runs "$RUNS" \
    --results_dir "$RESULTS_DIR" \
    --mode full \              # ← DOES NOT EXIST
    --use_real_models          # ← DOES NOT EXIST
```

**run_all.py actual arguments**:
- `--cfg`
- `--thresholds`
- `--perturbations` (NOT --mode!)
- `--runs`
- `--results_dir`
- `--skip_validation`

**Fix Required**: Remove `--mode` and `--use_real_models` from bash script, or add these to run_all.py

---

### ❌ ISSUE #2: Config File Format Mismatch

**File**: `run_all.sh` line 65
**Problem**: References `configs/publication_experiments.yaml` but `run_all.py` expects different format

`run_all.py` expects format from `configs/perturbations.yaml`:
```yaml
scenarios:
  - name: "occlusion"
    levels: [0.0, 0.2, 0.4, 0.6]
```

But `configs/publication_experiments.yaml` has:
```yaml
experiments:
  - name: "2D_Synthetic"
    mode: "2d"
    use_real_models: false
```

**Fix Required**: Either update run_all.py to parse new format, or change bash script to use correct config

---

### ⚠️  ISSUE #3: run_all.py doesn't support 3D/multi-task experiments

**File**: `scripts/run_all.py`
**Problem**: Only runs perturbation sweeps using `sweep_perturb.py`, which doesn't support:
- 3D PyBullet mode
- Real model selection
- Multi-task experiments

It calls:
```python
python scripts/sweep_perturb.py --cfg ... --scenario ... --levels ...
```

But for publication mode, we need:
```python
python scripts/run_unified_experiments.py --mode 3d --use_real_models --task ...
```

**Fix Required**: Update run_all.py to call run_unified_experiments.py for 3D/real model runs

---

## Moderate Issues

### ⚠️  ISSUE #4: Missing __init__.py files (might cause import issues)

**Directories missing __init__.py**:
- `vision/` (might need for `from vision import detector_stub`)
- `geometry/`
- `planning/`
- `control/`
- `simulators/`

**Impact**: Import statements like `from vision import detector_stub` may fail

**Fix**: Add empty `__init__.py` files to all module directories

---

### ⚠️  ISSUE #5: run_unified_experiments.py not integrated into pipeline

**File**: `scripts/run_unified_experiments.py`
**Problem**: Created but never called by automation scripts

Current flow: `run_all.sh` → `run_all.py` → `sweep_perturb.py` (2D only)
Needed flow: Support calling `run_unified_experiments.py` for 3D experiments

---

## Minor Issues

### ℹ️  ISSUE #6: Inconsistent naming in unified runner

**File**: `scripts/run_unified_experiments.py` line 50
Uses: `from planning import rrt_star_fallback`
But real file is: `planning/rrt_star_real.py`

Should import both:
```python
from planning import rrt_star_fallback  # stub
from planning.rrt_star_real import RRTStarPlanner  # real
```

---

## Summary

**Critical Issues**: 3
**Moderate Issues**: 2
**Minor Issues**: 1

**Status**: ❌ **WILL NOT RUN** - Critical issues prevent execution

**Priority Fixes**:
1. Fix run_all.sh argument passing
2. Update run_all.py to support 3D/real models
3. Add __init__.py files
4. Integrate run_unified_experiments.py properly
