# Full Mode Disk Space Impact Analysis

## What is `--mode full`?

`--mode full` runs the **complete 6-step Robo-Oracle pipeline**:

```bash
python run_robo_oracle_pipeline.py --mode full
```

### The 6 Steps

1. Generate Expert Demonstrations (Classical Oracle)
2. Train Diffusion Policy (learns from demos)
3. **Generate Labeled Failures** ← DISK SPACE INTENSIVE
4. Prepare VLM Training Dataset
5. Train Diagnostic VLM
6. Evaluate Diagnostic Model

## Step 3: Disk Space Analysis

### Full Mode Configuration

From `run_robo_oracle_pipeline.py` lines 61-73:

```python
# Full mode parameters
failure_scenarios = 5    # occlusion, lighting, motion_blur, overlap, camera_jitter
failure_levels = 6       # 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
failure_seeds = 100      # 100 random seeds per config
```

**Total runs**: 5 × 6 × 100 = **3,000 experiments**

Assuming **50% failure rate** → **~1,500 failures expected**

### Current Behavior (With My Fix)

The pipeline currently calls:

```bash
python robo_oracle/generate_labeled_failures.py \
  --scenarios occlusion lighting motion_blur overlap camera_jitter \
  --levels 0.3 0.4 0.5 0.6 0.7 0.8 \
  --num-seeds 100
  # NOTE: No --max-frames or --max-failures specified!
```

**Uses defaults**:
- `--max-frames 50` (default from my fix)
- `--max-failures 500` (default from my fix)

### Impact Table

| Scenario | Expected Failures | Videos Saved | Videos Skipped | Disk Usage |
|----------|------------------|--------------|----------------|------------|
| **Current (default)** | ~1500 | 500 | 1000 (67%) | **3.75GB** ✅ |
| With `balanced` | ~1500 | 1000 | 500 (33%) | **11.25GB** ✅ |
| With `maximum` | ~1500 | 1500 | 0 (0%) | **22.5GB** ✅ |

## Recommendations for Full Mode

### Option 1: Modify Pipeline to Use Balanced Budget (RECOMMENDED)

Edit `run_robo_oracle_pipeline.py` line 154-163:

```python
# OLD (current)
cmd = [
    "python", "robo_oracle/generate_labeled_failures.py",
    "--opaque-model", model_path,
    # ... other args ...
]

# NEW (recommended for 25GB budget)
cmd = [
    "python", "robo_oracle/generate_labeled_failures.py",
    "--opaque-model", model_path,
    "--max-frames", "75",        # Add this
    "--max-failures", "1000",    # Add this
    # ... other args ...
]
```

**Result**: ~11.25GB, saves 1000/1500 failures (67% coverage)

### Option 2: Run Pipeline Steps Manually

Instead of using `run_robo_oracle_pipeline.py`, run each step yourself:

```bash
# Steps 1-2: Generate demos and train policy
python run_robo_oracle_pipeline.py --mode full
# (Ctrl+C when it reaches Step 3)

# Step 3: Run with custom budget
bash generate_failures_with_budget.sh balanced \
  --opaque-model results/robo_oracle_pipeline/run_*/diffusion_policy/best_policy.pth \
  --scenarios occlusion lighting motion_blur overlap camera_jitter \
  --levels 0.3 0.4 0.5 0.6 0.7 0.8 \
  --num-seeds 100 \
  --output-dir results/robo_oracle_pipeline/run_*/labeled_failures

# Steps 4-6: Continue with VLM training and evaluation
# (You'll need to run these manually too)
```

### Option 3: Accept the Defaults

If you're okay with:
- Saving only 500 out of ~1500 failures
- Using only 3.75GB

Then just run:

```bash
python run_robo_oracle_pipeline.py --mode full
```

The first 500 failures will have full video data, the rest will only have labels (no videos).

## What Gets Saved When Video Limit is Reached?

When the limit (e.g., 500 failures) is reached:

✅ **Still saved**:
- Failure labels from Oracle (JSON)
- Metadata (scenario, perturbation, seed)
- Performance metrics
- All diagnostic information

❌ **Not saved**:
- Video frames (.npz files)

**Impact**: VLM training quality may be reduced with fewer video examples, but you still have all the failure labels.

## Quick Reference

| Your Goal | Recommendation | Command |
|-----------|---------------|---------|
| Just testing | Use default | `python run_robo_oracle_pipeline.py --mode full` |
| Best for 25GB | Edit pipeline for balanced | Modify line 154-163 to add `--max-frames 75 --max-failures 1000` |
| Maximum coverage | Edit pipeline for maximum | Modify line 154-163 to add `--max-frames 100 --max-failures 1500` |
| Quick test | Use quick-test mode | `python run_robo_oracle_pipeline.py --mode quick-test` |

## Summary

**In `--mode full`:**
- 🔴 3000 experiments will run
- 🟡 ~1500 failures expected (50% rate)
- 🟢 Only 500 failures saved by default (3.75GB)
- ✅ For 25GB budget: Edit pipeline to use `--max-failures 1000 --max-frames 75` (11.25GB)
