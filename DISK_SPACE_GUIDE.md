# Disk Space Management Guide for Robo-Oracle

## Problem Summary

The `failure_frames` directories can consume massive disk space:
- **Before fix**: 150 frames × 900KB per failure = 135MB per failure
- **With 1500+ failures**: Could exceed 200GB 💥

## Solution Implemented

Three optimizations to reduce disk usage by **90%**:

1. **Frame Limiting**: Only save essential frames (default: 50 instead of 150)
2. **Compression**: Use `.npz` format instead of individual `.npy` files (50-70% space saving)
3. **Smart Limits**: Automatic disk space monitoring and failure count limits

**Result**: ~15MB per failure (down from 135MB)

## Quick Start

### For 25GB Available Space (Recommended)

```bash
# Use the balanced preset (15GB, 1000 failures)
bash generate_failures_with_budget.sh balanced \
  --opaque-model checkpoints/diffusion_policy.pt
```

## All Available Presets

| Preset | Disk Usage | Failures | Frames/Failure | Best For |
|--------|-----------|----------|----------------|----------|
| `conservative` | 7.5GB | 500 | 50 | Quick experiments, limited space |
| **`balanced`** | **15GB** | **1000** | **75** | **Recommended default** |
| `maximum` | 24GB | 1500 | 100 | Maximum coverage |
| `full-quality` | 20GB | 800 | 150 | Detailed failure analysis |

### Examples

```bash
# Conservative mode (good for testing)
bash generate_failures_with_budget.sh conservative \
  --opaque-model checkpoints/diffusion_policy.pt \
  --num-seeds 10

# Balanced mode (RECOMMENDED for 25GB budget)
bash generate_failures_with_budget.sh balanced \
  --opaque-model checkpoints/diffusion_policy.pt

# Maximum coverage (use full 25GB)
bash generate_failures_with_budget.sh maximum \
  --opaque-model checkpoints/diffusion_policy.pt

# Full quality frames (for paper figures/videos)
bash generate_failures_with_budget.sh full-quality \
  --opaque-model checkpoints/diffusion_policy.pt
```

## Manual Configuration

If you prefer manual control:

```bash
python robo_oracle/generate_labeled_failures.py \
  --opaque-model checkpoints/diffusion_policy.pt \
  --max-frames 75 \           # Frames per failure
  --max-failures 1000 \       # Max failures to save
  --min-free-space 5.0        # Stop if < 5GB free
```

### Calculate Your Own Budget

```
Disk Usage = max_failures × max_frames × 150KB

Examples:
- 500 failures × 50 frames = 3.75GB
- 1000 failures × 75 frames = 11.25GB
- 1500 failures × 100 frames = 22.5GB
```

## Cleaning Up Existing Files

If you already have large `failure_frames` directories:

```bash
# See what would be deleted (safe)
python cleanup_failure_frames.py --dry-run

# Actually delete (requires confirmation)
python cleanup_failure_frames.py --confirm
```

## Understanding the Trade-offs

### More Frames per Failure
**Pros**: Better temporal detail, smoother videos, easier to see failure progression
**Cons**: More disk space per failure

### More Failures Saved
**Pros**: Better dataset diversity, more training data for VLM
**Cons**: More total disk space

### Recommended Balance
- **75 frames**: Captures full failure sequence without redundancy
- **1000 failures**: Enough diversity for robust VLM training
- **Total: ~15GB**: Comfortable margin under 25GB budget

## Features

### Automatic Disk Space Monitoring
The system will automatically stop saving videos if:
- Free disk space drops below 5GB (configurable via `--min-free-space`)
- Maximum failure count is reached

### What Gets Saved

Even when video saving is skipped, the system still saves:
- ✅ Failure labels from Oracle (JSON)
- ✅ Metadata (scenario, perturbation level, seed)
- ✅ Performance metrics

Only the video frames are optional.

## Compressed Storage Format

### Old Format (Unoptimized)
```
failure_frames/occlusion_0.5_42/
├── frame_0000.npy  (900KB)
├── frame_0001.npy  (900KB)
├── ...
└── frame_0149.npy  (900KB)
Total: 135MB
```

### New Format (Optimized)
```
failure_frames/occlusion_0.5_42/
└── frames.npz  (compressed, ~7.5MB for 50 frames)
Total: 7.5MB (95% reduction!)
```

### Reading Compressed Frames

```python
import numpy as np

# Load compressed frames
data = np.load('failure_frames/occlusion_0.5_42/frames.npz')

# Access individual frames
frame_0 = data['frame_0000']
frame_1 = data['frame_0001']

# Or iterate
for key in sorted(data.keys()):
    frame = data[key]
    # Process frame...
```

## CUDNN Warning Note

The CUDNN warning (`CUDNN_STATUS_NOT_SUPPORTED`) is **harmless** and does not affect:
- Correctness of results
- Ability to train models
- Quality of failure detection

It only means a minor performance optimization is unavailable on your GPU.

For more details, run: `python fix_cudnn_warnings.py`

## Summary

For a **25GB budget**, use the **balanced preset**:

```bash
bash generate_failures_with_budget.sh balanced \
  --opaque-model checkpoints/diffusion_policy.pt
```

This gives you:
- ✅ 1000 diverse failure cases (excellent for VLM training)
- ✅ 75 frames per failure (complete failure sequences)
- ✅ ~15GB disk usage (40% of your budget)
- ✅ Plenty of margin for safety
