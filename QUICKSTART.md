# Robo-Oracle Quick Start Guide

This guide will help you get started with Robo-Oracle in under 30 minutes.

## Prerequisites

- Python 3.10+
- 8GB+ RAM
- GPU recommended (but not required for testing)

## Installation

```bash
# Clone repository (if not already done)
cd /path/to/xai

# Install dependencies
pip install -r requirements.txt
```

## Option 1: Quick Test (Recommended for First-Time Users)

Run a minimal pipeline with small datasets to verify everything works:

```bash
python run_robo_oracle_pipeline.py --mode quick-test
```

This will:
- Generate 100 expert demonstrations (2 scenarios × 2 levels × 50 seeds)
- Train a Diffusion Policy for 20 epochs
- Generate ~50-100 labeled failures
- Train a diagnostic VLM for 20 epochs
- Evaluate the model

**Time**: ~30-60 minutes on a laptop with GPU

**Output**: `results/robo_oracle_pipeline/run_YYYYMMDD_HHMMSS/`

---

## Option 2: Full Pipeline (For Production Results)

Run the complete pipeline with full-scale datasets:

```bash
python run_robo_oracle_pipeline.py --mode full
```

This will:
- Generate 900 expert demonstrations
- Train a Diffusion Policy for 100 epochs
- Generate ~800-1200 labeled failures
- Train a diagnostic VLM for 50 epochs
- Evaluate the model

**Time**: 8-12 hours on a GPU

**Output**: `results/robo_oracle_pipeline/run_YYYYMMDD_HHMMSS/`

---

## Option 3: Step-by-Step Manual Execution

If you prefer to run each step individually:

### Step 1: Generate Expert Demonstrations

```bash
python oracle/generate_demonstrations.py \
  --scenarios occlusion lighting \
  --levels 0.0 0.1 \
  --num-seeds 50 \
  --output-dir results/demos
```

### Step 2: Train Diffusion Policy

```bash
python opaque/train_diffusion_policy.py \
  --demonstrations results/demos/expert_demonstrations.json \
  --output-dir results/diffusion \
  --epochs 50
```

### Step 3: Generate Labeled Failures

```bash
python robo_oracle/generate_labeled_failures.py \
  --opaque-model results/diffusion/best_policy.pth \
  --scenarios occlusion lighting \
  --levels 0.5 0.7 \
  --num-seeds 50 \
  --output-dir results/failures
```

### Step 4: Prepare VLM Dataset

```bash
python diagnostic/prepare_vlm_dataset.py \
  --labeled-failures results/failures/labeled_failures.json \
  --output-dir results/vlm_data
```

### Step 5: Train Diagnostic VLM

```bash
python diagnostic/train_diagnostic_vlm.py \
  --train-dataset results/vlm_data/train_vlm_dataset.json \
  --val-dataset results/vlm_data/val_vlm_dataset.json \
  --output-dir results/vlm
```

### Step 6: Evaluate

```bash
python diagnostic/evaluate.py \
  --model results/vlm/best_diagnostic_vlm.pth \
  --test-dataset results/vlm_data/val_vlm_dataset.json \
  --output-dir results/eval
```

---

## Verifying Results

After running the pipeline, check these key files:

1. **Expert Demonstrations**:
   ```bash
   python -c "import json; d=json.load(open('results/robo_oracle_pipeline/run_*/expert_demos/expert_demonstrations.json')); print(f\"Demonstrations: {len(d['demonstrations'])}\")"
   ```

2. **Labeled Failures**:
   ```bash
   python -c "import json; d=json.load(open('results/robo_oracle_pipeline/run_*/labeled_failures/labeled_failures.json')); print(f\"Failures: {len(d['failures'])}\")"
   ```

3. **Evaluation Results**:
   ```bash
   cat results/robo_oracle_pipeline/run_*/evaluation/evaluation_results.json
   ```

---

## Expected Outputs

### Quick Test Mode

- **Expert Demonstrations**: ~50-80 successful trajectories
- **Labeled Failures**: ~30-70 failures with causal labels
- **Diagnostic Accuracy**: 55-70% (limited data)

### Full Mode

- **Expert Demonstrations**: ~700-850 successful trajectories
- **Labeled Failures**: ~800-1200 failures with causal labels
- **Diagnostic Accuracy**: 70-85%

---

## Testing the Diagnostic Model

Once trained, you can test the diagnostic model on individual failures:

```python
from diagnostic.diagnostic_interface import DiagnosticInterface
import numpy as np

# Load model
diagnostic = DiagnosticInterface(
    model_checkpoint="results/robo_oracle_pipeline/run_*/diagnostic_vlm/best_diagnostic_vlm.pth"
)

# Load a failure video (frames)
frame_paths = [
    "results/robo_oracle_pipeline/run_*/labeled_failures/failure_frames/occlusion_0.5_0/frame_0000.npy",
    "results/robo_oracle_pipeline/run_*/labeled_failures/failure_frames/occlusion_0.5_0/frame_0001.npy",
    # ... more frames
]

# Diagnose
result = diagnostic.diagnose_from_paths(frame_paths)

print(f"Predicted Module: {result['predicted_module']}")
print(f"Confidence: {result['confidence']*100:.1f}%")
print(f"Explanation: {result['explanation']}")
```

---

## Troubleshooting

### Error: "PyTorch not available"

```bash
pip install torch torchvision
```

### Error: "Out of memory"

Reduce batch sizes:
```bash
python run_robo_oracle_pipeline.py --mode quick-test  # Uses smaller batches
```

Or edit the batch sizes in the pipeline script.

### Error: "No failures generated"

The opaque policy is too good! Increase perturbation levels:
```bash
# Edit run_robo_oracle_pipeline.py
# Change failure_levels to [0.7, 0.8, 0.9]
```

### Low Diagnostic Accuracy

- Ensure balanced dataset (check failure category distribution)
- Train for more epochs
- Collect more data

---

## Next Steps

1. **Read the Full Documentation**: See `ROBO_ORACLE_README.md` for details
2. **Analyze Results**: Check evaluation visualizations in `results/.../evaluation/`
3. **Customize**: Modify scenarios, perturbations, and architectures
4. **Deploy**: Use the diagnostic model in your own robotic systems

---

## Support

For issues or questions:
- Check `ROBO_ORACLE_README.md` for detailed documentation
- Open an issue on GitHub
- Contact: [Your Email]

---

Happy diagnosing! 🤖
