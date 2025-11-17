# Robo-Oracle: Verifiable Causal Failure Attribution for Opaque Visuomotor Policies

## Overview

**Robo-Oracle** is a novel methodology for self-supervised causal failure attribution in end-to-end robotic policies. It solves a critical bottleneck in modern robotics research: **diagnosing why opaque, black-box policies fail**.

### The Problem

Modern robotic policies (e.g., Diffusion Policies, robot foundation models) are opaque end-to-end (E2E) neural networks. When they fail, it's a mystery:
- Was it a perception error?
- A planning failure?
- Poor generalization?

Traditional diagnostic methods provide only correlational labels or require expensive human annotation.

### Our Solution

Robo-Oracle uses a **deterministic, white-box classical pipeline** (Vision → Geometry → Planning → Control) as a ground-truth "Oracle" to **causally label** the failures of opaque policies.

**Key Innovation**: When an opaque policy fails on a configuration, we run the Oracle on the *exact same configuration*. The Oracle's programmatic attribution (e.g., "GEOMETRY: PNP_RMSE_VIOLATION") becomes the ground-truth label for training a diagnostic model.

This creates the world's first large-scale dataset of opaque policy failures with **causal, verifiable labels**.

## Why This Matters (Publication Impact)

### SOTA Problem Being Solved

This work directly addresses unresolved challenges highlighted at:
- **RSS 2024 Workshop**: "Robot Execution Failures and Failure Management Strategies"
- **RoboMD** (2024): Automated failure diagnosis for manipulation policies
- **AHA** (2025): Vision-Language-Model for failure detection and reasoning

### Our Advantage Over SOTA

| Aspect | RoboMD | AHA (FailGen) | Robo-Oracle (Ours) |
|--------|--------|---------------|-------------------|
| **Label Type** | Active RL exploration | Correlational (post-perturbation) | **Causal (programmatic)** |
| **Verifiability** | Black-box | Heuristic | **Deterministic** |
| **Granularity** | Failure probability | Binary detection | **Module-level attribution** |
| **Scalability** | Expensive (online RL) | Good | **Excellent (offline)** |

### Target Venues

- **CoRL** (Conference on Robot Learning) - Main track oral
- **RSS** (Robotics: Science and Systems) - Main track oral
- **ICRA** (International Conference on Robotics and Automation) - Oral
- **NeurIPS** - Datasets & Benchmarks track

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Robo-Oracle System                        │
└─────────────────────────────────────────────────────────────┘

         ┌──────────────┐
         │Configuration │
         │(scenario,    │
         │ level, seed) │
         └──────┬───────┘
                │
        ┌───────▼────────┐
        │                │
        │  OPAQUE POLICY │  ← Diffusion Policy (Module 2)
        │  (Black Box)   │     End-to-end trained
        │                │
        └───────┬────────┘
                │
         ┌──────▼───────┐
         │   Failure?   │
         └──────┬───────┘
                │ Yes
                │
        ┌───────▼────────────────────────────────┐
        │                                        │
        │  ORACLE (Classical Pipeline)           │
        │  ┌──────┐  ┌─────────┐  ┌────────┐   │
        │  │Vision│→ │Geometry │→ │Planning│   │
        │  └──────┘  └─────────┘  └────────┘   │
        │       ↓          ↓           ↓         │
        │  ┌─────────────────────────────────┐  │
        │  │  Attribution Engine             │  │
        │  │  - Module identification        │  │
        │  │  - Threshold violation          │  │
        │  │  - Cascading failure detection  │  │
        │  └─────────────────────────────────┘  │
        │                                        │
        └────────────────┬───────────────────────┘
                         │
                 ┌───────▼────────┐
                 │  CAUSAL LABEL  │
                 │  {             │
                 │   module: "GEO"│
                 │   reason: "..."│
                 │   value: 0.47  │
                 │  }             │
                 └────────────────┘
```

## Implementation Status

### ✅ Completed (Module 1)

- **Oracle Attribution Engine** (`attribution/oracle_attribution.py`)
  - Programmatic failure labels with causal attribution
  - Natural language generation for VLM training
  - Cascading failure detection
  - Severity and recoverability classification

- **Oracle Interface** (`oracle/oracle_interface.py`)
  - Clean callable API: `run_classical_oracle(scenario, level, seed)`
  - Expert demonstration generation
  - Batch execution support
  - Full trajectory collection

- **Data Generation Pipeline** (`oracle/data_generation.py`)
  - Automated dataset generation
  - (Opaque failure, Oracle label) pair creation
  - Balance analysis and statistics
  - Ready for Module 2 integration

### 🔄 In Progress (Module 2)

- **Diffusion Policy Implementation**
  - Architecture design (denoising diffusion)
  - Imitation learning from expert demos
  - Visuomotor policy training
  - Integration with data pipeline

### ⏳ Planned (Modules 3-5)

- **Module 3**: Data pipeline integration (structure complete, needs Module 2)
- **Module 4**: VLM fine-tuning for diagnostic reasoning
- **Module 5**: Evaluation and analysis framework

## Quick Start

### 1. Test the Oracle

```bash
# Test Oracle attribution on a single run
python3 -c "
from oracle.oracle_interface import run_classical_oracle

success, label, log = run_classical_oracle(
    scenario='occlusion',
    level=0.6,
    seed=42
)

print('Success:', success)
if not success:
    print('Failure Module:', label.failure_module)
    print('Failure Reason:', label.failure_reason)
    print('Description:', label.natural_language_description)
"
```

### 2. Generate Expert Demonstrations

```bash
python3 -c "
from oracle.oracle_interface import Oracle

oracle = Oracle()
dataset_path = oracle.generate_expert_demonstrations(
    num_demos=100,
    output_dir='data/expert_demonstrations'
)
print('Saved to:', dataset_path)
"
```

### 3. Generate Labeled Failure Dataset

```bash
python3 oracle/data_generation.py \
  --num-samples 1000 \
  --output-dir data/robo_oracle_failures \
  --scenarios occlusion module_failure data_corruption
```

## The Five Modules (Roadmap)

### Module 1: Oracle ✅

**Status**: Complete

**Components**:
- `attribution/oracle_attribution.py` - Causal attribution engine
- `oracle/oracle_interface.py` - Oracle API

**Outputs**:
- Programmatic failure labels
- Expert demonstration dataset
- Natural language descriptions

### Module 2: Opaque Policy 🔄

**Status**: In progress

**Goal**: Implement a Diffusion Policy as the "opaque" E2E policy

**Tasks**:
- [ ] Implement denoising diffusion architecture
- [ ] Train on expert demonstrations (from Module 1)
- [ ] Create opaque policy interface
- [ ] Integrate with data pipeline

**Reference**: Chi et al., "Diffusion Policy" (CoRL 2023)

### Module 3: Data Pipeline ✅ (Structure)

**Status**: Structure complete, awaiting Module 2

**Implementation**: `oracle/data_generation.py`

**Process**:
1. Run opaque policy on configuration
2. If fails: Run Oracle on same configuration
3. Extract causal label from Oracle
4. Save (opaque_failure, oracle_label) pair

### Module 4: Diagnostic VLM ⏳

**Status**: Planned

**Goal**: Train a VLM to diagnose opaque failures

**Approach**:
- Fine-tune open VLM (e.g., LLaVA) on our dataset
- Input: Failure trajectory video
- Output: Causal explanation matching Oracle labels

### Module 5: Evaluation ⏳

**Status**: Planned

**Metrics**:
- Classification accuracy (module prediction)
- Qualitative comparison with GPT-4V
- Ablation: What can/can't be diagnosed from video alone?

## Dataset Format

### Expert Demonstrations (Module 1 Output)

```json
{
  "demo_id": 0,
  "scenario": "occlusion",
  "level": 0.05,
  "seed": 12345,
  "trajectory": {
    "states": [...],
    "actions": [...],
    "images": [...]
  },
  "run_log": {...}
}
```

### Labeled Failures (Module 3 Output)

```json
{
  "sample_id": 0,
  "timestamp": "2025-01-17T12:00:00Z",
  "config": {
    "scenario": "module_failure",
    "level": 0.6,
    "seed": 42
  },
  "opaque_policy": {
    "success": false,
    "trajectory": {
      "rollout_video": "path/to/video.mp4",
      "states": [...]
    }
  },
  "oracle_label": {
    "failure_occurred": true,
    "failure_module": "GEOMETRY",
    "failure_reason": "PNP_RMSE_VIOLATION",
    "threshold_violated": "max_pnp_rmse",
    "threshold_value": 0.4,
    "actual_value": 0.47,
    "severity": "MODERATE",
    "is_cascading": false
  },
  "natural_language": "The task failed due to a MODERATE error in the GEOMETRY module. Specifically, the pose estimation had excessive error (high RMSE). The max_pnp_rmse threshold was 0.4, but the actual value was 0.470."
}
```

## Key Contributions (For Paper)

### 1. Methodological Novelty

**First-ever dataset of opaque policy failures with causal, verifiable labels.**

Traditional approaches:
- RoboMD: Active RL (expensive, online)
- AHA/FailGen: Correlational labels (not causal)

Robo-Oracle: Deterministic causal attribution from white-box Oracle.

### 2. Scalability

- Fully automated (no human annotation)
- Offline generation (no expensive online RL)
- Deterministic (reproducible labels)
- Scalable to millions of samples

### 3. Multi-Modal Attribution

Our labels include:
- Programmatic (JSON) for analysis
- Natural language for VLM training
- Module-level granularity
- Cascading failure detection
- Severity and recoverability

### 4. Verifiability

Every label is traceable to:
- Exact configuration (scenario, level, seed)
- Specific threshold violation
- Quantitative metrics
- Deterministic pipeline execution

This enables scientific reproducibility and debugging.

## Integration with Existing Work

The Robo-Oracle system seamlessly integrates with your existing comprehensive experiment suite:

### From Publication Suite

Your 5 scenarios provide diverse failure modes:
1. Occlusion → Perception failures
2. Module Failure → Multi-module failures
3. Data Corruption → Noisy sensor data
4. Noise Injection → Stochastic failures
5. Adversarial Patches → Targeted failures

### To Robo-Oracle

These scenarios become the perturbation source for:
- Training opaque policies
- Generating failure conditions
- Creating diverse labeled datasets
- Benchmarking diagnostic models

## Next Steps

### Immediate (This Session)

1. ✅ Complete Module 1 (Oracle)
2. 🔄 Begin Module 2 (Diffusion Policy architecture)
3. Test end-to-end pipeline with placeholder opaque policy

### Short-term (Next Phase)

1. Implement full Diffusion Policy
2. Generate large-scale labeled dataset (10k+ samples)
3. Analyze failure distribution and dataset balance

### Medium-term (Publication)

1. Implement VLM fine-tuning (Module 4)
2. Comprehensive evaluation (Module 5)
3. Write paper (target: CoRL/RSS)

## Technical Requirements

### Dependencies

```bash
pip install numpy scipy pandas matplotlib
# For Module 2 (Diffusion Policy):
pip install torch torchvision
# For Module 4 (VLM):
pip install transformers accelerate
```

### Computational Resources

- **Module 1** (Oracle): CPU only, fast
- **Module 2** (Diffusion Policy): GPU recommended (training)
- **Module 3** (Data generation): CPU, parallelizable
- **Module 4** (VLM fine-tuning): GPU required (A100 recommended)

## Citation (Planned)

```bibtex
@inproceedings{robo_oracle_2025,
  title={Robo-Oracle: Verifiable Causal Failure Attribution for Opaque Visuomotor Policies},
  author={Your Name},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2025}
}
```

## Contact & Collaboration

This is cutting-edge research at the intersection of:
- Explainable AI
- Robot learning
- Failure diagnosis
- Vision-language models

The Oracle infrastructure is complete and ready for integration with Module 2.

---

**Status**: Module 1 ✅ Complete | Ready for Diffusion Policy Implementation
