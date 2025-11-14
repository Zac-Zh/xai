# Server Setup Guide - R2 Project

Complete guide for running R2 on a new server from scratch.

## 📍 Repository Information

- **GitHub**: https://github.com/Zac-Zh/xai
- **Branch**: `claude/complete-automation-pipelines-01QfZQ82dnAMFuCZbmtxbfA1`
- **Clone Command**: `git clone https://github.com/Zac-Zh/xai.git`

---

## ⚡ **FASTEST START** (Copy-Paste Ready)

```bash
# One-liner for new server (Linux/Mac with bash)
git clone https://github.com/Zac-Zh/xai.git && cd xai && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && bash run_all.sh fast
```

Or step-by-step:
```bash
git clone https://github.com/Zac-Zh/xai.git
cd xai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
bash run_all.sh fast
```

Results will be in: `results/final_report/research_report.html`

---

## 📋 Prerequisites

### System Requirements

**Minimum (Fast Mode - 2D only)**:
- OS: Linux, macOS, or Windows with WSL2
- Python: 3.10 or newer
- RAM: 4GB
- Disk: 2GB free space
- Time: 10-15 minutes for setup

**Recommended (Full Mode - with real models)**:
- OS: Linux (Ubuntu 20.04+ recommended)
- Python: 3.10+
- RAM: 8GB (16GB for GPU inference)
- GPU: CUDA-capable GPU (optional, speeds up YOLO/Mask R-CNN)
- Disk: 10GB free space (for model weights)
- Time: 30-45 minutes for setup

---

## 🚀 Quick Start (Step-by-Step)

### Step 1: Get the Code

**Option A: Clone from GitHub**
```bash
# Clone the repository
git clone https://github.com/Zac-Zh/xai.git
cd xai
```

**Option B: If you already have the code**
```bash
# Navigate to existing directory
cd /path/to/xai

# Pull latest changes
git pull origin claude/complete-automation-pipelines-01QfZQ82dnAMFuCZbmtxbfA1
```

**Option C: Download as ZIP**
```bash
# Download from GitHub and extract
wget https://github.com/Zac-Zh/xai/archive/refs/heads/claude/complete-automation-pipelines-01QfZQ82dnAMFuCZbmtxbfA1.zip
unzip complete-automation-pipelines-01QfZQ82dnAMFuCZbmtxbfA1.zip
cd xai-*
```

### Step 2: Check Python Version

```bash
# Check Python version (must be 3.10+)
python --version
# or
python3 --version

# If Python < 3.10, install newer version:
# Ubuntu/Debian:
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev

# macOS (with Homebrew):
brew install python@3.10
```

### Step 3: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3.10 -m venv venv

# Activate it
# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# Verify activation (should show venv path)
which python
```

### Step 4: Install Dependencies

#### Option A: Core Dependencies Only (Fast Mode)

```bash
# Install minimal dependencies for 2D experiments
pip install --upgrade pip
pip install numpy pandas matplotlib plotly pyyaml pillow
```

#### Option B: All Dependencies (Full Mode with Real Models)

```bash
# Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt

# This installs:
# - numpy, pandas, matplotlib, plotly (core)
# - opencv-python (computer vision)
# - torch, torchvision (deep learning)
# - ultralytics (YOLOv8)
# - segment-anything (SAM)
# - pybullet (3D simulation)
# - scikit-learn, networkx (planning)
```

**Note**: Full installation may take 5-15 minutes depending on connection speed.

#### Verify Installation

```bash
# Check core packages
python -c "import numpy; import pandas; import matplotlib; import plotly; print('✓ Core packages OK')"

# Check deep learning packages (if installed)
python -c "import torch; import ultralytics; import pybullet; print('✓ All packages OK')"
```

### Step 5: Validate Code Structure

```bash
# Run validation script (checks structure without running experiments)
python validate_code.py

# Expected output:
# ✓ All core files present
# ✓ All __init__.py files present
# ✓ All unified interface functions present
# ✅ ALL CHECKS PASSED!
```

### Step 6: Run First Experiment (Fast Mode)

```bash
# Run fast mode (2D synthetic, ~5-10 minutes)
bash run_all.sh fast

# This will:
# 1. Validate environment
# 2. Run tests
# 3. Execute experiments for all 5 perturbation scenarios
# 4. Generate visualizations and reports
```

**Expected Output**:
```
╔════════════════════════════════════════════════════════════════════╗
║  R2: COMPLETE Research Automation Pipeline                        ║
║  Publication-Ready: 2D+3D, Real Models, Multi-Task                ║
╚════════════════════════════════════════════════════════════════════╝

Configuration:
  • Mode:              fast
  • Runs per level:    3
  • Results directory: results

...

✓ SUCCESS! Complete pipeline finished
```

### Step 7: View Results

```bash
# Results are in:
ls results/

# Open the final report (in browser)
# Linux:
xdg-open results/final_report/research_report.html

# macOS:
open results/final_report/research_report.html

# Or manually navigate to:
file:///full/path/to/xai/results/final_report/research_report.html
```

---

## 🔧 Detailed Installation Options

### GPU Support (Optional but Recommended for Real Models)

If you have an NVIDIA GPU and want faster inference:

```bash
# Check if CUDA is available
nvidia-smi

# Install PyTorch with CUDA support
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Specific Package Versions (If Issues Occur)

```bash
# If you encounter compatibility issues, use exact versions:
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install matplotlib==3.7.2
pip install plotly==5.14.1
pip install opencv-python==4.8.0.76
pip install torch==2.0.1
pip install torchvision==0.15.2
pip install ultralytics==8.0.196
pip install pybullet==3.2.5
pip install scikit-learn==1.3.0
pip install pyyaml==6.0
```

---

## 📊 Running Different Modes

### Fast Mode (Testing - 5-10 minutes)
```bash
bash run_all.sh fast
```
- Uses 2D synthetic environment
- Stub implementations (no deep learning)
- 3 runs per perturbation level
- All 5 scenarios: occlusion, lighting, motion_blur, camera_jitter, overlap

### Full Mode (Comprehensive - 30-60 minutes)
```bash
bash run_all.sh full
```
- Same as fast mode but with comprehensive analysis
- More detailed reporting
- Same 2D synthetic experiments

### Custom Configuration
```bash
# Custom number of runs
bash run_all.sh fast 10  # 10 runs per level instead of 3

# Custom output directory
bash run_all.sh fast 3 my_results

# Advanced: Python script directly
python scripts/run_all.py \
    --cfg configs/robosuite_grasp.yaml \
    --thresholds configs/thresholds.yaml \
    --perturbations configs/perturbations.yaml \
    --runs 5 \
    --results_dir custom_output
```

---

## 🧪 Running with Real Models (Advanced)

Once you have all dependencies installed:

```bash
# Run unified experiments with real models
python scripts/run_unified_experiments.py \
    --mode 2d \
    --use_real_models \
    --task Lift \
    --scenario occlusion \
    --levels 0.0 0.2 0.4 0.6 \
    --runs 3 \
    --thresholds configs/thresholds.yaml \
    --output results/real_models_test.jsonl

# This will use:
# - YOLOv8 for object detection (downloads ~6MB model first run)
# - Mask R-CNN for segmentation (downloads ~170MB model first run)
# - OpenCV PnP for pose estimation
# - Real RRT* for planning
# - Physics-based controller
```

**Note**: First run downloads model weights automatically (internet required).

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'numpy'"

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "python: command not found"

**Solution**:
```bash
# Try python3 instead
python3 --version

# Or create alias
alias python=python3
```

### Issue: "Permission denied: ./run_all.sh"

**Solution**:
```bash
# Make script executable
chmod +x run_all.sh

# Or run with bash directly
bash run_all.sh fast
```

### Issue: PyTorch GPU not detected

**Solution**:
```bash
# Check CUDA version
nvidia-smi

# Reinstall PyTorch with correct CUDA version
# See: https://pytorch.org/get-started/locally/

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: "Out of memory" during experiments

**Solution**:
```bash
# Reduce number of runs
bash run_all.sh fast 1  # Use 1 run instead of 3

# Or use CPU only (slower but less memory)
export CUDA_VISIBLE_DEVICES=""  # Disable GPU
bash run_all.sh fast
```

### Issue: ultralytics/YOLO model download fails

**Solution**:
```bash
# Manually download YOLOv8 model
mkdir -p ~/.cache/torch/hub/checkpoints/
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt \
     -O ~/.cache/torch/hub/checkpoints/yolov8n.pt

# Or use offline mode in code
# (model will use cached version)
```

### Issue: Mask R-CNN download fails

**Solution**:
```bash
# Models are downloaded via torchvision
# Manually download if needed:
python -c "from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights; maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)"
```

---

## 📈 Understanding Results

After running experiments, you'll have:

```
results/
├── final_report/
│   └── research_report.html     # ← Open this first!
├── reports/
│   ├── occlusion/
│   │   ├── dashboard.html       # Interactive Plotly dashboard
│   │   ├── stacked.png          # Module failure bars
│   │   ├── sensitivity.png      # Degradation curve
│   │   └── sankey.png           # Failure flow
│   ├── lighting/
│   ├── motion_blur/
│   ├── camera_jitter/
│   └── overlap/
├── occlusion_sweep.csv          # Raw data (easy to analyze)
├── lighting_sweep.csv
├── motion_blur_sweep.csv
├── camera_jitter_sweep.csv
├── overlap_sweep.csv
├── logs/                        # JSONL logs per run
└── artifacts/                   # Images, masks, plots
```

**Key files**:
1. **`research_report.html`** - Start here! Comprehensive analysis
2. **`*_sweep.csv`** - Load into pandas/Excel for custom analysis
3. **`dashboard.html`** - Interactive exploration

---

## 🔒 Security Considerations

### On Shared Servers

```bash
# Use user-specific directories
mkdir -p ~/xai_results
bash run_all.sh fast 3 ~/xai_results

# Set restrictive permissions
chmod 700 ~/xai_results
```

### Docker/Container Setup (Optional)

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["bash", "run_all.sh", "fast"]
```

```bash
# Build and run
docker build -t r2-xai .
docker run -v $(pwd)/results:/app/results r2-xai
```

---

## 🌐 Remote Server Access

### SSH Setup

```bash
# On your local machine
ssh user@remote-server

# On remote server
cd /path/to/xai
source venv/bin/activate
bash run_all.sh fast

# View results remotely via SSH tunnel
# On local machine:
ssh -L 8080:localhost:8080 user@remote-server

# On remote server:
cd xai/results/final_report
python -m http.server 8080

# On local machine browser:
# Open: http://localhost:8080/research_report.html
```

### Headless Server (No GUI)

```bash
# All experiments run fine without GUI
bash run_all.sh fast

# Transfer results to local machine
# On local machine:
scp -r user@remote-server:/path/to/xai/results ./

# Then open locally
open results/final_report/research_report.html
```

---

## ⏱️ Estimated Times

| Mode | Setup Time | Run Time | Total |
|------|------------|----------|-------|
| **Fast** (core deps only) | 5 min | 5-10 min | ~15 min |
| **Fast** (all deps) | 15 min | 5-10 min | ~25 min |
| **Full** (comprehensive) | 15 min | 30-60 min | ~1 hour |
| **With GPU** (real models) | 20 min | varies | ~1-2 hours |

---

## ✅ Verification Checklist

Before running experiments, verify:

- [ ] Python 3.10+ installed: `python --version`
- [ ] Dependencies installed: `pip list | grep numpy`
- [ ] Virtual environment activated: `which python`
- [ ] Code validation passes: `python validate_code.py`
- [ ] Permissions correct: `ls -la run_all.sh`
- [ ] Enough disk space: `df -h .`
- [ ] (Optional) GPU detected: `nvidia-smi`

Then run:
```bash
bash run_all.sh fast
```

---

## 📞 Getting Help

If you encounter issues:

1. **Check validation**: `python validate_code.py`
2. **Check dependencies**: `pip list`
3. **Check Python version**: `python --version`
4. **Check logs**: Look at error messages in terminal
5. **Check GitHub issues**: [github.com/YOUR_REPO/issues](https://github.com)

---

## 🎓 Next Steps

After successful setup:

1. **Analyze results**: Open `results/final_report/research_report.html`
2. **Explore data**: Load CSV files into pandas
3. **Modify experiments**: Edit `configs/*.yaml`
4. **Add features**: See `PUBLICATION_READY.md`
5. **Write paper**: Use generated figures and tables

---

## 📚 Additional Documentation

- **[README.md](README.md)** - Project overview
- **[AUTOMATION_GUIDE.md](AUTOMATION_GUIDE.md)** - Detailed usage
- **[PUBLICATION_READY.md](PUBLICATION_READY.md)** - Publication guide
- **[VALIDATION_FIXES.md](VALIDATION_FIXES.md)** - Code validation details

---

**Congratulations! You're ready to run R2 on your server!** 🎉

For most users, this is all you need:
```bash
pip install -r requirements.txt
bash run_all.sh fast
open results/final_report/research_report.html
```

---

## 🔗 Quick Reference Card

**Repository**: https://github.com/Zac-Zh/xai  
**Branch**: `claude/complete-automation-pipelines-01QfZQ82dnAMFuCZbmtxbfA1`

**Clone & Run**:
```bash
git clone https://github.com/Zac-Zh/xai.git
cd xai
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
bash run_all.sh fast
```

**View Results**: `results/final_report/research_report.html`

**Documentation**:
- [README.md](README.md) - Project overview
- [SETUP_GUIDE.md](SETUP_GUIDE.md) - This file
- [AUTOMATION_GUIDE.md](AUTOMATION_GUIDE.md) - Usage guide  
- [PUBLICATION_READY.md](PUBLICATION_READY.md) - Publication guide

**Support**: Check GitHub Issues or validate locally with `python validate_code.py`
