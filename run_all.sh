#!/usr/bin/env bash
#
# R2: Complete Research Automation Pipeline
# One-command execution: bash run_all.sh
#
# This script runs the complete automated research pipeline:
#   1. Environment validation & tests
#   2. Data generation (synthetic)
#   3. All perturbation sweeps (occlusion, lighting, motion blur, jitter, overlap)
#   4. Individual scenario reports & visualizations
#   5. Final comprehensive research report with discussion
#

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  R2: Complete Research Automation Pipeline                        ║"
echo "║  Layered Failure Attribution & Explainability for Robotics         ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Parse arguments (simple defaults or custom)
RUNS="${1:-3}"
RESULTS_DIR="${2:-results}"

echo "Configuration:"
echo "  • Runs per level:   $RUNS"
echo "  • Results directory: $RESULTS_DIR"
echo ""
echo "Starting complete automation pipeline..."
echo ""

# Run the master automation script
python scripts/run_all.py \
    --cfg configs/robosuite_grasp.yaml \
    --thresholds configs/thresholds.yaml \
    --perturbations configs/perturbations.yaml \
    --runs "$RUNS" \
    --results_dir "$RESULTS_DIR"

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║  ✓ SUCCESS! Complete pipeline finished                            ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📊 View your results:"
    echo ""
    echo "  1. Final Research Report (START HERE!):"
    echo "     → $RESULTS_DIR/final_report/research_report.html"
    echo ""
    echo "  2. Individual Scenario Dashboards:"
    echo "     → $RESULTS_DIR/reports/<scenario>/dashboard.html"
    echo ""
    echo "  3. Visualizations (PNG plots):"
    echo "     → $RESULTS_DIR/reports/<scenario>/*.png"
    echo ""
    echo "  4. Raw Data (CSV):"
    echo "     → $RESULTS_DIR/*_sweep.csv"
    echo ""
    echo "  5. Artifacts (images, masks, paths):"
    echo "     → $RESULTS_DIR/artifacts/"
    echo ""
    echo "Open the final report in your browser:"
    echo "  → file://$(pwd)/$RESULTS_DIR/final_report/research_report.html"
    echo ""
else
    echo ""
    echo "✗ Pipeline failed. Check errors above."
    exit 1
fi
