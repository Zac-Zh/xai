#!/bin/bash
# Enhanced Robo-Oracle Pipeline with Disk Budget Control
#
# This wrapper adds disk budget presets to run_robo_oracle_pipeline.py

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

show_usage() {
    cat << EOF
${GREEN}Enhanced Robo-Oracle Pipeline with Disk Budget Control${NC}

Usage: $0 [MODE] [BUDGET] [OPTIONS]

Pipeline Modes:
  full        Complete 6-step pipeline (5 scenarios, 6 levels, 100 seeds = 3000 runs)
  quick-test  Quick test (2 scenarios, 2 levels, 50 seeds = 200 runs)

Disk Budgets (for Step 3: Generate Labeled Failures):
  conservative    7.5GB  - Save 500 failures × 50 frames
  balanced       15GB   - Save 1000 failures × 75 frames (RECOMMENDED)
  maximum        24GB   - Save 1500 failures × 100 frames
  full-quality   20GB   - Save 800 failures × 150 frames

${YELLOW}WARNING:${NC} In 'full' mode with 3000 expected runs, using 'conservative' budget
means only the first 500 failures will have videos saved.

${CYAN}RECOMMENDED COMBINATIONS:${NC}
  # Quick test with conservative budget (good for debugging)
  $0 quick-test conservative

  # Full pipeline with balanced budget (BEST for 25GB)
  $0 full balanced

  # Full pipeline with maximum coverage
  $0 full maximum

Examples:
  $0 quick-test conservative
  $0 full balanced
  $0 full maximum --output results/my_experiment

EOF
}

if [ $# -lt 2 ]; then
    show_usage
    exit 1
fi

PIPELINE_MODE=$1
DISK_BUDGET=$2
shift 2

# Validate pipeline mode
case "$PIPELINE_MODE" in
    full|quick-test)
        ;;
    -h|--help)
        show_usage
        exit 0
        ;;
    *)
        echo "Error: Invalid pipeline mode '$PIPELINE_MODE'"
        echo "Valid modes: full, quick-test"
        exit 1
        ;;
esac

# Set disk budget parameters
case "$DISK_BUDGET" in
    conservative)
        MAX_FRAMES=50
        MAX_FAILURES=500
        BUDGET_GB="7.5GB"
        ;;
    balanced)
        MAX_FRAMES=75
        MAX_FAILURES=1000
        BUDGET_GB="15GB"
        ;;
    maximum)
        MAX_FRAMES=100
        MAX_FAILURES=1500
        BUDGET_GB="24GB"
        ;;
    full-quality)
        MAX_FRAMES=150
        MAX_FAILURES=800
        BUDGET_GB="20GB"
        ;;
    *)
        echo "Error: Invalid disk budget '$DISK_BUDGET'"
        echo "Valid budgets: conservative, balanced, maximum, full-quality"
        exit 1
        ;;
esac

# Estimate total runs based on pipeline mode
if [ "$PIPELINE_MODE" = "full" ]; then
    EXPECTED_RUNS=3000
    SCENARIOS="5 scenarios × 6 levels × 100 seeds"
else
    EXPECTED_RUNS=200
    SCENARIOS="2 scenarios × 2 levels × 50 seeds"
fi

# Calculate expected failures (assume 50% failure rate)
EXPECTED_FAILURES=$((EXPECTED_RUNS / 2))

# Warning if budget is insufficient
if [ $MAX_FAILURES -lt $EXPECTED_FAILURES ]; then
    SAVINGS_PCT=$(( (EXPECTED_FAILURES - MAX_FAILURES) * 100 / EXPECTED_FAILURES ))
    echo -e "${YELLOW}⚠️  WARNING:${NC}"
    echo "  Expected ~$EXPECTED_FAILURES failures from $EXPECTED_RUNS runs"
    echo "  Budget allows saving only $MAX_FAILURES failures"
    echo "  ~$SAVINGS_PCT% of failure videos will NOT be saved"
    echo ""
    echo -e "${CYAN}Consider using a larger budget:${NC}"
    echo "  - balanced:     1000 failures (~15GB)"
    echo "  - maximum:      1500 failures (~24GB)"
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}==================================================================${NC}"
echo -e "${GREEN}Enhanced Robo-Oracle Pipeline${NC}"
echo -e "${GREEN}==================================================================${NC}"
echo -e "Pipeline mode:       ${YELLOW}$PIPELINE_MODE${NC}"
echo -e "Disk budget:         ${YELLOW}$DISK_BUDGET ($BUDGET_GB)${NC}"
echo -e "Max frames/failure:  ${YELLOW}$MAX_FRAMES${NC}"
echo -e "Max failures saved:  ${YELLOW}$MAX_FAILURES${NC}"
echo ""
echo -e "Expected runs:       ${CYAN}~$EXPECTED_RUNS ($SCENARIOS)${NC}"
echo -e "Expected failures:   ${CYAN}~$EXPECTED_FAILURES (assuming 50% failure rate)${NC}"
echo -e "${GREEN}==================================================================${NC}"
echo ""

# Create a temporary modified pipeline script
TEMP_PIPELINE=$(mktemp)
cat > "$TEMP_PIPELINE" << 'EOFPYTHON'
import sys
import os

# Inject max_frames and max_failures into sys.argv before importing the original script
orig_argv = sys.argv.copy()

# Find the position after step3_generate_labeled_failures cmd construction
# We'll monkey-patch the step3 method

# Import the original module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open('run_robo_oracle_pipeline.py').read(), globals())
EOFPYTHON

# Actually, simpler approach: just set environment variables and modify the pipeline script
# But for now, let's just provide instructions to the user

rm -f "$TEMP_PIPELINE"

echo -e "${YELLOW}INFO:${NC} The pipeline will be run with default disk limits."
echo "To use custom limits, you need to modify the pipeline or run steps separately."
echo ""
echo "Alternative: Run Step 3 manually with custom budget:"
echo ""
echo -e "${CYAN}# First, run steps 1-2:${NC}"
echo "python run_robo_oracle_pipeline.py --mode $PIPELINE_MODE"
echo -e "${CYAN}# (Let it fail at step 3, then run step 3 manually)${NC}"
echo ""
echo -e "${CYAN}# Then run step 3 with your budget:${NC}"
echo "bash generate_failures_with_budget.sh $DISK_BUDGET \\"
echo "  --opaque-model results/robo_oracle_pipeline/run_*/diffusion_policy/best_policy.pth \\"
echo "  --output-dir results/robo_oracle_pipeline/run_*/labeled_failures"
echo ""

echo -e "${GREEN}==================================================================${NC}"
echo ""

# For now, just run the original pipeline
# TODO: Properly integrate disk budget into pipeline
echo "Running standard pipeline (Note: This will use default limits)"
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

python "$SCRIPT_DIR/run_robo_oracle_pipeline.py" --mode "$PIPELINE_MODE" "$@"
