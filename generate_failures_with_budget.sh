#!/bin/bash
# Convenience wrapper for generate_labeled_failures.py with disk budget presets

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

show_usage() {
    cat << EOF
Usage: $0 [PRESET] [OPTIONS]

Disk Budget Presets (for 25GB available space):
  conservative    7.5GB  - 500 failures × 50 frames  (quick experiments)
  balanced       15GB   - 1000 failures × 75 frames (RECOMMENDED)
  maximum        24GB   - 1500 failures × 100 frames (full coverage)
  full-quality   20GB   - 800 failures × 150 frames (detailed analysis)

Required Arguments:
  --opaque-model PATH    Path to trained Diffusion Policy checkpoint

Optional Arguments:
  --cfg PATH             Config file (default: configs/robosuite_grasp.yaml)
  --thresholds PATH      Thresholds file (default: configs/thresholds.yaml)
  --output-dir PATH      Output directory (default: results/robo_oracle_dataset)
  --scenarios SCENARIOS  Space-separated scenarios (default: occlusion lighting motion_blur)
  --levels LEVELS        Space-separated levels (default: 0.3 0.4 0.5 0.6 0.7 0.8)
  --num-seeds N          Seeds per config (default: 100)
  --no-videos            Skip saving video frames

Examples:
  # Recommended for 25GB budget
  $0 balanced --opaque-model checkpoints/diffusion_policy.pt

  # Conservative mode for testing
  $0 conservative --opaque-model checkpoints/diffusion_policy.pt --num-seeds 10

  # Maximum coverage
  $0 maximum --opaque-model checkpoints/diffusion_policy.pt

  # Custom parameters (override preset)
  $0 balanced --opaque-model checkpoints/diffusion_policy.pt --max-frames 60 --max-failures 800

EOF
}

if [ $# -lt 1 ]; then
    show_usage
    exit 1
fi

PRESET=$1
shift

# Set defaults based on preset
case "$PRESET" in
    conservative)
        MAX_FRAMES=50
        MAX_FAILURES=500
        BUDGET="7.5GB"
        ;;
    balanced)
        MAX_FRAMES=75
        MAX_FAILURES=1000
        BUDGET="15GB"
        ;;
    maximum)
        MAX_FRAMES=100
        MAX_FAILURES=1500
        BUDGET="24GB"
        ;;
    full-quality)
        MAX_FRAMES=150
        MAX_FAILURES=800
        BUDGET="20GB"
        ;;
    -h|--help)
        show_usage
        exit 0
        ;;
    *)
        echo -e "${RED}Error: Unknown preset '$PRESET'${NC}"
        echo "Valid presets: conservative, balanced, maximum, full-quality"
        echo "Run '$0 --help' for more information"
        exit 1
        ;;
esac

echo -e "${GREEN}==================================================================${NC}"
echo -e "${GREEN}Robo-Oracle Failure Dataset Generation${NC}"
echo -e "${GREEN}==================================================================${NC}"
echo -e "Preset:              ${YELLOW}$PRESET${NC}"
echo -e "Max frames/failure:  ${YELLOW}$MAX_FRAMES${NC}"
echo -e "Max failures:        ${YELLOW}$MAX_FAILURES${NC}"
echo -e "Estimated disk use:  ${YELLOW}~$BUDGET${NC}"
echo -e "${GREEN}==================================================================${NC}"
echo ""

# Run the Python script with preset parameters and any additional arguments
python "$SCRIPT_DIR/robo_oracle/generate_labeled_failures.py" \
    --max-frames "$MAX_FRAMES" \
    --max-failures "$MAX_FAILURES" \
    "$@"
