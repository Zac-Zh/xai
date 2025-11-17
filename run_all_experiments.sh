#!/bin/bash
################################################################################
# Publication-Ready XAI Experiment Suite Runner
#
# This script runs all 5 scenarios with comprehensive analysis:
# 1. Occlusion
# 2. Module Failure
# 3. Data Corruption
# 4. Noise Injection
# 5. Adversarial Patches
#
# Features:
# - 100 runs per condition (sufficient statistical power)
# - Real 3D models
# - Automated validation
# - Cross-scenario analysis
# - Module vulnerability analysis
# - Interactive HTML dashboards
#
# Usage:
#   ./run_all_experiments.sh              # Run all scenarios
#   ./run_all_experiments.sh --quick      # Quick test (10 runs per condition)
#   ./run_all_experiments.sh --help       # Show help
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="configs/robosuite_grasp.yaml"
PERTURBATIONS_FILE="configs/perturbations.yaml"
THRESHOLDS_FILE="configs/thresholds.yaml"
OUTPUT_DIR="results/publication"
QUICK_TEST=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_TEST=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick       Run quick test with reduced sample size (10 runs per condition)"
            echo "  --help        Show this help message"
            echo ""
            echo "Example:"
            echo "  $0              # Full experiment suite (100 runs per condition)"
            echo "  $0 --quick      # Quick test (10 runs per condition)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

################################################################################
# Print banner
################################################################################
echo -e "${PURPLE}"
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                                                                    ║"
echo "║        XAI Publication-Ready Experiment Suite                     ║"
echo "║        Comprehensive Robustness Analysis                          ║"
echo "║                                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

if [ "$QUICK_TEST" = true ]; then
    echo -e "${YELLOW}⚡ QUICK TEST MODE: Running with reduced sample size (10 runs)${NC}"
else
    echo -e "${GREEN}📊 FULL MODE: Running with 100 runs per condition${NC}"
fi
echo ""

################################################################################
# Check dependencies
################################################################################
echo -e "${CYAN}[1/5] Checking dependencies...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ Python 3 found${NC}"

# Check required Python packages
python3 -c "import numpy, scipy, pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}  ⚠ Installing required Python packages...${NC}"
    pip install -q numpy scipy pandas matplotlib
fi
echo -e "${GREEN}  ✓ Python packages available${NC}"

# Check config files exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}✗ Configuration file not found: $CONFIG_FILE${NC}"
    exit 1
fi
if [ ! -f "$PERTURBATIONS_FILE" ]; then
    echo -e "${RED}✗ Perturbations file not found: $PERTURBATIONS_FILE${NC}"
    exit 1
fi
if [ ! -f "$THRESHOLDS_FILE" ]; then
    echo -e "${RED}✗ Thresholds file not found: $THRESHOLDS_FILE${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ Configuration files found${NC}"

################################################################################
# Run experiments
################################################################################
echo ""
echo -e "${CYAN}[2/5] Running experiments...${NC}"
echo ""

START_TIME=$(date +%s)

QUICK_FLAG=""
if [ "$QUICK_TEST" = true ]; then
    QUICK_FLAG="--quick-test"
fi

python3 scripts/run_publication_experiments.py \
    --config "$CONFIG_FILE" \
    --perturbations "$PERTURBATIONS_FILE" \
    --thresholds "$THRESHOLDS_FILE" \
    --output "$OUTPUT_DIR" \
    $QUICK_FLAG

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Experiment execution failed${NC}"
    exit 1
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo -e "${GREEN}✓ Experiments completed in ${ELAPSED}s${NC}"

################################################################################
# Generate dashboards
################################################################################
echo ""
echo -e "${CYAN}[3/5] Generating interactive dashboards...${NC}"

python3 scripts/generate_dashboards.py \
    --logs-dir "$OUTPUT_DIR/logs" \
    --reports-dir "$OUTPUT_DIR/reports" \
    --output-dir "$OUTPUT_DIR/reports"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Dashboard generation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Dashboards generated${NC}"

################################################################################
# Summarize results
################################################################################
echo ""
echo -e "${CYAN}[4/5] Summarizing results...${NC}"

# Count total experiments
TOTAL_RUNS=$(find "$OUTPUT_DIR/logs" -name "*.jsonl" -exec wc -l {} + | tail -1 | awk '{print $1}')

# List generated dashboards
DASHBOARD_COUNT=$(find "$OUTPUT_DIR/reports" -name "dashboard.html" | wc -l)

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                     EXPERIMENT SUMMARY                             ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BLUE}Total Experiments:${NC}    $TOTAL_RUNS runs"
echo -e "  ${BLUE}Dashboards Generated:${NC} $DASHBOARD_COUNT scenarios"
echo -e "  ${BLUE}Output Directory:${NC}     $OUTPUT_DIR"
echo ""

# List all dashboards
echo -e "${YELLOW}📊 Generated Dashboards:${NC}"
for dashboard in $(find "$OUTPUT_DIR/reports" -name "dashboard.html" | sort); do
    scenario=$(basename $(dirname "$dashboard"))
    echo -e "  • ${GREEN}$scenario${NC}: file://$PWD/$dashboard"
done

################################################################################
# Final instructions
################################################################################
echo ""
echo -e "${CYAN}[5/5] Next steps...${NC}"
echo ""
echo -e "${YELLOW}To view results:${NC}"
echo -e "  1. Open any dashboard HTML file in your browser"
echo -e "  2. Review the comprehensive analysis report:"
echo -e "     ${BLUE}$OUTPUT_DIR/reports/comprehensive_analysis.json${NC}"
echo ""
echo -e "${YELLOW}Key files:${NC}"
echo -e "  • Raw data:         ${BLUE}$OUTPUT_DIR/logs/*.jsonl${NC}"
echo -e "  • Visualizations:   ${BLUE}$OUTPUT_DIR/artifacts/*.png${NC}"
echo -e "  • Analysis:         ${BLUE}$OUTPUT_DIR/reports/comprehensive_analysis.json${NC}"
echo -e "  • Dashboards:       ${BLUE}$OUTPUT_DIR/reports/*/dashboard.html${NC}"
echo ""

# Check for warnings/errors in validation
VALIDATION_FILE="$OUTPUT_DIR/reports/comprehensive_analysis.json"
if [ -f "$VALIDATION_FILE" ]; then
    WARNING_COUNT=$(python3 -c "import json; d=json.load(open('$VALIDATION_FILE')); print(sum(len(v.get('warnings', [])) for v in d.get('validation', {}).values()))")
    ERROR_COUNT=$(python3 -c "import json; d=json.load(open('$VALIDATION_FILE')); print(sum(len(v.get('errors', [])) for v in d.get('validation', {}).values()))")

    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo -e "${RED}⚠  $ERROR_COUNT validation errors found. Please review the analysis report.${NC}"
    elif [ "$WARNING_COUNT" -gt 0 ]; then
        echo -e "${YELLOW}⚠  $WARNING_COUNT validation warnings found. Review recommended.${NC}"
    else
        echo -e "${GREEN}✓ All validation checks passed!${NC}"
    fi
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                  ✓ EXPERIMENT SUITE COMPLETE                       ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
