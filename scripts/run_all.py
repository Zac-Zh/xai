#!/usr/bin/env python3
"""
Master automation script for complete R2 research pipeline.
Handles: validation → experiments → analysis → visualization → final report
"""

from __future__ import annotations

import os
import sys
import subprocess
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.yload import load as yload


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def run_command(cmd: List[str], description: str) -> Tuple[bool, float]:
    """Run a command and return success status and duration."""
    print(f"→ {description}...")
    start = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        duration = time.time() - start
        print(f"  ✓ Completed in {duration:.2f}s")
        return True, duration
    except subprocess.CalledProcessError as e:
        duration = time.time() - start
        print(f"  ✗ Failed after {duration:.2f}s")
        print(f"    Error: {e.stderr[:200]}")
        return False, duration


def validate_environment() -> bool:
    """Run tests to validate the environment."""
    print_section("STEP 1: Environment Validation")

    print("Checking Python version...")
    py_version = sys.version_info
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 10):
        print(f"  ✗ Python 3.10+ required, found {py_version.major}.{py_version.minor}")
        return False
    print(f"  ✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")

    print("\nChecking required modules...")
    required = ["numpy", "pandas", "matplotlib", "plotly", "yaml"]
    missing = []
    for mod in required:
        try:
            __import__(mod)
            print(f"  ✓ {mod}")
        except ImportError:
            print(f"  ✗ {mod} (missing)")
            missing.append(mod)

    if missing:
        print(f"\n✗ Missing dependencies: {', '.join(missing)}")
        print("  Run: pip install -r requirements.txt")
        return False

    print("\nRunning tests...")
    tests = [
        (["python", "-m", "tests.test_schemas"], "Schema validation tests"),
        (["python", "-m", "tests.test_attribution"], "Attribution logic tests"),
    ]

    for cmd, desc in tests:
        success, _ = run_command(cmd, desc)
        if not success:
            return False

    print("\n✓ All validation checks passed")
    return True


def setup_directories(base_dir: str) -> Dict[str, str]:
    """Create directory structure for results."""
    print_section("STEP 2: Directory Setup")

    dirs = {
        "base": base_dir,
        "logs": os.path.join(base_dir, "logs"),
        "artifacts": os.path.join(base_dir, "artifacts"),
        "reports": os.path.join(base_dir, "reports"),
        "final": os.path.join(base_dir, "final_report"),
    }

    for name, path in dirs.items():
        os.makedirs(path, exist_ok=True)
        print(f"  ✓ {name:12s} → {path}")

    return dirs


def run_perturbation_sweeps(
    config_path: str,
    thresholds_path: str,
    perturb_config: str,
    dirs: Dict[str, str],
    runs: int,
) -> List[Tuple[str, str, List[float]]]:
    """Run all perturbation sweeps defined in config."""
    print_section("STEP 3: Running Perturbation Sweeps")

    # Load perturbation scenarios
    perturb_data = yload(perturb_config)
    scenarios = perturb_data["scenarios"]

    print(f"Found {len(scenarios)} perturbation scenarios:")
    for s in scenarios:
        print(f"  • {s['name']:15s} levels: {s['levels']}")

    results: List[Tuple[str, str, List[float]]] = []

    for i, scenario in enumerate(scenarios, 1):
        name = scenario["name"]
        levels = scenario["levels"]

        print(f"\n[{i}/{len(scenarios)}] Running {name} sweep...")

        csv_path = os.path.join(dirs["base"], f"{name}_sweep.csv")
        levels_str = " ".join(str(lvl) for lvl in levels)

        cmd = [
            "python", "scripts/sweep_perturb.py",
            "--cfg", config_path,
            "--scenario", name,
            "--levels", *levels_str.split(),
            "--thresholds", thresholds_path,
            "--runs", str(runs),
            "--merge_csv", csv_path,
        ]

        success, duration = run_command(cmd, f"{name} perturbation sweep")

        if success:
            results.append((name, csv_path, levels))
        else:
            print(f"  ⚠ Skipping {name} due to errors")

    print(f"\n✓ Completed {len(results)}/{len(scenarios)} sweeps successfully")
    return results


def generate_reports(
    results: List[Tuple[str, str, List[float]]],
    thresholds_path: str,
    reports_dir: str,
) -> List[str]:
    """Generate individual reports for each perturbation scenario."""
    print_section("STEP 4: Generating Individual Reports")

    report_paths = []

    for i, (scenario, csv_path, levels) in enumerate(results, 1):
        print(f"\n[{i}/{len(results)}] Generating report for {scenario}...")

        out_dir = os.path.join(reports_dir, scenario)

        cmd = [
            "python", "scripts/export_report.py",
            "--csv", csv_path,
            "--out", out_dir,
            "--thresholds", thresholds_path,
        ]

        success, _ = run_command(cmd, f"{scenario} visualization & dashboard")

        if success:
            report_paths.append(out_dir)
            print(f"  → Report saved to: {out_dir}")

    print(f"\n✓ Generated {len(report_paths)} reports")
    return report_paths


def generate_final_report(
    results: List[Tuple[str, str, List[float]]],
    report_paths: List[str],
    final_dir: str,
    thresholds_path: str,
) -> None:
    """Generate comprehensive final research report with discussion."""
    print_section("STEP 5: Generating Final Research Report")

    cmd = [
        "python", "scripts/generate_final_report.py",
        "--results_dir", os.path.dirname(report_paths[0]) if report_paths else "results/reports",
        "--output_dir", final_dir,
        "--thresholds", thresholds_path,
    ]

    # Add all CSV paths
    for scenario, csv_path, _ in results:
        cmd.extend(["--csv", csv_path])

    success, _ = run_command(cmd, "Comprehensive final report with cross-scenario analysis")

    if success:
        print(f"\n  → Final report: {os.path.join(final_dir, 'research_report.html')}")


def print_summary(start_time: float, results: List[Tuple[str, str, List[float]]]) -> None:
    """Print execution summary."""
    duration = time.time() - start_time
    minutes, seconds = divmod(int(duration), 60)

    print_section("EXECUTION SUMMARY")

    print(f"Total duration: {minutes}m {seconds}s")
    print(f"Scenarios completed: {len(results)}")
    print("\nResults:")
    for scenario, csv_path, levels in results:
        print(f"  • {scenario:15s} ({len(levels)} levels) → {csv_path}")

    print("\nOutputs:")
    print("  • Individual reports:  results/reports/<scenario>/")
    print("  • PNG plots:          results/reports/<scenario>/*.png")
    print("  • HTML dashboards:    results/reports/<scenario>/dashboard.html")
    print("  • Final report:       results/final_report/research_report.html")
    print("  • Raw data (CSV):     results/*_sweep.csv")
    print("  • Artifacts:          results/artifacts/")

    print("\n" + "=" * 80)
    print("  🎉 COMPLETE! All automation pipelines finished successfully.")
    print("=" * 80 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Complete automation pipeline for R2 research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with defaults
  python scripts/run_all.py

  # Custom configuration
  python scripts/run_all.py --runs 5 --results_dir custom_results

  # Skip validation (faster, but risky)
  python scripts/run_all.py --skip_validation
        """,
    )

    parser.add_argument(
        "--cfg",
        default="configs/robosuite_grasp.yaml",
        help="Task configuration file (default: configs/robosuite_grasp.yaml)",
    )
    parser.add_argument(
        "--thresholds",
        default="configs/thresholds.yaml",
        help="Thresholds configuration (default: configs/thresholds.yaml)",
    )
    parser.add_argument(
        "--perturbations",
        default="configs/perturbations.yaml",
        help="Perturbations configuration (default: configs/perturbations.yaml)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per perturbation level (default: 3)",
    )
    parser.add_argument(
        "--results_dir",
        default="results",
        help="Base directory for results (default: results)",
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip environment validation and tests (not recommended)",
    )

    args = parser.parse_args()

    start_time = time.time()

    print("\n" + "=" * 80)
    print("  R2: Complete Research Automation Pipeline")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)

    # Step 1: Validate environment
    if not args.skip_validation:
        if not validate_environment():
            print("\n✗ Validation failed. Fix errors and try again.")
            return 1
    else:
        print_section("STEP 1: Environment Validation")
        print("⚠ Skipped (--skip_validation enabled)")

    # Step 2: Setup directories
    dirs = setup_directories(args.results_dir)

    # Step 3: Run perturbation sweeps
    results = run_perturbation_sweeps(
        args.cfg,
        args.thresholds,
        args.perturbations,
        dirs,
        args.runs,
    )

    if not results:
        print("\n✗ No successful sweeps completed. Aborting.")
        return 1

    # Step 4: Generate individual reports
    report_paths = generate_reports(results, args.thresholds, dirs["reports"])

    # Step 5: Generate final comprehensive report
    generate_final_report(results, report_paths, dirs["final"], args.thresholds)

    # Summary
    print_summary(start_time, results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
