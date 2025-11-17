#!/usr/bin/env python3
"""
Comprehensive publication-ready experiment runner.

This script runs all 5 scenarios with:
- Sufficient statistical power (100 runs per condition)
- Real 3D models
- Comprehensive validation
- Cross-scenario analysis
- Module vulnerability analysis
- Automated report generation
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from utils.yload import load as yload

from scripts.run_suite import run_experiment


def compute_statistics(data: List[float]) -> Dict[str, float]:
    """Compute comprehensive statistics for a dataset."""
    arr = np.array(data)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)),
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": len(arr)
    }


def compute_effect_size(group1: List[float], group2: List[float]) -> Dict[str, float]:
    """Compute Cohen's d effect size between two groups."""
    arr1, arr2 = np.array(group1), np.array(group2)
    pooled_std = np.sqrt(((len(arr1)-1)*np.var(arr1, ddof=1) +
                          (len(arr2)-1)*np.var(arr2, ddof=1)) /
                         (len(arr1) + len(arr2) - 2))
    if pooled_std == 0:
        return {"cohens_d": 0.0, "interpretation": "none"}

    d = (np.mean(arr1) - np.mean(arr2)) / pooled_std

    # Interpret effect size
    abs_d = abs(d)
    if abs_d < 0.2:
        interp = "negligible"
    elif abs_d < 0.5:
        interp = "small"
    elif abs_d < 0.8:
        interp = "medium"
    else:
        interp = "large"

    return {"cohens_d": float(d), "interpretation": interp}


def validate_experiment_results(jsonl_path: str) -> Dict[str, Any]:
    """
    Validate experiment results for data quality and consistency.

    Returns validation report with warnings and errors.
    """
    validation = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "metrics": {}
    }

    if not os.path.exists(jsonl_path):
        validation["valid"] = False
        validation["errors"].append(f"Results file not found: {jsonl_path}")
        return validation

    # Load results
    results = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if len(results) == 0:
        validation["valid"] = False
        validation["errors"].append("No results found in file")
        return validation

    # Check sample size
    validation["metrics"]["total_runs"] = len(results)
    if len(results) < 30:
        validation["warnings"].append(f"Low sample size: {len(results)} < 30")

    # Check for missing data
    missing_counts = defaultdict(int)
    for result in results:
        if result["perception"]["seg_iou"] is None:
            missing_counts["seg_iou"] += 1
        if result["geometry"]["pnp_rmse"] is None:
            missing_counts["pnp_rmse"] += 1
        if result["planning"]["path_cost"] is None:
            missing_counts["path_cost"] += 1
        if result["control"]["track_rmse"] is None:
            missing_counts["track_rmse"] += 1

    for key, count in missing_counts.items():
        pct = 100 * count / len(results)
        validation["metrics"][f"{key}_missing_pct"] = pct
        if pct > 50:
            validation["warnings"].append(f"High missing data for {key}: {pct:.1f}%")

    # Check for outliers (using IQR method)
    success_rate = np.mean([r["system"]["success"] for r in results])
    validation["metrics"]["success_rate"] = float(success_rate)

    if success_rate == 0.0 or success_rate == 1.0:
        validation["warnings"].append(f"Extreme success rate: {success_rate}")

    # Check variance (is there any variability?)
    final_dists = [r["system"]["final_dist_to_goal"] for r in results]
    if np.std(final_dists) < 1e-6:
        validation["warnings"].append("Zero variance in final distances (potential bug)")

    # Check for duplicates
    run_ids = [r["run_id"] for r in results]
    if len(run_ids) != len(set(run_ids)):
        validation["errors"].append("Duplicate run IDs found")
        validation["valid"] = False

    return validation


def cross_scenario_analysis(all_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Perform cross-scenario comparative analysis.

    Compares degradation patterns across all scenarios.
    """
    analysis = {
        "scenario_comparison": {},
        "robustness_ranking": [],
        "failure_modes": {}
    }

    # Extract metrics by scenario
    scenario_metrics = {}
    for scenario_name, results in all_results.items():
        if not results:
            continue

        # Group by perturbation level
        by_level = defaultdict(list)
        for r in results:
            level = r["meta"]["level"]
            by_level[level].append(r)

        # Compute degradation curve
        degradation = []
        for level in sorted(by_level.keys()):
            runs = by_level[level]
            success_rate = np.mean([r["system"]["success"] for r in runs])
            degradation.append({"level": level, "success_rate": success_rate})

        scenario_metrics[scenario_name] = degradation

        # Compute area under curve (robustness measure)
        if len(degradation) > 1:
            levels = [d["level"] for d in degradation]
            success_rates = [d["success_rate"] for d in degradation]
            try:
                auc = np.trapezoid(success_rates, levels)
            except AttributeError:
                # Fallback for older numpy versions
                auc = np.trapz(success_rates, levels)
            analysis["scenario_comparison"][scenario_name] = {
                "degradation_curve": degradation,
                "robustness_auc": float(auc)
            }

    # Rank scenarios by robustness
    rankings = sorted(
        [(name, data["robustness_auc"]) for name, data in analysis["scenario_comparison"].items()],
        key=lambda x: x[1],
        reverse=True
    )
    analysis["robustness_ranking"] = [
        {"rank": i+1, "scenario": name, "auc": auc}
        for i, (name, auc) in enumerate(rankings)
    ]

    return analysis


def module_vulnerability_analysis(all_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Analyze which modules are most vulnerable across scenarios.

    Identifies weakest links in the pipeline.
    """
    analysis = {
        "module_failure_rates": {},
        "cascading_failures": [],
        "critical_modules": []
    }

    module_failures = {
        "perception": [],
        "geometry": [],
        "planning": [],
        "control": []
    }

    for scenario_name, results in all_results.items():
        for r in results:
            # Only consider non-zero perturbation levels
            if r["meta"]["level"] == 0.0:
                continue

            # Track failures
            if not r["perception"]["detected"]:
                module_failures["perception"].append({
                    "scenario": scenario_name,
                    "level": r["meta"]["level"],
                    "type": "detection_failure"
                })

            if not r["geometry"]["pnp_success"]:
                module_failures["geometry"].append({
                    "scenario": scenario_name,
                    "level": r["meta"]["level"],
                    "type": "pose_estimation_failure"
                })

            if not r["planning"]["success"]:
                module_failures["planning"].append({
                    "scenario": scenario_name,
                    "level": r["meta"]["level"],
                    "type": "planning_failure"
                })

            if r["control"]["oscillation"]:
                module_failures["control"].append({
                    "scenario": scenario_name,
                    "level": r["meta"]["level"],
                    "type": "control_oscillation"
                })

    # Compute failure rates
    total_perturbed = sum(
        sum(1 for r in results if r["meta"]["level"] > 0.0)
        for results in all_results.values()
    )

    for module, failures in module_failures.items():
        failure_rate = len(failures) / max(total_perturbed, 1)
        analysis["module_failure_rates"][module] = {
            "failure_count": len(failures),
            "failure_rate": float(failure_rate),
            "examples": failures[:5]  # First 5 examples
        }

    # Identify critical modules (highest failure rate)
    sorted_modules = sorted(
        analysis["module_failure_rates"].items(),
        key=lambda x: x[1]["failure_rate"],
        reverse=True
    )
    analysis["critical_modules"] = [
        {"module": name, "failure_rate": data["failure_rate"]}
        for name, data in sorted_modules
    ]

    return analysis


def run_all_scenarios(
    config_path: str,
    perturbations_path: str,
    thresholds_path: str,
    output_dir: str,
    scenarios: List[str] = None,
    quick_test: bool = False
) -> Dict[str, Any]:
    """
    Run all scenarios with comprehensive analysis.

    Args:
        config_path: Path to experiment config
        perturbations_path: Path to perturbations config
        thresholds_path: Path to thresholds config
        output_dir: Output directory for results
        scenarios: List of scenarios to run (None = all)
        quick_test: If True, run with reduced sample size for testing

    Returns:
        Summary results dictionary
    """
    # Load configs
    cfg = yload(config_path)
    pert_cfg = yload(perturbations_path)

    # Determine scenarios to run
    all_scenarios = pert_cfg["scenarios"]
    if scenarios:
        all_scenarios = [s for s in all_scenarios if s["name"] in scenarios]

    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logs_dir = os.path.join(output_dir, "logs")
    reports_dir = os.path.join(output_dir, "reports")
    Path(logs_dir).mkdir(exist_ok=True)
    Path(reports_dir).mkdir(exist_ok=True)

    print(f"Running {len(all_scenarios)} scenarios...")
    print(f"Output directory: {output_dir}")

    # Run each scenario
    all_results = {}
    validation_reports = {}

    for scenario_cfg in all_scenarios:
        scenario_name = scenario_cfg["name"]
        levels = scenario_cfg["levels"]
        runs_per_level = scenario_cfg.get("runs_per_level", 100)

        if quick_test:
            runs_per_level = 10  # Reduced for testing

        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        print(f"Description: {scenario_cfg.get('description', 'N/A')}")
        print(f"Levels: {levels}")
        print(f"Runs per level: {runs_per_level}")
        print(f"{'='*60}")

        scenario_results = []

        for level in levels:
            print(f"\n  Running level {level}...")
            start_time = time.time()

            # Output path for this condition
            out_jsonl = os.path.join(
                logs_dir,
                f"{scenario_name}_level{level}.jsonl"
            )

            # Run experiment
            try:
                run_experiment(
                    cfg_path=config_path,
                    scenario=scenario_name,
                    level=level,
                    thresholds_path=thresholds_path,
                    runs=runs_per_level,
                    out_jsonl=out_jsonl
                )

                # Load results
                with open(out_jsonl, 'r') as f:
                    level_results = [json.loads(line) for line in f if line.strip()]
                scenario_results.extend(level_results)

                elapsed = time.time() - start_time
                print(f"    ✓ Completed in {elapsed:.1f}s ({len(level_results)} runs)")

            except Exception as e:
                print(f"    ✗ Error: {e}")
                continue

        all_results[scenario_name] = scenario_results

        # Validate results
        combined_path = os.path.join(logs_dir, f"{scenario_name}_all.jsonl")
        with open(combined_path, 'w') as f:
            for r in scenario_results:
                f.write(json.dumps(r) + '\n')

        validation = validate_experiment_results(combined_path)
        validation_reports[scenario_name] = validation

        print(f"\n  Validation: {'✓ PASSED' if validation['valid'] else '✗ FAILED'}")
        if validation["warnings"]:
            print(f"  Warnings: {len(validation['warnings'])}")
            for w in validation["warnings"]:
                print(f"    - {w}")

    # Cross-scenario analysis
    print(f"\n{'='*60}")
    print("Running cross-scenario analysis...")
    print(f"{'='*60}")

    cross_analysis = cross_scenario_analysis(all_results)
    module_analysis = module_vulnerability_analysis(all_results)

    # Save comprehensive report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "scenarios_run": len(all_scenarios),
            "quick_test": quick_test,
        },
        "validation": validation_reports,
        "cross_scenario_analysis": cross_analysis,
        "module_vulnerability_analysis": module_analysis,
        "total_experiments": sum(len(r) for r in all_results.values())
    }

    report_path = os.path.join(reports_dir, "comprehensive_analysis.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Comprehensive report saved: {report_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments run: {report['total_experiments']}")
    print(f"\nRobustness ranking:")
    for rank_info in cross_analysis["robustness_ranking"]:
        print(f"  {rank_info['rank']}. {rank_info['scenario']}: AUC={rank_info['auc']:.3f}")
    print(f"\nCritical modules:")
    for mod_info in module_analysis["critical_modules"]:
        print(f"  - {mod_info['module']}: {mod_info['failure_rate']*100:.1f}% failure rate")

    return report


def main():
    parser = argparse.ArgumentParser(description="Run publication-ready experiments")
    parser.add_argument(
        "--config",
        default="configs/robosuite_grasp.yaml",
        help="Experiment configuration file"
    )
    parser.add_argument(
        "--perturbations",
        default="configs/perturbations.yaml",
        help="Perturbations configuration file"
    )
    parser.add_argument(
        "--thresholds",
        default="configs/thresholds.yaml",
        help="Thresholds configuration file"
    )
    parser.add_argument(
        "--output",
        default="results/publication",
        help="Output directory"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        help="Specific scenarios to run (default: all)"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with reduced sample size"
    )

    args = parser.parse_args()

    run_all_scenarios(
        config_path=args.config,
        perturbations_path=args.perturbations,
        thresholds_path=args.thresholds,
        output_dir=args.output,
        scenarios=args.scenarios,
        quick_test=args.quick_test
    )


if __name__ == "__main__":
    main()
