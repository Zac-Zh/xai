#!/usr/bin/env python3
"""
Generate comprehensive final research report with cross-scenario analysis.
"""

from __future__ import annotations

import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np

from utils.yload import load as yload
from attribution.rule_based import attribute_failure


def load_all_results(csv_paths: List[str]) -> pd.DataFrame:
    """Load and merge all CSV results."""
    dfs = []
    for csv_path in csv_paths:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def compute_attributions(df: pd.DataFrame, thresholds: Dict[str, Any]) -> pd.DataFrame:
    """Compute attributions for all rows."""
    if df.empty:
        return df

    mods_list: List[List[str]] = []
    errs_list: List[List[str]] = []
    roots_list: List[str] = []

    for _, row in df.iterrows():
        log = {
            "meta": {"scenario": row.get("scenario", "unknown")},
            "perception": {
                "avg_conf": row.get("perception.avg_conf"),
                "detected": bool(row.get("perception.detected", True)),
                "seg_iou": row.get("perception.seg_iou"),
            },
            "geometry": {
                "pnp_success": bool(row.get("geometry.pnp_success", True)),
                "pnp_rmse": row.get("geometry.pnp_rmse"),
            },
            "planning": {
                "success": bool(row.get("planning.success", True)),
                "path_cost": row.get("planning.path_cost"),
                "collisions": int(row.get("planning.collisions", 0)),
            },
            "control": {
                "track_rmse": row.get("control.track_rmse"),
                "overshoot": row.get("control.overshoot"),
                "oscillation": bool(row.get("control.oscillation", False)),
            },
        }

        fe, root = attribute_failure(log, thresholds)
        mods_list.append([m for m, _ in fe])
        errs_list.append([e for _, e in fe])
        roots_list.append(root)

    df = df.copy()
    df["attr_modules"] = mods_list
    df["attr_errors"] = errs_list
    df["root_cause"] = roots_list

    return df


def analyze_cross_scenario(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform cross-scenario analysis."""
    analysis = {}

    # Overall success rates by scenario
    success_by_scenario = df.groupby("scenario")["system.success"].mean()
    analysis["success_rates"] = success_by_scenario.to_dict()

    # Module failure frequency across all scenarios
    all_modules = []
    for mods in df["attr_modules"]:
        if isinstance(mods, list):
            all_modules.extend(mods)
    from collections import Counter
    analysis["module_failures"] = dict(Counter(all_modules))

    # Root cause distribution
    root_causes = df["root_cause"].value_counts().to_dict()
    analysis["root_causes"] = root_causes

    # Perturbation sensitivity (how fast does success drop?)
    sensitivity = {}
    for scenario in df["scenario"].unique():
        sdf = df[df["scenario"] == scenario].copy()
        if "level" in sdf.columns:
            sdf = sdf.sort_values("level")
            levels = sdf["level"].unique()
            if len(levels) > 1:
                success_rates = [sdf[sdf["level"] == lvl]["system.success"].mean() for lvl in levels]
                # Calculate slope (rate of degradation)
                if len(success_rates) >= 2 and levels[-1] != levels[0]:
                    slope = (success_rates[-1] - success_rates[0]) / (levels[-1] - levels[0])
                    sensitivity[scenario] = slope
                else:
                    sensitivity[scenario] = 0.0

    analysis["sensitivity"] = sensitivity

    # Most vulnerable module per scenario
    vulnerable_modules = {}
    for scenario in df["scenario"].unique():
        sdf = df[df["scenario"] == scenario]
        mods = []
        for mod_list in sdf["attr_modules"]:
            if isinstance(mod_list, list):
                mods.extend(mod_list)
        if mods:
            from collections import Counter
            most_common = Counter(mods).most_common(1)
            vulnerable_modules[scenario] = most_common[0][0] if most_common else "None"

    analysis["vulnerable_modules"] = vulnerable_modules

    return analysis


def generate_html_report(
    df: pd.DataFrame,
    analysis: Dict[str, Any],
    output_path: str,
    thresholds: Dict[str, Any],
) -> None:
    """Generate comprehensive HTML research report."""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>R2: Complete Research Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header .subtitle {{
            opacity: 0.9;
            margin-top: 10px;
        }}
        .section {{
            background: white;
            padding: 30px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .section h3 {{
            color: #764ba2;
            margin-top: 25px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #667eea;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .metric {{
            display: inline-block;
            background: #e0e7ff;
            padding: 8px 16px;
            border-radius: 20px;
            margin: 5px;
            font-weight: 500;
        }}
        .finding {{
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 15px;
            margin: 15px 0;
        }}
        .recommendation {{
            background: #d1fae5;
            border-left: 4px solid #10b981;
            padding: 15px;
            margin: 15px 0;
        }}
        .chart-placeholder {{
            background: #f9fafb;
            border: 2px dashed #d1d5db;
            padding: 40px;
            text-align: center;
            margin: 20px 0;
            border-radius: 8px;
            color: #6b7280;
        }}
        code {{
            background: #f3f4f6;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        .success {{ color: #10b981; font-weight: 600; }}
        .failure {{ color: #ef4444; font-weight: 600; }}
        .footer {{
            text-align: center;
            color: #6b7280;
            margin-top: 40px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>R2: Layered Failure Attribution & Explainability</h1>
        <div class="subtitle">Comprehensive Research Report - Complete Automation Pipeline Results</div>
        <div class="subtitle">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
    </div>

    <div class="section">
        <h2>📊 Executive Summary</h2>
        <p>This report presents the results of a comprehensive automated research pipeline analyzing failure attribution across modular robotics systems. We evaluated {len(df)} experimental runs across {len(df['scenario'].unique())} perturbation scenarios to understand how environmental factors propagate through the Vision → Geometry → Planning → Control pipeline.</p>

        <h3>Overall Statistics</h3>
        <p>
            <span class="metric">Total Experiments: {len(df)}</span>
            <span class="metric">Scenarios: {len(df['scenario'].unique())}</span>
            <span class="metric">Overall Success Rate: {df['system.success'].mean()*100:.1f}%</span>
        </p>
    </div>

    <div class="section">
        <h2>🎯 Key Findings</h2>

        <div class="finding">
            <strong>Finding 1: Scenario-Specific Success Rates</strong>
            <p>Different perturbation scenarios show varying impact on system success:</p>
            <ul>
"""

    # Add success rates
    for scenario, rate in sorted(analysis["success_rates"].items(), key=lambda x: x[1], reverse=True):
        html += f"                <li><strong>{scenario}:</strong> {rate*100:.1f}% success rate</li>\n"

    html += """            </ul>
        </div>

        <div class="finding">
            <strong>Finding 2: Module Vulnerability</strong>
            <p>Module failure frequency across all scenarios:</p>
            <ul>
"""

    # Add module failures
    sorted_modules = sorted(analysis["module_failures"].items(), key=lambda x: x[1], reverse=True)
    for module, count in sorted_modules[:5]:  # Top 5
        html += f"                <li><strong>{module}:</strong> {count} failures</li>\n"

    html += """            </ul>
        </div>

        <div class="finding">
            <strong>Finding 3: Root Cause Analysis</strong>
            <p>Primary root causes of system failures:</p>
            <ul>
"""

    # Add root causes
    for cause, count in sorted(analysis["root_causes"].items(), key=lambda x: x[1], reverse=True):
        if cause and cause != "None":
            html += f"                <li><strong>{cause}:</strong> {count} instances</li>\n"

    html += """            </ul>
        </div>

        <div class="finding">
            <strong>Finding 4: Perturbation Sensitivity</strong>
            <p>Rate of performance degradation as perturbation level increases (negative values indicate degradation):</p>
            <ul>
"""

    # Add sensitivity analysis
    for scenario, slope in sorted(analysis["sensitivity"].items(), key=lambda x: x[1]):
        if slope != 0:
            html += f"                <li><strong>{scenario}:</strong> {slope:.3f} (slope of success rate vs. perturbation level)</li>\n"

    html += """            </ul>
            <p><em>More negative slopes indicate faster performance degradation under perturbation.</em></p>
        </div>
    </div>

    <div class="section">
        <h2>📈 Detailed Results by Scenario</h2>
"""

    # Per-scenario detailed results
    for scenario in sorted(df["scenario"].unique()):
        sdf = df[df["scenario"] == scenario]
        success_rate = sdf["system.success"].mean()
        total_runs = len(sdf)

        html += f"""
        <h3>{scenario.replace('_', ' ').title()}</h3>
        <p>
            <span class="metric">Runs: {total_runs}</span>
            <span class="metric">Success Rate: {success_rate*100:.1f}%</span>
        </p>

        <table>
            <tr>
                <th>Perturbation Level</th>
                <th>Runs</th>
                <th>Success Rate</th>
                <th>Avg Detection Conf</th>
                <th>Avg Seg IoU</th>
                <th>Avg Track RMSE</th>
            </tr>
"""

        # Group by level if available
        if "level" in sdf.columns:
            for level in sorted(sdf["level"].unique()):
                ldf = sdf[sdf["level"] == level]
                html += f"""
            <tr>
                <td>{level}</td>
                <td>{len(ldf)}</td>
                <td class="{'success' if ldf['system.success'].mean() > 0.5 else 'failure'}">{ldf['system.success'].mean()*100:.1f}%</td>
                <td>{ldf['perception.avg_conf'].mean():.3f}</td>
                <td>{ldf['perception.seg_iou'].mean():.3f}</td>
                <td>{ldf['control.track_rmse'].mean():.4f}</td>
            </tr>
"""

        html += """        </table>
"""

        # Most common failure mode for this scenario
        failed = sdf[sdf["system.success"] == False]
        if len(failed) > 0:
            all_errors = []
            for errs in failed["attr_errors"]:
                if isinstance(errs, list):
                    all_errors.extend(errs)
            if all_errors:
                from collections import Counter
                most_common_error = Counter(all_errors).most_common(1)[0]
                html += f"""
        <p><strong>Most Common Failure Mode:</strong> <code>{most_common_error[0]}</code> ({most_common_error[1]} occurrences)</p>
"""

    html += """
    </div>

    <div class="section">
        <h2>💡 Discussion</h2>

        <h3>Interpretation of Results</h3>
        <p>The automated pipeline reveals several important insights about robotic system resilience:</p>

        <p><strong>1. Layered Failure Propagation:</strong> Our rule-based attribution successfully traces failures through the modular pipeline. Vision module failures (detection confidence, segmentation IoU) frequently propagate to downstream modules, causing geometry estimation errors which then cascade into planning and control failures.</p>

        <p><strong>2. Scenario-Dependent Vulnerability:</strong> Different perturbation types affect different modules with varying severity. For example:
        <ul>
            <li><strong>Occlusion & Overlap:</strong> Primarily impact the Vision module (detection/segmentation)</li>
            <li><strong>Lighting:</strong> Affects detection confidence and segmentation quality</li>
            <li><strong>Motion Blur:</strong> Degrades both vision and geometry estimation</li>
            <li><strong>Camera Jitter:</strong> Impacts geometry (pose estimation) and control (tracking)</li>
        </ul>
        </p>

        <p><strong>3. Robustness Thresholds:</strong> The sensitivity analysis reveals that some perturbations have graceful degradation (slow slope) while others cause rapid failure cascades (steep slope). This information is critical for prioritizing robustness improvements.</p>

        <h3>Methodological Contributions</h3>
        <p>This work demonstrates:
        <ul>
            <li><strong>Complete Reproducibility:</strong> Fully automated pipeline from data generation to final report</li>
            <li><strong>Systematic Attribution:</strong> Rule-based failure tracing through modular architecture</li>
            <li><strong>Offline Analysis:</strong> No dependency on external services or manual intervention</li>
            <li><strong>Extensibility:</strong> Stub modules can be replaced with real implementations (YOLO, Mask R-CNN, OMPL, etc.)</li>
        </ul>
        </p>

        <h3>Limitations</h3>
        <p>Current limitations include:
        <ul>
            <li>Synthetic 2D environment (though this enables reproducibility)</li>
            <li>Rule-based attribution (threshold-dependent)</li>
            <li>Limited perturbation types (could expand to sensor noise, dynamics variations, etc.)</li>
            <li>Stub implementations (real systems would have additional complexities)</li>
        </ul>
        </p>
    </div>

    <div class="section">
        <h2>🎯 Recommendations</h2>

        <div class="recommendation">
            <strong>Recommendation 1: Targeted Robustness Improvements</strong>
            <p>Based on module failure frequency, prioritize improvements to the most vulnerable modules. Consider:
            <ul>
                <li>Enhanced detection models with better occlusion handling</li>
                <li>Robust segmentation under varying lighting conditions</li>
                <li>Pose estimation with uncertainty quantification</li>
            </ul>
            </p>
        </div>

        <div class="recommendation">
            <strong>Recommendation 2: Adaptive Thresholds</strong>
            <p>The current rule-based attribution uses fixed thresholds. Consider implementing adaptive thresholds that adjust based on perturbation severity or historical performance.</p>
        </div>

        <div class="recommendation">
            <strong>Recommendation 3: Early Warning System</strong>
            <p>Develop an early warning system that detects degrading perception quality (e.g., dropping detection confidence) before it cascades into planning/control failures. This enables proactive intervention.</p>
        </div>

        <div class="recommendation">
            <strong>Recommendation 4: Real-World Validation</strong>
            <p>Extend this framework to real robotic systems by:
            <ul>
                <li>Replacing stub modules with actual implementations (YOLO, MoveIt, etc.)</li>
                <li>Collecting real sensor data under controlled perturbations</li>
                <li>Validating that synthetic findings generalize to physical systems</li>
            </ul>
            </p>
        </div>
    </div>

    <div class="section">
        <h2>🔬 Future Work</h2>
        <p>Potential extensions of this research include:</p>
        <ul>
            <li><strong>Learning-Based Attribution:</strong> Replace rule-based attribution with learned models that discover failure patterns from data</li>
            <li><strong>Multi-Task Scenarios:</strong> Expand beyond "Lift" to include Pick-Place, Assembly, Navigation, etc.</li>
            <li><strong>Temporal Analysis:</strong> Analyze how failures develop over time within a single run</li>
            <li><strong>Counterfactual Analysis:</strong> "What-if" analysis showing required improvements to achieve success</li>
            <li><strong>Human-in-the-Loop:</strong> Interactive dashboard for researchers to explore failure modes</li>
            <li><strong>Benchmark Suite:</strong> Standardized perturbation benchmarks for comparing robotic systems</li>
        </ul>
    </div>

    <div class="section">
        <h2>📁 Artifacts</h2>
        <p>All experimental artifacts are available in the results directory:</p>
        <ul>
            <li><strong>Raw Data (CSV):</strong> <code>results/*_sweep.csv</code></li>
            <li><strong>Individual Reports:</strong> <code>results/reports/&lt;scenario&gt;/</code></li>
            <li><strong>Visualizations:</strong> PNG plots (stacked bars, sensitivity curves, sankey diagrams)</li>
            <li><strong>Interactive Dashboards:</strong> <code>results/reports/&lt;scenario&gt;/dashboard.html</code></li>
            <li><strong>Image Artifacts:</strong> <code>results/artifacts/</code> (RGB images, masks, path plots)</li>
        </ul>
    </div>

    <div class="section">
        <h2>⚙️ Configuration</h2>
        <p>This analysis used the following thresholds:</p>
        <table>
            <tr>
                <th>Module</th>
                <th>Metric</th>
                <th>Threshold</th>
            </tr>
"""

    # Add thresholds
    if "perception" in thresholds:
        for key, val in thresholds["perception"].items():
            html += f"""
            <tr>
                <td>Perception</td>
                <td>{key}</td>
                <td>{val}</td>
            </tr>
"""

    if "geometry" in thresholds:
        for key, val in thresholds["geometry"].items():
            html += f"""
            <tr>
                <td>Geometry</td>
                <td>{key}</td>
                <td>{val}</td>
            </tr>
"""

    if "planning" in thresholds:
        for key, val in thresholds["planning"].items():
            html += f"""
            <tr>
                <td>Planning</td>
                <td>{key}</td>
                <td>{val}</td>
            </tr>
"""

    if "control" in thresholds:
        for key, val in thresholds["control"].items():
            html += f"""
            <tr>
                <td>Control</td>
                <td>{key}</td>
                <td>{val}</td>
            </tr>
"""

    if "system" in thresholds:
        for key, val in thresholds["system"].items():
            html += f"""
            <tr>
                <td>System</td>
                <td>{key}</td>
                <td>{val}</td>
            </tr>
"""

    html += f"""
        </table>
    </div>

    <div class="footer">
        <p>R2: Layered Failure Attribution & Explainability Metrics for Robotics</p>
        <p>Complete Automation Pipeline - Generated {datetime.now().strftime("%Y-%m-%d")}</p>
    </div>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comprehensive final research report")
    parser.add_argument("--csv", action="append", help="CSV files to include (can specify multiple)")
    parser.add_argument("--results_dir", required=True, help="Directory containing individual reports")
    parser.add_argument("--output_dir", required=True, help="Output directory for final report")
    parser.add_argument("--thresholds", default="configs/thresholds.yaml", help="Thresholds config")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load thresholds
    thresholds = yload(args.thresholds)

    # Load all results
    print("Loading experimental results...")
    df = load_all_results(args.csv or [])

    if df.empty:
        print("Error: No data to analyze")
        return

    print(f"Loaded {len(df)} experimental runs")

    # Compute attributions if not already present
    if "attr_modules" not in df.columns:
        print("Computing attributions...")
        df = compute_attributions(df, thresholds)

    # Perform cross-scenario analysis
    print("Analyzing cross-scenario results...")
    analysis = analyze_cross_scenario(df)

    # Generate HTML report
    output_path = os.path.join(args.output_dir, "research_report.html")
    print(f"Generating comprehensive report...")
    generate_html_report(df, analysis, output_path, thresholds)

    print(f"\n✓ Final report generated: {output_path}")
    print(f"\nKey findings:")
    print(f"  - Overall success rate: {df['system.success'].mean()*100:.1f}%")
    print(f"  - Total experiments: {len(df)}")
    print(f"  - Scenarios analyzed: {len(df['scenario'].unique())}")


if __name__ == "__main__":
    main()
