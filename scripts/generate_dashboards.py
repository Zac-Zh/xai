#!/usr/bin/env python3
"""
Generate comprehensive dashboards for all scenarios.

Creates interactive HTML dashboards with:
- Degradation curves
- Statistical comparisons
- Module-level analysis
- Cross-scenario comparisons
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats


def load_results(logs_dir: str, scenario: str) -> List[Dict]:
    """Load all results for a scenario."""
    pattern = f"{scenario}_level*.jsonl"
    results = []

    for jsonl_file in Path(logs_dir).glob(pattern):
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

    return results


def create_html_dashboard(
    scenario_name: str,
    results: List[Dict],
    cross_scenario_data: Dict = None,
    module_analysis: Dict = None,
    output_path: str = "dashboard.html"
):
    """
    Create comprehensive HTML dashboard for a scenario.

    Args:
        scenario_name: Name of the scenario
        results: List of experiment results
        cross_scenario_data: Cross-scenario comparison data
        module_analysis: Module vulnerability analysis
        output_path: Where to save the HTML file
    """
    # Group results by level
    by_level = defaultdict(list)
    for r in results:
        level = r["meta"]["level"]
        by_level[level].append(r)

    # Compute statistics for each level
    level_stats = {}
    for level, runs in sorted(by_level.items()):
        stats_dict = {
            "level": level,
            "n_runs": len(runs),
            "success_rate": np.mean([r["system"]["success"] for r in runs]),
            "final_dist_mean": np.mean([r["system"]["final_dist_to_goal"] for r in runs]),
            "final_dist_std": np.std([r["system"]["final_dist_to_goal"] for r in runs]),
            "perception": {
                "avg_conf_mean": np.mean([r["perception"]["avg_conf"] for r in runs]),
                "detection_rate": np.mean([r["perception"]["detected"] for r in runs]),
            },
            "geometry": {
                "pnp_success_rate": np.mean([r["geometry"]["pnp_success"] for r in runs]),
            },
            "planning": {
                "success_rate": np.mean([r["planning"]["success"] for r in runs]),
                "collision_mean": np.mean([r["planning"]["collisions"] for r in runs]),
            },
            "control": {
                "oscillation_rate": np.mean([r["control"]["oscillation"] for r in runs]),
            }
        }
        level_stats[level] = stats_dict

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{scenario_name.upper()} - XAI Experiment Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 2px;
        }}
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .content {{
            padding: 40px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #667eea;
            font-size: 1.8em;
            margin-bottom: 20px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        .section h3 {{
            color: #764ba2;
            font-size: 1.3em;
            margin: 20px 0 10px 0;
        }}
        .plot {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .stat-card h4 {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .stat-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .stat-card .label {{
            font-size: 0.9em;
            opacity: 0.8;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 1px;
        }}
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e9ecef;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .badge-success {{ background: #28a745; color: white; }}
        .badge-warning {{ background: #ffc107; color: #333; }}
        .badge-danger {{ background: #dc3545; color: white; }}
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{scenario_name}</h1>
            <p>Comprehensive XAI Experiment Analysis</p>
        </div>

        <div class="content">
"""

    # Summary statistics
    total_runs = len(results)
    overall_success = np.mean([r["system"]["success"] for r in results])

    html += f"""
            <section class="section">
                <h2>📊 Summary Statistics</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <h4>Total Runs</h4>
                        <div class="value">{total_runs}</div>
                        <div class="label">Experiments conducted</div>
                    </div>
                    <div class="stat-card">
                        <h4>Overall Success Rate</h4>
                        <div class="value">{overall_success*100:.1f}%</div>
                        <div class="label">System-level success</div>
                    </div>
                    <div class="stat-card">
                        <h4>Perturbation Levels</h4>
                        <div class="value">{len(by_level)}</div>
                        <div class="label">Conditions tested</div>
                    </div>
                    <div class="stat-card">
                        <h4>Statistical Power</h4>
                        <div class="value">{"✓" if total_runs >= 100 else "?"}</div>
                        <div class="label">{"Sufficient (n≥100)" if total_runs >= 100 else f"Limited (n={total_runs})"}</div>
                    </div>
                </div>
            </section>
"""

    # Main degradation plot
    levels = sorted(by_level.keys())
    success_rates = [level_stats[l]["success_rate"] * 100 for l in levels]
    detection_rates = [level_stats[l]["perception"]["detection_rate"] * 100 for l in levels]
    planning_rates = [level_stats[l]["planning"]["success_rate"] * 100 for l in levels]

    html += f"""
            <section class="section">
                <h2>📈 System Degradation Curves</h2>
                <div class="plot" id="degradation-plot"></div>
                <script>
                    var trace1 = {{
                        x: {levels},
                        y: {success_rates},
                        mode: 'lines+markers',
                        name: 'System Success',
                        line: {{color: '#667eea', width: 3}},
                        marker: {{size: 10}}
                    }};
                    var trace2 = {{
                        x: {levels},
                        y: {detection_rates},
                        mode: 'lines+markers',
                        name: 'Detection Rate',
                        line: {{color: '#28a745', width: 2, dash: 'dash'}},
                        marker: {{size: 8}}
                    }};
                    var trace3 = {{
                        x: {levels},
                        y: {planning_rates},
                        mode: 'lines+markers',
                        name: 'Planning Success',
                        line: {{color: '#ffc107', width: 2, dash: 'dot'}},
                        marker: {{size: 8}}
                    }};
                    var layout = {{
                        title: 'Performance vs Perturbation Level',
                        xaxis: {{title: 'Perturbation Level', gridcolor: '#e9ecef'}},
                        yaxis: {{title: 'Success Rate (%)', range: [0, 105], gridcolor: '#e9ecef'}},
                        plot_bgcolor: '#ffffff',
                        paper_bgcolor: '#f8f9fa',
                        font: {{family: 'inherit'}},
                        showlegend: true,
                        legend: {{x: 0.7, y: 1}}
                    }};
                    Plotly.newPlot('degradation-plot', [trace1, trace2, trace3], layout, {{responsive: true}});
                </script>
            </section>
"""

    # Detailed statistics table
    html += """
            <section class="section">
                <h2>📋 Detailed Results by Perturbation Level</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Level</th>
                            <th>N</th>
                            <th>Success Rate</th>
                            <th>Detection</th>
                            <th>PnP Success</th>
                            <th>Planning</th>
                            <th>Oscillation</th>
                            <th>Final Distance (m)</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    for level in sorted(level_stats.keys()):
        stats = level_stats[level]
        success_badge = "success" if stats["success_rate"] > 0.8 else ("warning" if stats["success_rate"] > 0.5 else "danger")

        html += f"""
                        <tr>
                            <td><strong>{level:.1f}</strong></td>
                            <td>{stats["n_runs"]}</td>
                            <td><span class="badge badge-{success_badge}">{stats["success_rate"]*100:.1f}%</span></td>
                            <td>{stats["perception"]["detection_rate"]*100:.1f}%</td>
                            <td>{stats["geometry"]["pnp_success_rate"]*100:.1f}%</td>
                            <td>{stats["planning"]["success_rate"]*100:.1f}%</td>
                            <td>{stats["control"]["oscillation_rate"]*100:.1f}%</td>
                            <td>{stats["final_dist_mean"]:.3f} ± {stats["final_dist_std"]:.3f}</td>
                        </tr>
"""

    html += """
                    </tbody>
                </table>
            </section>
"""

    # Module breakdown
    html += """
            <section class="section">
                <h2>🔧 Module-Level Analysis</h2>
                <div class="plot" id="module-plot"></div>
                <script>
"""

    # Prepare module data
    module_data = {
        "Perception": [],
        "Geometry": [],
        "Planning": [],
        "Control": []
    }

    for level in levels:
        stats = level_stats[level]
        module_data["Perception"].append(stats["perception"]["detection_rate"] * 100)
        module_data["Geometry"].append(stats["geometry"]["pnp_success_rate"] * 100)
        module_data["Planning"].append(stats["planning"]["success_rate"] * 100)
        module_data["Control"].append((1 - stats["control"]["oscillation_rate"]) * 100)

    html += f"""
                    var modules = ['Perception', 'Geometry', 'Planning', 'Control'];
                    var colors = ['#667eea', '#28a745', '#ffc107', '#dc3545'];
                    var traces = [];
"""

    for i, (module, values) in enumerate(module_data.items()):
        html += f"""
                    traces.push({{
                        x: {levels},
                        y: {values},
                        mode: 'lines+markers',
                        name: '{module}',
                        line: {{color: colors[{i}], width: 2}},
                        marker: {{size: 8}}
                    }});
"""

    html += """
                    var layout = {
                        title: 'Module Success Rates vs Perturbation Level',
                        xaxis: {title: 'Perturbation Level', gridcolor: '#e9ecef'},
                        yaxis: {title: 'Success Rate (%)', range: [0, 105], gridcolor: '#e9ecef'},
                        plot_bgcolor: '#ffffff',
                        paper_bgcolor: '#f8f9fa',
                        font: {family: 'inherit'}
                    };
                    Plotly.newPlot('module-plot', traces, layout, {responsive: true});
                </script>
            </section>
"""

    # Cross-scenario comparison (if available)
    if cross_scenario_data:
        html += """
            <section class="section">
                <h2>🔀 Cross-Scenario Comparison</h2>
                <p>Robustness ranking across all tested scenarios:</p>
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Scenario</th>
                            <th>Robustness (AUC)</th>
                            <th>Interpretation</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        for rank_info in cross_scenario_data.get("robustness_ranking", []):
            auc = rank_info["auc"]
            interp = "High" if auc > 0.7 else ("Medium" if auc > 0.4 else "Low")
            badge = "success" if auc > 0.7 else ("warning" if auc > 0.4 else "danger")
            html += f"""
                        <tr>
                            <td><strong>#{rank_info["rank"]}</strong></td>
                            <td>{rank_info["scenario"]}</td>
                            <td>{auc:.3f}</td>
                            <td><span class="badge badge-{badge}">{interp}</span></td>
                        </tr>
"""
        html += """
                    </tbody>
                </table>
            </section>
"""

    # Module vulnerability (if available)
    if module_analysis:
        html += """
            <section class="section">
                <h2>⚠️ Module Vulnerability Analysis</h2>
                <p>Failure rates across the perception-planning-control pipeline:</p>
                <table>
                    <thead>
                        <tr>
                            <th>Module</th>
                            <th>Failure Rate</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        for mod_info in module_analysis.get("critical_modules", []):
            rate = mod_info["failure_rate"] * 100
            badge = "success" if rate < 10 else ("warning" if rate < 30 else "danger")
            status = "Robust" if rate < 10 else ("Moderate" if rate < 30 else "Critical")
            html += f"""
                        <tr>
                            <td><strong>{mod_info["module"].title()}</strong></td>
                            <td>{rate:.1f}%</td>
                            <td><span class="badge badge-{badge}">{status}</span></td>
                        </tr>
"""
        html += """
                    </tbody>
                </table>
            </section>
"""

    # Footer
    html += """
        </div>
        <div class="footer">
            <p>Generated by XAI Publication Experiment Suite</p>
            <p>For publication-quality analysis with comprehensive validation</p>
        </div>
    </div>
</body>
</html>
"""

    # Write to file
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"✓ Dashboard created: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate experiment dashboards")
    parser.add_argument(
        "--logs-dir",
        default="results/publication/logs",
        help="Directory containing experiment logs"
    )
    parser.add_argument(
        "--reports-dir",
        default="results/publication/reports",
        help="Directory containing analysis reports"
    )
    parser.add_argument(
        "--output-dir",
        default="results/publication/reports",
        help="Output directory for dashboards"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        help="Specific scenarios to generate dashboards for"
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load cross-scenario analysis if available
    cross_scenario_data = None
    module_analysis = None
    analysis_path = os.path.join(args.reports_dir, "comprehensive_analysis.json")
    if os.path.exists(analysis_path):
        with open(analysis_path, 'r') as f:
            analysis = json.load(f)
            cross_scenario_data = analysis.get("cross_scenario_analysis")
            module_analysis = analysis.get("module_vulnerability_analysis")

    # Determine scenarios
    if args.scenarios:
        scenarios = args.scenarios
    else:
        # Find all scenarios from log files
        scenarios = set()
        for jsonl_file in Path(args.logs_dir).glob("*.jsonl"):
            name = jsonl_file.stem
            if "_level" in name:
                scenario = name.split("_level")[0]
                scenarios.add(scenario)
            elif "_all" in name:
                scenario = name.replace("_all", "")
                scenarios.add(scenario)
        scenarios = list(scenarios)

    print(f"Generating dashboards for {len(scenarios)} scenarios...")

    for scenario in scenarios:
        print(f"\nProcessing {scenario}...")

        # Load results
        results = load_results(args.logs_dir, scenario)

        if not results:
            print(f"  ⚠ No results found for {scenario}")
            continue

        # Generate dashboard
        output_path = os.path.join(args.output_dir, scenario, "dashboard.html")
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

        create_html_dashboard(
            scenario_name=scenario,
            results=results,
            cross_scenario_data=cross_scenario_data,
            module_analysis=module_analysis,
            output_path=output_path
        )

    print(f"\n✓ All dashboards generated in: {args.output_dir}")


if __name__ == "__main__":
    main()
