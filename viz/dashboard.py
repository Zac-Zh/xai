"""Offline Plotly dashboard with summary plots and runs table."""
from __future__ import annotations

import os
from typing import List

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot as plot_offline


def _fig_stack_bars(df: pd.DataFrame) -> go.Figure:
    rows = []
    for _, row in df.iterrows():
        for m in (row.get("attr_modules", []) or []):
            rows.append({"scenario": row["scenario"], "level": row["level"], "module": m})
    agg = pd.DataFrame(rows)
    if agg.empty:
        return go.Figure()
    counts = agg.groupby(["level", "module"]).size().unstack(fill_value=0)
    fig = go.Figure()
    for module in counts.columns:
        fig.add_trace(go.Bar(name=str(module), x=counts.index.astype(str), y=counts[module]))
    fig.update_layout(barmode="stack", title="Module Failure Contribution", xaxis_title="Level", yaxis_title="Counts")
    return fig


def _fig_sensitivity(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    agg = df.groupby("level").agg(success_rate=("system.success", lambda x: x.astype(bool).mean()),
                                   avg_conf=("perception.avg_conf", "mean")).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=agg["level"], y=agg["success_rate"], mode="lines+markers", name="Success rate"))
    fig.add_trace(go.Scatter(x=agg["level"], y=agg["avg_conf"], mode="lines+markers", name="Avg conf", yaxis="y2"))
    fig.update_layout(title="Sensitivity", xaxis_title="Level",
                      yaxis=dict(title="Success rate", range=[0, 1]),
                      yaxis2=dict(title="Avg conf", overlaying="y", side="right"))
    return fig


def _fig_sankey(df: pd.DataFrame) -> go.Figure:
    # Build Sankey counts Module->Error->Root
    rows = []
    for _, row in df.iterrows():
        mods = row.get("attr_modules", []) or []
        errs = row.get("attr_errors", []) or []
        root = row.get("root_cause", "Unknown")
        for m, e in zip(mods, errs):
            rows.append({"module": m, "error": e, "root": root})
    if not rows:
        return go.Figure()
    dfm = pd.DataFrame(rows)
    counts = dfm.groupby(["module", "error", "root"]).size().reset_index(name="value")
    # Create node list
    modules = sorted(counts["module"].unique().tolist())
    errors = sorted(counts["error"].unique().tolist())
    roots = sorted(counts["root"].unique().tolist())
    labels = modules + errors + roots
    idx = {lab: i for i, lab in enumerate(labels)}
    # links: module->error and error->root
    src, tgt, val = [], [], []
    for _, r in counts.iterrows():
        src.append(idx[r["module"]])
        tgt.append(idx[r["error"]])
        val.append(r["value"])
    # aggregate error->root
    er = dfm.groupby(["error", "root"]).size().reset_index(name="value")
    for _, r in er.iterrows():
        src.append(idx[r["error"]])
        tgt.append(idx[r["root"]])
        val.append(r["value"])
    fig = go.Figure(data=[go.Sankey(node=dict(label=labels), link=dict(source=src, target=tgt, value=val))])
    fig.update_layout(title="Failure → Module → Error → Root Cause")
    return fig


def export_html(df: pd.DataFrame, out_dir: str) -> None:
    """Export an offline Plotly HTML dashboard with three plots and a runs table.

    The HTML file is written to out_dir/index.html.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Assume single scenario per df chunk is passed; otherwise group outside.
    fig1 = _fig_stack_bars(df)
    fig2 = _fig_sensitivity(df)
    fig3 = _fig_sankey(df)

    parts: List[str] = []
    parts.append("<h1>R2 Dashboard</h1>")
    parts.append(plot_offline(fig1, include_plotlyjs=True, output_type="div"))
    parts.append(plot_offline(fig2, include_plotlyjs=False, output_type="div"))
    parts.append(plot_offline(fig3, include_plotlyjs=False, output_type="div"))

    # Runs table
    cols = [
        "run_id",
        "scenario",
        "level",
        "system.success",
        "perception.avg_conf",
        "perception.seg_iou",
        "geometry.pnp_rmse",
        "planning.path_cost",
        "control.track_rmse",
        "artifacts.rgb_path",
        "artifacts.mask_path",
        "artifacts.path_plot",
    ]
    # Flatten df for table display
    fdf = df.copy()
    # Ensure needed columns exist
    for c in cols:
        if c not in fdf.columns:
            fdf[c] = None
    parts.append("<h2>Runs</h2>")
    parts.append(fdf[cols].to_html(index=False, escape=False))

    html = "\n".join(parts)
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

