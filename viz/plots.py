"""Matplotlib offline plots for R2."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_stack_bars(df: pd.DataFrame, out_png: str) -> None:
    """Plot module failure contribution as stacked bars by scenario/level.

    Expects df to have columns: 'scenario', 'level', 'attr_modules' (list), 'attr_errors' (list).
    """
    # Aggregate counts per module
    rows = []
    for _, row in df.iterrows():
        mods = row.get("attr_modules", []) or []
        for m in mods:
            rows.append({"scenario": row["scenario"], "level": row["level"], "module": m})
    agg = pd.DataFrame(rows)
    if agg.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("No failures")
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    counts = agg.groupby(["scenario", "level", "module"]).size().unstack(fill_value=0)
    counts.sort_index(inplace=True)
    counts.plot(kind="bar", stacked=True, figsize=(8, 5))
    plt.ylabel("Failure counts")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_sensitivity(df: pd.DataFrame, out_png: str) -> None:
    """Plot success rate vs level with a key metric curve (avg_conf)."""
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("No data")
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    # Assume single scenario per df input
    agg = df.groupby("level").agg(success_rate=("system.success", lambda x: np.mean(x.astype(bool))),
                                   avg_conf=("perception.avg_conf", "mean")).reset_index()
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(agg["level"], agg["success_rate"], marker="o", label="Success rate")
    ax1.set_xlabel("Level")
    ax1.set_ylabel("Success rate")
    ax2 = ax1.twinx()
    ax2.plot(agg["level"], agg["avg_conf"], marker="x", linestyle="--", label="Avg conf")
    ax2.set_ylabel("Avg conf")
    plt.title("Sensitivity")
    fig.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_sankey(df: pd.DataFrame, out_png: str) -> None:
    """Draw a simple sankey-like horizontal stacked bars per module-error.

    Uses matplotlib only to avoid extra dependencies for PNG export.
    Expects df with 'attr_modules' and 'attr_errors' columns.
    """
    rows = []
    for _, row in df.iterrows():
        mods = row.get("attr_modules", []) or []
        errs = row.get("attr_errors", []) or []
        for m, e in zip(mods, errs):
            rows.append({"module": m, "error": e})
    if not rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("No failures")
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    dfm = pd.DataFrame(rows)
    counts = dfm.groupby(["module", "error"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    left = np.zeros(len(counts))
    for err in counts.columns:
        vals = counts[err].values
        ax.barh(counts.index, vals, left=left, label=err)
        left += vals
    ax.set_xlabel("Counts")
    ax.set_title("Failure → Module → Error (counts)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

