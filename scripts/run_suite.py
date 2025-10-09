
from __future__ import annotations

import os, sys
# Ensure repository root on sys.path for direct script invocation
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import importlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from utils.yload import load as yload

from simulators.synth_env import SynthLiftEnv
from vision.detector_stub import detect_target
from vision.segmenter_stub import segment_target
from vision.viz_overlay import overlay_detection
from geometry.pose_pnp_stub import estimate_pose
from planning.rrt_star_fallback import plan
from control.controller_pd import track
from metrics.system_metrics import success as sys_success


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _flatten(d: Dict[str, Any], parent: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent}.{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def run_experiment(cfg_path: str, scenario: str, level: float, thresholds_path: str, runs: int, out_jsonl: str) -> str:
    cfg = yload(cfg_path)
    th = yload(thresholds_path)
    seeds: List[int] = list(cfg.get("seeds", [0]))
    out_dir = cfg.get("paths", {}).get("out_dir", "results")
    art_dir = os.path.join(out_dir, "artifacts")
    log_dir = os.path.join(out_dir, "logs")
    _ensure_dir(art_dir)
    _ensure_dir(log_dir)
    _ensure_dir(os.path.dirname(out_jsonl))

    # Prepare perturbation module
    perturb_mod = importlib.import_module(f"perturb.{scenario}")

    lines: List[str] = []
    for i in range(runs):
        seed = seeds[i % len(seeds)]
        ctx: Dict[str, Any] = {"noise": {}, "obstacles": [], "meta": {"task": cfg["task"], "robot": cfg["robot"], "camera": cfg["camera"]}}
        rng = np.random.default_rng(seed)
        ctx["rng"] = rng
        # Apply perturbation to context
        perturb_mod.apply(ctx, float(level))
        env = SynthLiftEnv(seed=seed, max_steps=int(cfg["max_steps"]), camera=cfg["camera"])
        env.reset()
        state = env.get_state()
        ctx["gt"] = {"target_x": state["target_x"], "target_y": state["target_y"]}

        # Render and run perception
        rgb = env.render_rgb()
        det = detect_target(rgb, ctx)
        seg = segment_target(rgb, ctx)
        ctx.setdefault("perception", {})["seg_iou"] = seg.get("seg_iou")
        annotated = overlay_detection(rgb, det["bbox"], seg["mask"]) if seg["mask"] is not None else overlay_detection(rgb, det["bbox"], None)

        # Save artifacts
        run_tag = f"{scenario}_{level}_seed{seed}"
        rgb_path = os.path.join(art_dir, f"{run_tag}_rgb.png")
        mask_path = os.path.join(art_dir, f"{run_tag}_mask.png")
        path_plot = os.path.join(art_dir, f"{run_tag}_path.png")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(annotated)
            ax.axis('off')
            fig.tight_layout()
            fig.savefig(rgb_path, dpi=150)
            plt.close(fig)
            # mask if available
            if seg["mask"] is not None:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.imshow(seg["mask"], cmap='gray')
                ax.axis('off')
                fig.tight_layout()
                fig.savefig(mask_path, dpi=150)
                plt.close(fig)
            else:
                mask_path = None
        except Exception:
            mask_path = None

        # Geometry
        geom = estimate_pose(det, ctx)
        # Planning
        start = np.array([state["agent_x"], state["agent_y"]], dtype=float)
        goal = np.array([state["target_x"], state["target_y"]], dtype=float)
        plan_out = plan(start, goal, ctx)

        # Control
        ctrl = track(plan_out.get("path", []), ctx)

        # Final distance approximated as distance from last path point to goal +/- tracking error
        if plan_out.get("path"):
            final_pos = np.array(plan_out["path"][-1], dtype=float)
            final_dist = float(np.linalg.norm(final_pos - goal) + (ctrl["track_rmse"] or 0.0))
        else:
            final_dist = float(np.linalg.norm(start - goal))
        sys_ok = sys_success(final_dist, float(th["system"]["success_dist_tau"]))

        # Path plot
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.plot([start[0]], [start[1]], marker='o', label='start')
            ax.plot([goal[0]], [goal[1]], marker='x', label='goal')
            if plan_out.get("path"):
                pts = np.asarray(plan_out["path"], dtype=float)
                ax.plot(pts[:, 0], pts[:, 1], '-')
            ax.legend()
            fig.tight_layout()
            fig.savefig(path_plot, dpi=150)
            plt.close(fig)
        except Exception:
            path_plot = None

        # Build log object per schema
        run_id = f"{_utcnow()}_{seed}"
        log: Dict[str, Any] = {
            "run_id": run_id,
            "meta": {
                "task": cfg["task"],
                "robot": cfg["robot"],
                "seed": int(seed),
                "scenario": str(scenario),
                "level": float(level),
                "steps": int(cfg["max_steps"]),
                "utc_time": _utcnow(),
            },
            "perception": {
                "avg_conf": float(det["avg_conf"]),
                "detected": bool(det["detected"]),
                "seg_iou": None if seg.get("seg_iou") is None else float(seg["seg_iou"]),
                "bbox": [int(x) for x in det["bbox"]],
            },
            "geometry": {
                "pnp_success": bool(geom["pnp_success"]),
                "pnp_rmse": None if geom.get("pnp_rmse") is None else float(geom["pnp_rmse"]),
                "pose_estimate": [float(v) for v in geom["pose_estimate"]],
            },
            "planning": {
                "success": bool(plan_out["success"]),
                "path_cost": None if plan_out.get("path_cost") is None else float(plan_out["path_cost"]),
                "collisions": int(plan_out["collisions"]),
                "planner": "RRTstar",
            },
            "control": {
                "track_rmse": None if ctrl.get("track_rmse") is None else float(ctrl["track_rmse"]),
                "overshoot": None if ctrl.get("overshoot") is None else float(ctrl["overshoot"]),
                "oscillation": bool(ctrl.get("oscillation", False)),
            },
            "system": {
                "success": bool(sys_ok),
                "final_dist_to_goal": float(final_dist),
                "stop_reason": None,
            },
            "artifacts": {"rgb_path": rgb_path, "mask_path": mask_path, "path_plot": path_plot},
        }
        lines.append(json.dumps(log))

    # Write JSONL
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    return out_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--level", required=True, type=float)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run_experiment(args.cfg, args.scenario, float(args.level), args.thresholds, int(args.runs), args.out)


if __name__ == "__main__":
    main()
