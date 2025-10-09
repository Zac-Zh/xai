
from __future__ import annotations

import os, sys
# Ensure repository root on sys.path for direct script invocation
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
from typing import Any, Dict, List

import pandas as pd

from utils.yload import load as yload
from scripts.run_suite import run_experiment


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _flatten(prefix: str, obj: Any, out: Dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else k, v, out)
    else:
        out[prefix] = obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--levels", nargs="+", required=True, type=float)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--merge_csv", required=True)
    args = ap.parse_args()

    out_dir = os.path.dirname(args.merge_csv)
    os.makedirs(out_dir, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    for lvl in args.levels:
        out_jsonl = os.path.join(out_dir, f"{args.scenario}_{lvl}.jsonl")
        run_experiment(args.cfg, args.scenario, float(lvl), args.thresholds, int(args.runs), out_jsonl)
        rows = _read_jsonl(out_jsonl)
        for r in rows:
            flat: Dict[str, Any] = {}
            _flatten("", r, flat)
            # Expose common top-level columns for convenience
            flat["scenario"] = r["meta"]["scenario"]
            flat["level"] = r["meta"]["level"]
            all_rows.append(flat)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.merge_csv, index=False)


if __name__ == "__main__":
    main()
