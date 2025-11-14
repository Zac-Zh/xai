#!/usr/bin/env python3
"""
Code validation script - checks structure without requiring all dependencies.
"""

import os
import sys
import ast
import argparse

def check_file_exists(path):
    """Check if file exists."""
    return os.path.exists(path)

def check_function_exists(filepath, function_name):
    """Check if function exists in Python file."""
    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    return True
        return False
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return False

def check_argparse_args(filepath, expected_args):
    """Check if argparse has expected arguments."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            for arg in expected_args:
                if f'"{arg}"' not in content and f"'{arg}'" not in content:
                    return False, arg
        return True, None
    except Exception as e:
        return False, f"Error: {e}"

def main():
    print("=" * 70)
    print("  R2 CODE VALIDATION")
    print("=" * 70)
    print()

    errors = []
    warnings = []

    # Check 1: Core files exist
    print("1. Checking core files...")
    core_files = [
        "simulators/synth_env.py",
        "simulators/pybullet_env.py",
        "vision/detector_stub.py",
        "vision/detector_real.py",
        "vision/segmenter_stub.py",
        "vision/segmenter_real.py",
        "geometry/pose_pnp_stub.py",
        "geometry/pose_pnp_real.py",
        "planning/rrt_star_fallback.py",
        "planning/rrt_star_real.py",
        "control/controller_pd.py",
        "control/controller_real.py",
        "scripts/run_all.py",
        "scripts/run_unified_experiments.py",
        "run_all.sh",
    ]

    for filepath in core_files:
        if check_file_exists(filepath):
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} - MISSING")
            errors.append(f"Missing file: {filepath}")

    # Check 2: __init__.py files
    print("\n2. Checking __init__.py files...")
    modules = ["vision", "geometry", "planning", "control", "simulators", "perturb", "attribution", "metrics", "viz", "utils"]
    for module in modules:
        init_file = f"{module}/__init__.py"
        if check_file_exists(init_file):
            print(f"  ✓ {module}/__init__.py")
        else:
            print(f"  ✗ {module}/__init__.py - MISSING")
            warnings.append(f"Missing {init_file} (may cause import issues)")

    # Check 3: run_all.py argparse arguments
    print("\n3. Checking run_all.py arguments...")
    expected_args = ["cfg", "thresholds", "perturbations", "runs", "results_dir", "skip_validation"]
    has_args, missing = check_argparse_args("scripts/run_all.py", expected_args)
    if has_args:
        print(f"  ✓ All expected arguments present")
    else:
        print(f"  ✗ Missing argument: {missing}")
        errors.append(f"run_all.py missing argument: {missing}")

    # Check 4: run_all.sh doesn't pass invalid arguments
    print("\n4. Checking run_all.sh...")
    with open("run_all.sh", 'r') as f:
        content = f.read()
        invalid_args = ["--mode", "--use_real_models"]
        found_invalid = []
        for arg in invalid_args:
            if arg in content and "run_all.py" in content:
                # Check if it's actually passed to run_all.py
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if "run_all.py" in line:
                        # Check next few lines
                        chunk = '\n'.join(lines[i:min(i+10, len(lines))])
                        if arg in chunk and not chunk.strip().startswith('#'):
                            found_invalid.append(arg)
                            break

        if not found_invalid:
            print(f"  ✓ No invalid arguments passed to run_all.py")
        else:
            print(f"  ✗ Invalid arguments found: {found_invalid}")
            errors.append(f"run_all.sh passes invalid arguments: {found_invalid}")

    # Check 5: Key functions exist
    print("\n5. Checking key functions...")
    function_checks = [
        ("vision/detector_stub.py", "detect"),
        ("vision/segmenter_stub.py", "segment"),
        ("geometry/pose_pnp_stub.py", "estimate_pose"),
        ("planning/rrt_star_fallback.py", "plan_path"),
        ("control/controller_pd.py", "track_path"),
    ]

    for filepath, func_name in function_checks:
        if check_file_exists(filepath):
            if check_function_exists(filepath, func_name):
                print(f"  ✓ {filepath}::{func_name}()")
            else:
                print(f"  ✗ {filepath}::{func_name}() - NOT FOUND")
                errors.append(f"Missing function {func_name} in {filepath}")
        else:
            print(f"  ⊘ {filepath} - file missing, skipping function check")

    # Summary
    print()
    print("=" * 70)
    print("  VALIDATION SUMMARY")
    print("=" * 70)
    print()
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    print()

    if errors:
        print("❌ CRITICAL ERRORS:")
        for error in errors:
            print(f"  - {error}")
        print()

    if warnings:
        print("⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
        print()

    if not errors and not warnings:
        print("✅ ALL CHECKS PASSED!")
        print()
        print("The code structure is valid. To run experiments:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run fast mode: bash run_all.sh fast")
        print("  3. Run full mode: bash run_all.sh full")
        return 0
    elif not errors:
        print("✅ NO CRITICAL ERRORS")
        print("⚠️  Some warnings present, but code should run")
        return 0
    else:
        print("❌ CRITICAL ERRORS FOUND - CODE WILL NOT RUN")
        return 1

if __name__ == "__main__":
    sys.exit(main())
