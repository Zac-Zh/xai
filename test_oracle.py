#!/usr/bin/env python3
"""
Test script for Robo-Oracle Module 1

Demonstrates the Oracle's causal attribution capabilities across
different scenarios and perturbation levels.
"""
from __future__ import annotations

import sys
from oracle.oracle_interface import Oracle, run_classical_oracle


def test_single_attribution():
    """Test Oracle attribution on a single configuration."""
    print("="*70)
    print("Test 1: Single Oracle Attribution")
    print("="*70)

    success, label, log = run_classical_oracle(
        scenario='occlusion',
        level=0.6,
        seed=42
    )

    print(f"\nConfiguration:")
    print(f"  Scenario: occlusion")
    print(f"  Level: 0.6")
    print(f"  Seed: 42")

    print(f"\nResult:")
    print(f"  Success: {success}")

    if not success:
        print(f"\n Failure Attribution:")
        print(f"  Module: {label.failure_module}")
        print(f"  Reason: {label.failure_reason}")
        if label.threshold_violated:
            print(f"  Threshold: {label.threshold_violated} = {label.threshold_value}")
            print(f"  Actual: {label.actual_value:.3f}")
        print(f"  Severity: {label.severity}")
        print(f"  Recoverable: {label.recoverable}")

        print(f"\n📝 Natural Language Description:")
        print(f"  {label.natural_language_description}")

    print("\n✓ Test 1 passed\n")


def test_multiple_scenarios():
    """Test Oracle across multiple scenarios."""
    print("="*70)
    print("Test 2: Multiple Scenarios")
    print("="*70)

    scenarios = [
        ('occlusion', 0.4),
        ('module_failure', 0.5),
        ('data_corruption', 0.6),
        ('noise_injection', 0.3),
    ]

    oracle = Oracle()

    for scenario, level in scenarios:
        print(f"\nScenario: {scenario}, Level: {level}")

        result = oracle.run_single(scenario, level, seed=123)

        print(f"  Success: {result.success}")
        if not result.success:
            print(f"  Failure: {result.failure_label.failure_module} - {result.failure_label.failure_reason}")

    print("\n✓ Test 2 passed\n")


def test_expert_demonstrations():
    """Test expert demonstration generation."""
    print("="*70)
    print("Test 3: Expert Demonstration Generation")
    print("="*70)

    oracle = Oracle()

    print("\nGenerating 20 expert demonstrations...")
    dataset_path = oracle.generate_expert_demonstrations(
        num_demos=20,
        output_dir="data/test_expert_demos",
        max_perturbation=0.1
    )

    print(f"\n✓ Test 3 passed")
    print(f"  Dataset saved to: {dataset_path}\n")


def test_batch_execution():
    """Test batch Oracle execution."""
    print("="*70)
    print("Test 4: Batch Execution")
    print("="*70)

    oracle = Oracle()

    scenarios = ['occlusion', 'module_failure']
    levels = [0.2, 0.4, 0.6]

    print(f"\nRunning batch execution:")
    print(f"  Scenarios: {scenarios}")
    print(f"  Levels: {levels}")
    print(f"  Runs per condition: 5")

    results = oracle.batch_execute(
        scenarios=scenarios,
        levels=levels,
        runs_per_condition=5
    )

    # Analyze results
    total = len(results)
    failures = sum(1 for r in results if not r.success)
    failure_modules = {}

    for r in results:
        if not r.success and r.failure_label.failure_module:
            module = r.failure_label.failure_module
            failure_modules[module] = failure_modules.get(module, 0) + 1

    print(f"\nResults:")
    print(f"  Total runs: {total}")
    print(f"  Failures: {failures}")
    print(f"  Failure rate: {failures/total*100:.1f}%")

    if failure_modules:
        print(f"\n  Failure distribution:")
        for module, count in sorted(failure_modules.items(), key=lambda x: -x[1]):
            print(f"    {module}: {count} ({count/failures*100:.1f}%)")

    print("\n✓ Test 4 passed\n")


def test_natural_language_generation():
    """Test natural language generation for different failure types."""
    print("="*70)
    print("Test 5: Natural Language Generation")
    print("="*70)

    oracle = Oracle()

    # Test different scenarios to get different failure types
    test_cases = [
        ('occlusion', 0.8, 'High occlusion'),
        ('module_failure', 0.7, 'Module failures'),
        ('data_corruption', 0.6, 'Data corruption'),
    ]

    print("\nGenerating natural language descriptions:\n")

    for scenario, level, description in test_cases:
        result = oracle.run_single(scenario, level, seed=999)

        print(f"Case: {description} ({scenario}, level={level})")
        if not result.success:
            print(f"Description: {result.failure_label.natural_language_description}")
            print()

    print("✓ Test 5 passed\n")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("Robo-Oracle Module 1 Test Suite")
    print("="*70 + "\n")

    try:
        test_single_attribution()
        test_multiple_scenarios()
        test_expert_demonstrations()
        test_batch_execution()
        test_natural_language_generation()

        print("="*70)
        print("✅ All Tests Passed!")
        print("="*70)
        print("\nModule 1 (Oracle) is ready for integration with Module 2.")
        print("Next step: Implement Diffusion Policy (Module 2)\n")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
