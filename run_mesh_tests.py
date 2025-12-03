#!/usr/bin/env python
"""
Run all mesh-related tests to verify mesh convergence and functionality.

This script runs:
1. test_mesh_integration.py - Tests mesh integration with other components
2. test_mesh_convergence.py - Tests mesh convergence with simple structures
3. test_complex_structures.py - Tests mesh with complex structures, angles, and multiple loads
"""

import sys
import subprocess
import os
from pathlib import Path

def run_test(test_file, project_root):
    """Run a test file and return success status."""
    print(f"\n{'='*70}")
    print(f"Running: {test_file}")
    print('='*70)
    
    result = subprocess.run(
        [sys.executable, f"tests/{test_file}"],
        cwd=project_root,
        env={"PYTHONPATH": str(project_root)}
    )
    
    return result.returncode == 0

def main():
    """Run all mesh tests."""
    # Determine project root (directory containing this script)
    project_root = Path(__file__).parent.resolve()
    
    print("\n" + "="*70)
    print("RUNNING ALL MESH TESTS")
    print("="*70)
    
    tests = [
        "test_mesh_integration.py",
        "test_mesh_convergence.py",
        "test_complex_structures.py"
    ]
    
    results = {}
    for test in tests:
        results[test] = run_test(test, project_root)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for test, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test:<35} {status}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n✅ ALL MESH TESTS PASSED!")
        print("\nThe mesh has been verified to:")
        print("  • Correctly generate nodes and elements")
        print("  • Integrate properly with all FEM components")
        print("  • Converge displacement and force solutions with mesh refinement")
        print("  • Work correctly with Euler-Bernoulli and Timoshenko elements")
        print("  • Handle complex structures with multiple loads and constraints")
        print("  • Handle elements at various angles correctly")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
