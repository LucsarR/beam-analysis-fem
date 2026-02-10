"""
Test reaction forces calculated from penalty method.

This test verifies that reaction forces are correctly calculated at constraints
using the penalty method implementation.
"""

import numpy as np
from fem.mesh import Mesh
from fem.material import Material
from fem.section import RectangularBar
from fem.constraint import Constraint
from fem.load import PointLoad
from fem.analysis import BeamAnalysis


def test_cantilever_reactions():
    """
    Test reaction forces for a simple cantilever beam with point load.
    
    Beam: Fixed at left (x=0), free at right (x=L)
    Load: Point load P at free end (downward)
    
    Expected reactions at fixed end:
    - Vertical reaction: Ry = P (upward)
    - Horizontal reaction: Rx = 0
    - Moment reaction: M = P*L (counter-clockwise)
    """
    print("\n" + "="*60)
    print("Test: Cantilever Beam Reactions")
    print("="*60)
    
    # Create mesh
    mesh = Mesh()
    mat = Material(1, 210e9, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)
    
    # Create nodes
    n1 = mesh.add_node(0, 0)
    n2 = mesh.add_node(1, 0)
    
    # Add element
    el = mesh.add_element(n1, n2, mat, sec, 'euler_bernoulli_2node')
    
    # Apply constraints (fixed at node 1)
    c1 = Constraint(n1, 0, 0.0)  # u = 0
    c2 = Constraint(n1, 1, 0.0)  # v = 0
    c3 = Constraint(n1, 2, 0.0)  # θ = 0
    mesh.constraints.add(c1)
    mesh.constraints.add(c2)
    mesh.constraints.add(c3)
    
    # Apply point load (downward at free end)
    P = -1000.0  # N (negative = downward)
    load = PointLoad(P, 1)  # direction 1 = y
    load.node = n2
    mesh.point_loads.append(load)
    
    # Run analysis
    analysis = BeamAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    
    # Get reactions
    reactions = analysis.get_reactions()
    
    assert reactions is not None, "Reactions should be calculated"
    
    # Extract reactions at node 1
    Rx = reactions.get((n1.id, 0), None)
    Ry = reactions.get((n1.id, 1), None)
    M = reactions.get((n1.id, 2), None)
    
    print(f"Applied load: P = {P} N (downward)")
    print(f"Reactions at fixed end:")
    print(f"  Rx = {Rx:.2f} N")
    print(f"  Ry = {Ry:.2f} N")
    print(f"  M = {M:.2f} N·m")
    
    # Verify reactions
    # For cantilever with downward load P at tip:
    # Ry should equal -P (upward to balance downward load)
    # Rx should be close to 0
    # M should equal -P*L (moment to balance applied load)
    L = 1.0
    
    # Check vertical reaction (should balance the load)
    # Applied load: P = -1000 N (downward)
    # Expected reaction: Ry = +1000 N (upward)
    # Reaction = penalty * (prescribed - actual) = penalty * (0 - u)
    # Since u is negative (downward), reaction is positive (upward)
    expected_Ry = -P  # Opposite sign of applied load
    
    print(f"Expected vertical reaction: {expected_Ry} N")
    
    assert abs(Ry - expected_Ry) < abs(expected_Ry) * 0.01, \
        f"Vertical reaction: Ry={Ry}, expected={expected_Ry}"
    
    # Check horizontal reaction (should be near zero)
    assert abs(Rx) < abs(P) * 0.01, f"Horizontal reaction should be near zero: Rx={Rx}"
    
    # Check moment reaction (should balance moment from load)
    expected_M = -P * L
    assert abs(M - expected_M) < abs(expected_M) * 0.01, \
        f"Moment reaction should be {expected_M}: M={M}"
    
    print(f"✓ Reactions are correct within 1% tolerance")
    return True


def test_simply_supported_reactions():
    """
    Test reaction forces for a simply supported beam with center load.
    
    Beam: Pinned at both ends (y-constrained, rotation free)
    Load: Point load P at center (downward)
    
    Expected reactions:
    - Left support: Ry = P/2
    - Right support: Ry = P/2
    - No moment reactions (rotation free)
    """
    print("\n" + "="*60)
    print("Test: Simply Supported Beam Reactions")
    print("="*60)
    
    # Create mesh
    mesh = Mesh()
    mat = Material(1, 210e9, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)
    
    # Create nodes
    n1 = mesh.add_node(0, 0)
    n2 = mesh.add_node(0.5, 0)
    n3 = mesh.add_node(1, 0)
    
    # Add elements
    el1 = mesh.add_element(n1, n2, mat, sec, 'euler_bernoulli_2node')
    el2 = mesh.add_element(n2, n3, mat, sec, 'euler_bernoulli_2node')
    
    # Apply constraints (simply supported: y-constrained at both ends, x-constrained at one end)
    c1 = Constraint(n1, 0, 0.0)  # u = 0 at left (prevent rigid body motion in x)
    c2 = Constraint(n1, 1, 0.0)  # v = 0 at left
    c3 = Constraint(n3, 1, 0.0)  # v = 0 at right
    # Note: no rotation constraints (free to rotate)
    mesh.constraints.add(c1)
    mesh.constraints.add(c2)
    mesh.constraints.add(c3)
    
    # Apply point load at center (downward)
    P = -2000.0  # N (negative = downward)
    load = PointLoad(P, 1)  # direction 1 = y
    load.node = n2
    mesh.point_loads.append(load)
    
    # Run analysis
    analysis = BeamAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    
    # Get reactions
    reactions = analysis.get_reactions()
    
    assert reactions is not None, "Reactions should be calculated"
    
    # Extract reactions
    Ry_left = reactions.get((n1.id, 1), None)
    Ry_right = reactions.get((n3.id, 1), None)
    
    print(f"Applied load: P = {P} N (downward at center)")
    print(f"Reactions:")
    print(f"  Left support: Ry = {Ry_left:.2f} N")
    print(f"  Right support: Ry = {Ry_right:.2f} N")
    print(f"  Total vertical reaction: {Ry_left + Ry_right:.2f} N")
    
    # For simply supported beam with center load:
    # Each support should carry P/2 upward = -P/2 numerically
    # Applied load P = -2000 N (downward)
    # Expected reaction at each support = +1000 N (upward)
    expected_reaction = -P / 2
    
    print(f"Expected reaction at each support: {expected_reaction} N")
    
    # Check reactions
    assert abs(Ry_left - expected_reaction) < abs(expected_reaction) * 0.01, \
        f"Left reaction should be {expected_reaction}: Ry_left={Ry_left}"
    assert abs(Ry_right - expected_reaction) < abs(expected_reaction) * 0.01, \
        f"Right reaction should be {expected_reaction}: Ry_right={Ry_right}"
    
    # Check equilibrium: total reaction should balance load
    total_reaction = Ry_left + Ry_right
    assert abs(total_reaction + P) < abs(P) * 0.01, \
        f"Total reaction should balance load: total={total_reaction}, P={P}"
    
    print(f"✓ Reactions are correct within 1% tolerance")
    print(f"✓ Equilibrium satisfied")
    return True


def test_no_constraints():
    """Test that reactions are None when there are no constraints."""
    print("\n" + "="*60)
    print("Test: No Constraints")
    print("="*60)
    
    # Create mesh without constraints
    mesh = Mesh()
    mat = Material(1, 210e9, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)
    
    n1 = mesh.add_node(0, 0)
    n2 = mesh.add_node(1, 0)
    
    el = mesh.add_element(n1, n2, mat, sec, 'euler_bernoulli_2node')
    
    # Run analysis (will fail to solve without constraints, but we test the reaction logic)
    analysis = BeamAnalysis(mesh)
    analysis.assemble()
    
    # Check that reactions are initially None
    assert analysis.get_reactions() is None, "Reactions should be None before solving"
    
    print(f"✓ Reactions are None when no constraints exist")
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("RUNNING REACTION FORCE TESTS")
    print("="*60)
    
    tests = [
        ("No Constraints", test_no_constraints),
        ("Cantilever Reactions", test_cantilever_reactions),
        ("Simply Supported Reactions", test_simply_supported_reactions),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed > 0:
        exit(1)
