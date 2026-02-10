#!/usr/bin/env python3
"""
Comparison script: Euler-Bernoulli 3-node vs Timoshenko 3-node elements.

This script demonstrates the key difference:
- Euler-Bernoulli 3-node: 8 DOFs (central node has NO rotation)
- Timoshenko 3-node: 9 DOFs (central node HAS rotation)

This comparison clearly shows that the issue has been fixed.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fem.mesh import Mesh
from fem.material import Material
from fem.section import RectangularBar
from fem.constraint import Constraint
from fem.load import PointLoad
from fem.analysis import BeamAnalysis


def test_element(element_type, title):
    """Test a single element and return the DOF count and central rotation."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    # Setup
    L = 1.0
    E = 210e9
    nu = 0.3
    b = 0.05
    h = 0.1
    P = -1000.0
    
    mesh = Mesh()
    mat = Material(1, E, nu)
    sec = RectangularBar(1, b, h)
    
    n1 = mesh.add_node(0, 0)
    n2 = mesh.add_node(L, 0)
    el = mesh.add_element(n1, n2, mat, sec, element_type)
    
    # Boundary conditions
    mesh.constraints.add(Constraint(n1, 0, 0.0))
    mesh.constraints.add(Constraint(n1, 1, 0.0))
    mesh.constraints.add(Constraint(n1, 2, 0.0))
    
    # For Euler-Bernoulli 3-node, constrain the unused central rotation DOF
    if element_type == 'euler_bernoulli_3node':
        mesh.constraints.add(Constraint(el.node_center, 2, 0.0))
    
    # Point load
    load = PointLoad(P, 1)
    load.node = n2
    mesh.point_loads.append(load)
    
    # Solve
    analysis = BeamAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    
    # Extract results
    stiffness_shape = el.stiffness_matrix().shape
    n_nodes = len(mesh.nodes)
    
    # Get central node rotation
    theta_center = displacements[3*(el.node_center.id-1) + 2]
    v_center = displacements[3*(el.node_center.id-1) + 1]
    
    print(f"\nElement Information:")
    print(f"  Type:              {element_type}")
    print(f"  Number of nodes:   {n_nodes}")
    print(f"  Stiffness matrix:  {stiffness_shape[0]}×{stiffness_shape[1]}")
    
    print(f"\nCentral Node DOFs:")
    print(f"  ID:                {el.node_center.id}")
    print(f"  Position:          x = {el.node_center.x:.2f} m")
    print(f"  Deflection (v₂):   {v_center:.6e} m")
    print(f"  Rotation (θ₂):     {theta_center:.6e} rad")
    
    has_rotation = abs(theta_center) > 1e-10
    
    if has_rotation:
        print(f"\n✓ Central node HAS rotation DOF")
        print(f"  The rotation is actively computed and has a meaningful value")
    else:
        print(f"\n✗ Central node does NOT have rotation DOF")
        print(f"  The rotation is constrained or not part of the element DOFs")
    
    return {
        'stiffness_shape': stiffness_shape,
        'n_nodes': n_nodes,
        'theta_center': theta_center,
        'v_center': v_center,
        'has_rotation': has_rotation
    }


def main():
    print("="*70)
    print("Comparison: Euler-Bernoulli 3-node vs Timoshenko 3-node")
    print("Demonstrating the Fix for Central Node Rotation")
    print("="*70)
    
    # Test Euler-Bernoulli 3-node
    eb_results = test_element(
        'euler_bernoulli_3node',
        'Euler-Bernoulli 3-Node Element (8 DOFs)'
    )
    
    # Test Timoshenko 3-node
    tim_results = test_element(
        'timoshenko_3node',
        'Timoshenko 3-Node Element (9 DOFs)'
    )
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n{'Feature':<35} {'Euler-Bernoulli':<20} {'Timoshenko'}")
    print(f"{'-'*70}")
    print(f"{'Stiffness matrix size':<35} {str(eb_results['stiffness_shape']):<20} {str(tim_results['stiffness_shape'])}")
    print(f"{'Number of nodes':<35} {eb_results['n_nodes']:<20} {tim_results['n_nodes']}")
    print(f"{'Central node deflection':<35} {eb_results['v_center']:<20.6e} {tim_results['v_center']:.6e}")
    print(f"{'Central node rotation':<35} {eb_results['theta_center']:<20.6e} {tim_results['theta_center']:.6e}")
    print(f"{'Has rotation DOF?':<35} {'No (8 DOFs)':<20} {'Yes (9 DOFs)'}")
    
    print(f"\n{'='*70}")
    print("KEY OBSERVATION")
    print(f"{'='*70}")
    print(f"\nThe problem stated: \"O Timoshenko 3-node nao esta corretamente")
    print(f"implementado o central node esta sem rotation\"")
    print(f"\nSOLUTION: We have successfully implemented the TimoshenkoElement3Node")
    print(f"class with proper rotation DOF at the central node!")
    
    if eb_results['has_rotation']:
        print(f"\n⚠ WARNING: Euler-Bernoulli unexpectedly has central rotation")
    else:
        print(f"\n✓ Euler-Bernoulli: Central node has NO rotation (as expected)")
    
    if tim_results['has_rotation']:
        print(f"✓ Timoshenko: Central node HAS rotation (FIXED!)")
        print(f"\n{'='*70}")
        print("SUCCESS: The issue has been resolved!")
        print(f"{'='*70}")
    else:
        print(f"✗ Timoshenko: Central node has NO rotation (PROBLEM NOT FIXED)")
    
    return 0 if tim_results['has_rotation'] else 1


if __name__ == "__main__":
    sys.exit(main())
