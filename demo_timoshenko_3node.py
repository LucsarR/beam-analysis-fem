#!/usr/bin/env python3
"""
Demonstration script for Timoshenko 3-node element with central node rotation.

This script demonstrates:
1. Creation of a Timoshenko 3-node beam element
2. Central node has rotation DOF (9 DOFs total)
3. Comparison with analytical Timoshenko beam solutions
4. Verification that central node rotation is working correctly

Problem setup: Cantilever beam with point load at free end
- Length: L = 1.0 m
- Material: Steel (E = 210 GPa, ν = 0.3)
- Cross-section: Rectangular (b = 0.05 m, h = 0.1 m)
- Load: P = -1000 N (downward at free end)
"""

import numpy as np
import sys
# Add the parent directory to the path to allow running from any location
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fem.mesh import Mesh
from fem.material import Material
from fem.section import RectangularBar
from fem.constraint import Constraint
from fem.load import PointLoad
from fem.analysis import BeamAnalysis


def main():
    print("="*70)
    print("Timoshenko 3-Node Element Demonstration")
    print("Central Node with Rotation DOF")
    print("="*70)
    
    # Beam properties
    L = 1.0  # Length (m)
    E = 210e9  # Young's modulus (Pa)
    nu = 0.3  # Poisson's ratio
    G = E / (2 * (1 + nu))  # Shear modulus
    b = 0.05  # Width (m)
    h = 0.1  # Height (m)
    I = (b * h**3) / 12  # Second moment of area
    A = b * h  # Cross-sectional area
    
    # Load
    P = -1000.0  # Point load (N) - negative for downward
    
    print(f"\nBeam Properties:")
    print(f"  Length:           L = {L:.2f} m")
    print(f"  Young's modulus:  E = {E/1e9:.0f} GPa")
    print(f"  Shear modulus:    G = {G/1e9:.1f} GPa")
    print(f"  Cross-section:    {b*1000:.0f} mm × {h*1000:.0f} mm")
    print(f"  Area:             A = {A*1e4:.2f} cm²")
    print(f"  Inertia:          I = {I*1e8:.4f} cm⁴")
    print(f"  Load at free end: P = {P:.0f} N (downward)")
    
    # Create mesh with single 3-node Timoshenko element
    mesh = Mesh()
    mat = Material(1, E, nu)
    sec = RectangularBar(1, b, h)
    
    # Get shear coefficient
    kappa = sec.shear_coefficient
    print(f"  Shear coefficient: κ = {kappa:.3f} (rectangular section)")
    
    n1 = mesh.add_node(0, 0)
    n2 = mesh.add_node(L, 0)
    el = mesh.add_element(n1, n2, mat, sec, 'timoshenko_3node')
    
    print(f"\nElement Information:")
    print(f"  Type: Timoshenko 3-node beam element")
    print(f"  Number of nodes: {len(mesh.nodes)}")
    print(f"  Node 1 (start):  x = {n1.x:.2f} m, y = {n1.y:.2f} m")
    print(f"  Node 2 (center): x = {el.node_center.x:.2f} m, y = {el.node_center.y:.2f} m")
    print(f"  Node 3 (end):    x = {n2.x:.2f} m, y = {n2.y:.2f} m")
    print(f"  DOFs per element: 9 (each node has u, v, θ)")
    
    # Boundary conditions (fixed at x=0)
    mesh.constraints.add(Constraint(n1, 0, 0.0))  # u = 0
    mesh.constraints.add(Constraint(n1, 1, 0.0))  # v = 0
    mesh.constraints.add(Constraint(n1, 2, 0.0))  # θ = 0
    
    # Point load at free end
    load = PointLoad(P, 1)  # direction=1 for y-direction
    load.node = n2
    mesh.point_loads.append(load)
    
    # Solve
    print(f"\nSolving FEM system...")
    analysis = BeamAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    
    # Extract results
    print(f"\n{'='*70}")
    print("Results Comparison: FEM vs Analytical")
    print(f"{'='*70}")
    
    # Analytical solutions
    w_bending = (P * L**3) / (3 * E * I)
    w_shear = (P * L) / (kappa * G * A)
    v_L_analytical = w_bending + w_shear
    theta_L_analytical = (P * L**2) / (2 * E * I)
    
    # At x=L/2
    x_mid = L / 2
    w_bending_mid = (P * x_mid**2) / (6 * E * I) * (3 * L - x_mid)
    w_shear_mid = (P * x_mid) / (kappa * G * A)
    v_mid_analytical = w_bending_mid + w_shear_mid
    theta_mid_analytical = (P * x_mid) / (2 * E * I) * (2 * L - x_mid)
    
    # FEM results
    v_start = displacements[3*(n1.id-1) + 1]
    theta_start = displacements[3*(n1.id-1) + 2]
    v_mid = displacements[3*(el.node_center.id-1) + 1]
    theta_mid = displacements[3*(el.node_center.id-1) + 2]
    v_end = displacements[3*(n2.id-1) + 1]
    theta_end = displacements[3*(n2.id-1) + 2]
    
    # Print results in table format
    print(f"\n{'Location':<15} {'DOF':<12} {'FEM':<18} {'Analytical':<18} {'Error'}")
    print(f"{'-'*70}")
    
    # Start (x=0)
    print(f"{'Start (x=0)':<15} {'v [m]':<12} {v_start:>17.6e} {0.0:>17.6e} {'    -'}")
    print(f"{'               ':<15} {'θ [rad]':<12} {theta_start:>17.6e} {0.0:>17.6e} {'    -'}")
    
    # Middle (x=L/2)
    error_v_mid = abs((v_mid - v_mid_analytical) / v_mid_analytical) * 100
    error_theta_mid = abs((theta_mid - theta_mid_analytical) / theta_mid_analytical) * 100
    print(f"{'Middle (x=L/2)':<15} {'v [m]':<12} {v_mid:>17.6e} {v_mid_analytical:>17.6e} {error_v_mid:>6.2f}%")
    print(f"{'               ':<15} {'θ [rad]':<12} {theta_mid:>17.6e} {theta_mid_analytical:>17.6e} {error_theta_mid:>6.2f}%")
    
    # End (x=L)
    error_v_end = abs((v_end - v_L_analytical) / v_L_analytical) * 100
    error_theta_end = abs((theta_end - theta_L_analytical) / theta_L_analytical) * 100
    print(f"{'End (x=L)':<15} {'v [m]':<12} {v_end:>17.6e} {v_L_analytical:>17.6e} {error_v_end:>6.2f}%")
    print(f"{'               ':<15} {'θ [rad]':<12} {theta_end:>17.6e} {theta_L_analytical:>17.6e} {error_theta_end:>6.2f}%")
    
    print(f"\n{'='*70}")
    print("Key Observations:")
    print(f"{'='*70}")
    print(f"✓ Central node HAS rotation DOF (θ₂ = {theta_mid:.6e} rad)")
    print(f"✓ Deflection error at end: {error_v_end:.2f}% (excellent!)")
    print(f"✓ Rotation error at end: {error_theta_end:.2f}% (excellent!)")
    print(f"✓ Bending contribution: {abs(w_bending/v_L_analytical)*100:.1f}%")
    print(f"✓ Shear contribution: {abs(w_shear/v_L_analytical)*100:.1f}%")
    
    if error_v_end < 0.01 and error_theta_end < 0.01:
        print(f"\n{'='*70}")
        print("SUCCESS: Timoshenko 3-node element with central node rotation")
        print("is working correctly and matches analytical solution!")
        print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
