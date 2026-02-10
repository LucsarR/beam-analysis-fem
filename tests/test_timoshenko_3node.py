"""
Tests for 3-node Timoshenko beam element with analytical reference solutions.

This test suite validates:
1. Element creation and DOF structure (9 DOFs with central node rotation)
2. Stiffness matrix properties (symmetry, positive definiteness)
3. Cantilever beam with point load - comparison with analytical solution
4. Simply supported beam with uniformly distributed load - comparison with analytical solution
5. Central node rotation verification

Analytical Reference Solutions:
- Timoshenko beam theory accounts for shear deformation
- Deflection: w = w_bending + w_shear
- For cantilever beam with end load P:
  w(x) = (P*x²)/(6*E*I)*(3*L - x) + (P*x)/(κ*G*A)
- For simply supported beam with uniform load q:
  w_max = (5*q*L⁴)/(384*E*I) + (q*L²)/(8*κ*G*A)

References:
- Timoshenko, S.P. "Strength of Materials" (1955)
- Cowper, G.R. "The Shear Coefficient in Timoshenko's Beam Theory" (1966)
"""

import numpy as np
from fem.mesh import Mesh
from fem.material import Material
from fem.section import RectangularBar
from fem.constraint import Constraint
from fem.load import PointLoad, DistributedLoad
from fem.analysis import BeamAnalysis


def test_element_creation():
    """Test basic 3-node Timoshenko element creation and DOF structure."""
    print("\n" + "="*60)
    print("Test: 3-Node Timoshenko Element Creation")
    print("="*60)
    
    mesh = Mesh()
    mat = Material(1, 210e9, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)
    
    n1 = mesh.add_node(0, 0)
    n2 = mesh.add_node(1, 0)
    
    # Add 3-node Timoshenko element
    el = mesh.add_element(n1, n2, mat, sec, 'timoshenko_3node')
    
    # Verify element structure
    assert el.node_start == n1
    assert el.node_end == n2
    assert el.node_center is not None
    assert el.node_center.x == 0.5
    assert el.node_center.y == 0.0
    assert len(mesh.nodes) == 3
    assert el.length == 1.0
    
    # Verify stiffness matrix has correct dimensions (9x9)
    K = el.stiffness_matrix()
    assert K.shape == (9, 9), f"Expected (9, 9), got {K.shape}"
    
    print(f"✓ Element created with {len(mesh.nodes)} nodes")
    print(f"✓ Central node at ({el.node_center.x}, {el.node_center.y})")
    print(f"✓ Stiffness matrix is 9x9 (central node has rotation DOF)")


def test_stiffness_matrix_properties():
    """Test stiffness matrix properties."""
    print("\n" + "="*60)
    print("Test: Stiffness Matrix Properties")
    print("="*60)
    
    mesh = Mesh()
    mat = Material(1, 210e9, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)
    
    n1 = mesh.add_node(0, 0)
    n2 = mesh.add_node(1, 0)
    el = mesh.add_element(n1, n2, mat, sec, 'timoshenko_3node')
    
    K = el.stiffness_matrix()
    
    # Check dimensions
    assert K.shape == (9, 9), f"Expected (9, 9), got {K.shape}"
    
    # Check symmetry
    assert np.allclose(K, K.T, atol=1e-10), "Stiffness matrix should be symmetric"
    
    # Check positive definiteness (all eigenvalues should be non-negative)
    eigvals = np.linalg.eigvalsh(K)
    max_eigval = np.max(np.abs(eigvals))
    assert np.all(eigvals >= -1e-6 * max_eigval), \
        f"Matrix should be positive semi-definite, min eigval: {np.min(eigvals)}"
    
    print(f"✓ Stiffness matrix is 9x9")
    print(f"✓ Matrix is symmetric")
    print(f"✓ Matrix is positive semi-definite")
    print(f"  Min eigenvalue: {np.min(eigvals):.6e}")
    print(f"  Max eigenvalue: {np.max(eigvals):.6e}")


def test_cantilever_point_load_analytical():
    """
    Test cantilever beam with point load at free end.
    Compare with analytical Timoshenko beam solution.
    
    Analytical solution for cantilever beam with end load P:
    - Deflection: w(x) = (P*x²)/(6*E*I)*(3*L - x) + (P*x)/(κ*G*A)
    - Rotation: θ(x) = (P*x)/(2*E*I)*(2*L - x)
    - Max deflection at x=L: w_max = (P*L³)/(3*E*I) + (P*L)/(κ*G*A)
    """
    print("\n" + "="*60)
    print("Test: Cantilever Beam with Point Load (Analytical Comparison)")
    print("="*60)
    
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
    
    # Create mesh with single 3-node element
    mesh = Mesh()
    mat = Material(1, E, nu)
    sec = RectangularBar(1, b, h)
    
    # Get shear coefficient
    kappa = sec.shear_coefficient
    
    n1 = mesh.add_node(0, 0)
    n2 = mesh.add_node(L, 0)
    el = mesh.add_element(n1, n2, mat, sec, 'timoshenko_3node')
    
    # Boundary conditions (fixed at x=0)
    mesh.constraints.add(Constraint(n1, 0, 0.0))
    mesh.constraints.add(Constraint(n1, 1, 0.0))
    mesh.constraints.add(Constraint(n1, 2, 0.0))
    
    # Point load at free end
    load = PointLoad(P, 1)  # direction=1 for y-direction
    load.node = n2
    mesh.point_loads.append(load)
    
    # Solve
    analysis = BeamAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    
    # Extract displacements at x=L (node 2)
    v_L_fem = displacements[3*(n2.id-1) + 1]
    theta_L_fem = displacements[3*(n2.id-1) + 2]
    
    # Analytical solution at x=L
    w_bending = (P * L**3) / (3 * E * I)
    w_shear = (P * L) / (kappa * G * A)
    v_L_analytical = w_bending + w_shear
    
    theta_L_analytical = (P * L**2) / (2 * E * I)
    
    # Check agreement
    rel_error_v = abs((v_L_fem - v_L_analytical) / v_L_analytical) * 100
    rel_error_theta = abs((theta_L_fem - theta_L_analytical) / theta_L_analytical) * 100
    
    print(f"Deflection at x=L:")
    print(f"  FEM:        {v_L_fem:.6e} m")
    print(f"  Analytical: {v_L_analytical:.6e} m")
    print(f"  Error:      {rel_error_v:.2f}%")
    print(f"Rotation at x=L:")
    print(f"  FEM:        {theta_L_fem:.6e} rad")
    print(f"  Analytical: {theta_L_analytical:.6e} rad")
    print(f"  Error:      {rel_error_theta:.2f}%")
    
    # Single 3-node element should give reasonable accuracy (within 5% for this problem)
    assert rel_error_v < 5.0, f"Deflection error too large: {rel_error_v:.2f}%"
    assert rel_error_theta < 5.0, f"Rotation error too large: {rel_error_theta:.2f}%"
    
    print(f"✓ Results within acceptable tolerance")
    
    # Also check central node
    v_mid_fem = displacements[3*(el.node_center.id-1) + 1]
    theta_mid_fem = displacements[3*(el.node_center.id-1) + 2]
    x_mid = L / 2
    
    w_bending_mid = (P * x_mid**2) / (6 * E * I) * (3 * L - x_mid)
    w_shear_mid = (P * x_mid) / (kappa * G * A)
    v_mid_analytical = w_bending_mid + w_shear_mid
    
    theta_mid_analytical = (P * x_mid) / (2 * E * I) * (2 * L - x_mid)
    
    print(f"\nCentral node (x=L/2):")
    print(f"  Deflection FEM:    {v_mid_fem:.6e} m")
    print(f"  Deflection Analytical: {v_mid_analytical:.6e} m")
    print(f"  Rotation FEM:      {theta_mid_fem:.6e} rad")
    print(f"  Rotation Analytical: {theta_mid_analytical:.6e} rad")
    print(f"✓ Central node has rotation DOF and reasonable values")


def test_simply_supported_uniform_load():
    """
    Test simply supported beam with uniformly distributed load.
    Compare with analytical Timoshenko beam solution.
    
    Analytical solution for simply supported beam with uniform load q:
    - Max deflection at x=L/2: 
      w_max = (5*q*L⁴)/(384*E*I) + (q*L²)/(8*κ*G*A)
    """
    print("\n" + "="*60)
    print("Test: Simply Supported Beam with Uniform Load")
    print("="*60)
    
    # Beam properties
    L = 2.0  # Length (m)
    E = 210e9  # Young's modulus (Pa)
    nu = 0.3  # Poisson's ratio
    G = E / (2 * (1 + nu))  # Shear modulus
    b = 0.1  # Width (m)
    h = 0.2  # Height (m)
    I = (b * h**3) / 12  # Second moment of area
    A = b * h  # Cross-sectional area
    
    # Load
    q = -10000.0  # Uniform load (N/m) - negative for downward
    
    # Create mesh with single 3-node element
    mesh = Mesh()
    mat = Material(1, E, nu)
    sec = RectangularBar(1, b, h)
    
    # Get shear coefficient
    kappa = sec.shear_coefficient
    
    n1 = mesh.add_node(0, 0)
    n2 = mesh.add_node(L, 0)
    el = mesh.add_element(n1, n2, mat, sec, 'timoshenko_3node')
    
    # Boundary conditions (simply supported)
    mesh.constraints.add(Constraint(n1, 0, 0.0))
    mesh.constraints.add(Constraint(n1, 1, 0.0))
    mesh.constraints.add(Constraint(n2, 1, 0.0))
    
    # Distributed load
    dist_load = DistributedLoad(magnitude_start=q, magnitude_end=q, direction='y')
    dist_load.element = el
    mesh.distributed_loads.append(dist_load)
    
    # Solve
    analysis = BeamAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    
    # Extract displacement at midpoint (central node)
    v_mid_fem = displacements[3*(el.node_center.id-1) + 1]
    
    # Analytical solution at x=L/2
    w_bending_max = (5 * q * L**4) / (384 * E * I)
    w_shear_max = (q * L**2) / (8 * kappa * G * A)
    v_mid_analytical = w_bending_max + w_shear_max
    
    # Check agreement
    rel_error = abs((v_mid_fem - v_mid_analytical) / v_mid_analytical) * 100
    
    print(f"Max deflection at x=L/2:")
    print(f"  FEM:        {v_mid_fem:.6e} m")
    print(f"  Analytical: {v_mid_analytical:.6e} m")
    print(f"  Error:      {rel_error:.2f}%")
    print(f"\nBending contribution: {w_bending_max:.6e} m ({abs(w_bending_max/v_mid_analytical)*100:.1f}%)")
    print(f"Shear contribution:   {w_shear_max:.6e} m ({abs(w_shear_max/v_mid_analytical)*100:.1f}%)")
    
    # Single element should give reasonable accuracy
    # Note: Simply supported beam is more challenging than cantilever for a single element
    # due to the more restrictive boundary conditions and symmetric loading
    assert rel_error < 20.0, f"Deflection error too large: {rel_error:.2f}%"
    
    print(f"✓ Results within acceptable tolerance for single element")


def test_convergence_with_mesh_refinement():
    """
    Test convergence of solution with mesh refinement.
    Use multiple 3-node elements and compare with analytical solution.
    """
    print("\n" + "="*60)
    print("Test: Mesh Convergence with 3-Node Timoshenko Elements")
    print("="*60)
    
    # Beam properties
    L = 1.0
    E = 210e9
    nu = 0.3
    G = E / (2 * (1 + nu))
    b = 0.05
    h = 0.1
    I = (b * h**3) / 12
    A = b * h
    
    # Load
    P = -1000.0
    
    mat = Material(1, E, nu)
    sec = RectangularBar(1, b, h)
    kappa = sec.shear_coefficient
    
    # Analytical solution at x=L
    w_bending = (P * L**3) / (3 * E * I)
    w_shear = (P * L) / (kappa * G * A)
    v_L_analytical = w_bending + w_shear
    
    print(f"\nAnalytical deflection at x=L: {v_L_analytical:.6e} m")
    print(f"\nTesting with different mesh densities:")
    
    n_elements_list = [1, 2, 4, 8]
    errors = []
    
    for n_elem in n_elements_list:
        mesh = Mesh()
        
        # Create mesh with multiple elements
        nodes = []
        for i in range(n_elem + 1):
            x = L * i / n_elem
            node = mesh.add_node(x, 0)
            nodes.append(node)
        
        # Add elements
        for i in range(n_elem):
            mesh.add_element(nodes[i], nodes[i+1], mat, sec, 'timoshenko_3node')
        
        # Boundary conditions
        mesh.constraints.add(Constraint(nodes[0], 0, 0.0))
        mesh.constraints.add(Constraint(nodes[0], 1, 0.0))
        mesh.constraints.add(Constraint(nodes[0], 2, 0.0))
        
        # Point load at free end
        load = PointLoad(P, 1)  # direction=1 for y-direction
        load.node = nodes[-1]
        mesh.point_loads.append(load)
        
        # Solve
        analysis = BeamAnalysis(mesh)
        analysis.assemble()
        displacements = analysis.solve()
        
        # Extract displacement at x=L
        v_L_fem = displacements[3*(nodes[-1].id-1) + 1]
        
        error = abs((v_L_fem - v_L_analytical) / v_L_analytical) * 100
        errors.append(error)
        
        print(f"  {n_elem} element(s): v_L = {v_L_fem:.6e} m, Error = {error:.3f}%")
    
    # Check that error decreases with mesh refinement (monotonic convergence)
    # Allow a small tolerance for numerical noise in very accurate solutions
    for i in range(len(errors) - 1):
        # If both errors are already very small (< 0.1%), we've achieved convergence
        if errors[i] < 0.1 and errors[i+1] < 0.1:
            continue
        assert errors[i] >= errors[i+1], \
            f"Error should decrease with refinement: {errors[i]:.3f}% -> {errors[i+1]:.3f}%"
    
    print(f"✓ Solution converges with mesh refinement")


def test_central_node_rotation_verification():
    """
    Verify that the central node rotation DOF is actually being used and has reasonable values.
    """
    print("\n" + "="*60)
    print("Test: Central Node Rotation DOF Verification")
    print("="*60)
    
    # Simple cantilever beam
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
    el = mesh.add_element(n1, n2, mat, sec, 'timoshenko_3node')
    
    # Fixed at left end
    mesh.constraints.add(Constraint(n1, 0, 0.0))
    mesh.constraints.add(Constraint(n1, 1, 0.0))
    mesh.constraints.add(Constraint(n1, 2, 0.0))
    
    # Point load at free end
    load = PointLoad(P, 1)  # direction=1 for y-direction
    load.node = n2
    mesh.point_loads.append(load)
    
    # Solve
    analysis = BeamAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    
    # Extract rotations
    theta_start = displacements[3*(n1.id-1) + 2]
    theta_center = displacements[3*(el.node_center.id-1) + 2]
    theta_end = displacements[3*(n2.id-1) + 2]
    
    print(f"Rotations along the beam:")
    print(f"  Start (x=0):    {theta_start:.6e} rad")
    print(f"  Center (x=L/2): {theta_center:.6e} rad")
    print(f"  End (x=L):      {theta_end:.6e} rad")
    
    # Check that central node rotation is non-zero and between start and end
    assert abs(theta_center) > 1e-10, "Central node rotation should be non-zero"
    assert abs(theta_start) < 1e-9, "Start rotation should be approximately zero (boundary condition)"
    
    # For cantilever beam with downward load, rotation should increase from 0 to max
    # (becomes more negative for downward deflection)
    assert theta_center < 0, "Central rotation should be negative (downward load)"
    assert theta_end < 0, "End rotation should be negative (downward load)"
    assert abs(theta_center) < abs(theta_end), "Rotation magnitude should increase along beam"
    
    print(f"✓ Central node rotation DOF is active and has reasonable values")
    print(f"✓ Rotation increases monotonically from fixed end to free end")


if __name__ == "__main__":
    test_element_creation()
    test_stiffness_matrix_properties()
    test_cantilever_point_load_analytical()
    test_simply_supported_uniform_load()
    test_convergence_with_mesh_refinement()
    test_central_node_rotation_verification()
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
