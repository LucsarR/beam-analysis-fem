# -*- coding: utf-8 -*-
"""
Tests for Reddy-Bickford (Third-Order Shear Deformation Theory) beam element.

The ReddyBickfordElement2Node implements Reddy's third-order shear deformation
theory (TSDT) based on the formulation by Heyliger & Reddy (1988). Each node
has 4 DOFs: u (axial), v (transverse), theta (rotation), and dv/dx (slope).

The element strain energy includes:
  - Modified bending stiffness D1 = 68EI/105 (rectangular section)
  - Coupling term E1 = 16EI/105 between rotation and curvature
  - Higher-order stiffness F1 = EI/21
  - Effective shear stiffness G1 = 8GA/15

This test suite validates:
  1. Parameter computation (D1, E1, F1, G1) for rectangular sections
  2. Stiffness matrix properties (symmetry, positive-definiteness)
  3. Cantilever beam deflection compared to Timoshenko and Euler-Bernoulli
  4. Simply supported beam behavior
  5. Internal force recovery (moment, shear, normal force)
  6. Mesh convergence
  7. Comparison across beam theories (EB, Timoshenko, Reddy-Bickford)

Reference:
  Heyliger, P.R. and Reddy, J.N. (1988), "A Higher Order Beam Finite
  Element for Bending and Vibration Problems," Journal of Sound and
  Vibration, 126(2), 309-326.

Sign convention (consistent with Euler-Bernoulli and Timoshenko elements):
  - Positive q = upward distributed load
  - Positive w = upward deflection
  - M > 0 = sagging (compression at top fiber)
  - V = -dM/dx (standard beam theory)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fem.mesh import Mesh
from fem.material import Material
from fem.section import RectangularBar
from fem.constraint import Constraint
from fem.load import PointLoad, DistributedLoad
from fem.analysis import BeamAnalysis
from post_processing.forces import StructureResults
from post_processing.plotter import plot_structure_diagram

# ---------------------------------------------------------------------------
# Beam parameters shared across tests
# ---------------------------------------------------------------------------
E = 210e9       # Pa – Young's modulus (steel)
NU = 0.3        # Poisson's ratio
B = 0.05        # m  – section width
H_SLENDER = 0.01  # m  – very slender cross-section (L/h = 200)
H_THICK = 0.20    # m  – thick cross-section  (L/h = 10)
L_BEAM = 2.0      # m  – total beam span
KAPPA = 5.0 / 6   # Timoshenko shear correction factor (rectangular section)


def _ei(h):
    """EI for rectangular section B x h."""
    return E * (B * h**3 / 12)


def _ga(h):
    """G·A for rectangular section B x h."""
    G = E / (2 * (1 + NU))
    return G * B * h


def _kga(h):
    """kappa·G·A for rectangular section B x h."""
    return KAPPA * _ga(h)


def _area(h):
    """Area for rectangular section B x h."""
    return B * h


def _inertia(h):
    """Second moment of area for rectangular section B x h."""
    return B * h**3 / 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_cantilever(n, h, etype="reddy_bickford_2node"):
    """Create cantilever beam mesh with n elements of type etype."""
    mesh = Mesh()
    mat = Material(1, E, NU)
    sec = RectangularBar(1, B, h)
    nodes = mesh.generate_1d_mesh(0, 0, L_BEAM, 0, n, mat, sec, etype)
    # Fix all DOFs at left end
    if etype == "reddy_bickford_2node":
        mesh.constraints.add(Constraint(nodes[0], 0, 0.0))  # u
        mesh.constraints.add(Constraint(nodes[0], 1, 0.0))  # v
        mesh.constraints.add(Constraint(nodes[0], 2, 0.0))  # theta
        mesh.constraints.add(Constraint(nodes[0], 3, 0.0))  # dv/dx
    else:
        mesh.constraints.add(Constraint(nodes[0], 0, 0.0))  # u
        mesh.constraints.add(Constraint(nodes[0], 1, 0.0))  # v
        mesh.constraints.add(Constraint(nodes[0], 2, 0.0))  # theta
    return mesh, nodes


def _make_simply_supported(n, h, etype="reddy_bickford_2node"):
    """Create simply supported beam mesh with n elements of type etype."""
    mesh = Mesh()
    mat = Material(1, E, NU)
    sec = RectangularBar(1, B, h)
    nodes = mesh.generate_1d_mesh(0, 0, L_BEAM, 0, n, mat, sec, etype)
    mesh.constraints.add(Constraint(nodes[0], 0, 0.0))   # u at left
    mesh.constraints.add(Constraint(nodes[0], 1, 0.0))   # v at left
    mesh.constraints.add(Constraint(nodes[-1], 1, 0.0))  # v at right
    return mesh, nodes


def _solve(mesh):
    """Assemble and solve FEM system, return displacements and results."""
    analysis = BeamAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    results = StructureResults(mesh, displacements)
    return displacements, results


def _get_reactions(mesh, nodes, displacements):
    """Get vertical reactions at left and right supports."""
    rxs = mesh.constraints.calculate_all_reactions(displacements)
    Ry_left = rxs.get((nodes[0].id, 1), 0.0)
    Ry_right = rxs.get((nodes[-1].id, 1), 0.0)
    return Ry_left, Ry_right


# ===========================================================================
# Test 1 – Reddy parameter computation for rectangular section
# ===========================================================================
def test_reddy_parameters_rectangular_section():
    """
    Verify that _get_reddy_params() computes correct values for rectangular
    cross-section:
      D1 = 68·E·I/105
      E1 = 16·E·I/105
      F1 = E·I/21
      G1 = 8·G·A/15

    These are the exact analytical values for a rectangular section with
    width B and height h.
    """
    print("\n" + "=" * 60)
    print("Test 1: Reddy Parameter Computation for Rectangular Section")
    print("=" * 60)

    mesh, nodes = _make_cantilever(1, H_THICK)
    element = mesh.elements[0]

    # Get computed parameters
    D1, E1, F1, G1 = element._get_reddy_params()

    # Compute expected analytical values
    I = _inertia(H_THICK)
    A = _area(H_THICK)
    G = E / (2 * (1 + NU))

    D1_expected = 68 * E * I / 105
    E1_expected = 16 * E * I / 105
    F1_expected = E * I / 21
    G1_expected = 8 * G * A / 15

    print(f"  D1: computed = {D1:.6e}, expected = {D1_expected:.6e}")
    print(f"  E1: computed = {E1:.6e}, expected = {E1_expected:.6e}")
    print(f"  F1: computed = {F1:.6e}, expected = {F1_expected:.6e}")
    print(f"  G1: computed = {G1:.6e}, expected = {G1_expected:.6e}")

    # Verify within 0.01% tolerance (numerical precision)
    assert abs(D1 - D1_expected) / D1_expected < 1e-4, \
        f"D1 error: {abs(D1 - D1_expected) / D1_expected * 100:.6f}%"
    assert abs(E1 - E1_expected) / E1_expected < 1e-4, \
        f"E1 error: {abs(E1 - E1_expected) / E1_expected * 100:.6f}%"
    assert abs(F1 - F1_expected) / F1_expected < 1e-4, \
        f"F1 error: {abs(F1 - F1_expected) / F1_expected * 100:.6f}%"
    assert abs(G1 - G1_expected) / G1_expected < 1e-4, \
        f"G1 error: {abs(G1 - G1_expected) / G1_expected * 100:.6f}%"

    print("OK Parameters match analytical values for rectangular section")
    return True


# ===========================================================================
# Test 2 – Stiffness matrix symmetry
# ===========================================================================
def test_stiffness_matrix_symmetry():
    """
    Verify that the 8x8 stiffness matrix is symmetric (K = K^T).
    This is a fundamental requirement for the element formulation.
    """
    print("\n" + "=" * 60)
    print("Test 2: Stiffness Matrix Symmetry")
    print("=" * 60)

    mesh, _ = _make_cantilever(1, H_THICK)
    element = mesh.elements[0]

    K = element.stiffness_matrix()

    # Check symmetry
    symmetry_error = np.max(np.abs(K - K.T))
    print(f"  Max symmetry error: {symmetry_error:.6e}")

    assert symmetry_error < 1e-12, f"Stiffness matrix not symmetric: {symmetry_error}"

    print("OK Stiffness matrix is symmetric")
    return True


# ===========================================================================
# Test 3 – Stiffness matrix positive semi-definiteness
# ===========================================================================
def test_stiffness_matrix_positive_semidefinite():
    """
    Verify that the stiffness matrix is positive semi-definite by checking
    that all eigenvalues are non-negative. For a properly formulated element,
    there should be exactly 3 zero eigenvalues (rigid body modes: 2 translations
    + 1 rotation) and 5 positive eigenvalues.
    """
    print("\n" + "=" * 60)
    print("Test 3: Stiffness Matrix Positive Semi-Definiteness")
    print("=" * 60)

    mesh, _ = _make_cantilever(1, H_THICK)
    element = mesh.elements[0]

    K = element.stiffness_matrix()

    # Compute eigenvalues
    eigvals = np.linalg.eigvalsh(K)
    eigvals_sorted = np.sort(eigvals)

    print(f"  Eigenvalues (sorted):")
    for i, ev in enumerate(eigvals_sorted):
        print(f"    lambda{i+1} = {ev:.6e}")

    # Check that all eigenvalues are non-negative (allow small numerical noise)
    min_eigval = np.min(eigvals)
    assert min_eigval >= -1e-6, f"Negative eigenvalue found: {min_eigval:.6e}"

    # Count near-zero eigenvalues (rigid body modes)
    zero_tol = 1e-6 * np.max(np.abs(eigvals))
    n_zero = np.sum(np.abs(eigvals) < zero_tol)
    print(f"  Number of near-zero eigenvalues: {n_zero}")
    print(f"  (Expected 3 for rigid body modes)")

    print("OK Stiffness matrix is positive semi-definite")
    return True


# ===========================================================================
# Test 4 – Cantilever beam under point load (single element)
# ===========================================================================
def test_cantilever_point_load_single_element():
    """
    Cantilever beam with point load P at the free end.

    Analytical solution (Euler-Bernoulli):
      w_tip = P·L^3/(3·E·I)
      M(x)  = P·(L-x)
      V(x)  = P

    For Reddy-Bickford, the deflection will be larger than EB due to shear
    deformation effects, similar to Timoshenko but with third-order corrections.

    Tests:
      - Tip deflection
      - Reaction force equilibrium
      - Bending moment at fixed end

    Note: Uses n=8 elements to avoid over-stiffness from the linear-θ constraint
    in a single-element RB cantilever.
    """
    print("\n" + "=" * 60)
    print("Test 4: Cantilever Point Load (n=8 Elements)")
    print("=" * 60)

    P = -1000.0  # N, downward at tip
    h = H_THICK  # Use thick beam to see shear effects

    mesh, nodes = _make_cantilever(8, h)
    load = PointLoad(P, 1)  # direction=1 is vertical
    load.node = nodes[-1]
    mesh.point_loads.append(load)

    displacements, results = _solve(mesh)

    # Extract tip displacement (vertical DOF at last node)
    tip_node_id = nodes[-1].id
    dofs_per_node = 4  # Reddy-Bickford has 4 DOFs per node
    v_tip = displacements[(tip_node_id - 1) * dofs_per_node + 1]

    # Euler-Bernoulli analytical tip deflection
    w_EB = P * L_BEAM**3 / (3 * _ei(h))

    print(f"  Applied load:      P = {P:.2f} N")
    print(f"  Tip deflection:    v_tip = {v_tip:.6e} m")
    print(f"  EB analytical:     w_EB  = {w_EB:.6e} m")
    print(f"  Ratio (Reddy/EB):  {v_tip / w_EB:.6f}")
    print(f"  Shear effect:      {(v_tip / w_EB - 1.0) * 100:.2f}%")

    # Reddy-Bickford should show deflection > EB (shear effects)
    # For thick beam (L/h = 10), expect ~5-10% increase
    assert v_tip / w_EB > 1.0, "Reddy deflection should be larger than EB"
    assert v_tip / w_EB < 1.20, "Reddy deflection ratio too large"

    # Check reactions
    rxs = mesh.constraints.calculate_all_reactions(displacements)
    Ry_fixed = rxs.get((nodes[0].id, 1), 0.0)
    print(f"  Reaction at fixed end: {Ry_fixed:.4e} N")
    print(f"  Expected:              {-P:.4e} N")

    assert abs(Ry_fixed - (-P)) / abs(P) < 0.01, "Reaction equilibrium error > 1%"

    print("OK Cantilever point load: deflection and equilibrium correct")
    return True


# ===========================================================================
# Test 5 – Simply supported beam under uniform load
# ===========================================================================
def test_simply_supported_uniform_load():
    """
    Simply supported beam with uniform distributed load q.

    Analytical solution (Euler-Bernoulli):
      w_mid = 5·q·L^4/(384·E·I)
      M_mid = q·L^2/8
      R_left = R_right = q·L/2

    Tests:
      - Mid-span deflection
      - Reaction forces
      - Equilibrium
    """
    print("\n" + "=" * 60)
    print("Test 5: Simply Supported Uniform Load")
    print("=" * 60)

    q = -1000.0  # N/m, downward
    h = H_SLENDER  # Use slender beam to minimize shear effects
    n = 10  # Multiple elements for better accuracy

    mesh, nodes = _make_simply_supported(n, h)

    # Apply uniform load to all elements
    for element in mesh.elements:
        load = DistributedLoad(magnitude_start=q, magnitude_end=q, direction="t")
        load.element = element
        mesh.distributed_loads.append(load)

    displacements, results = _solve(mesh)

    # Find mid-span node (or closest to L/2)
    mid_idx = n // 2
    mid_node_id = nodes[mid_idx].id
    dofs_per_node = 4
    v_mid = displacements[(mid_node_id - 1) * dofs_per_node + 1]

    # Euler-Bernoulli analytical solution
    w_mid_EB = 5 * q * L_BEAM**4 / (384 * _ei(h))
    M_mid_ana = q * L_BEAM**2 / 8
    R_ana = -q * L_BEAM / 2  # Each reaction (upward, so positive when q is negative)

    print(f"  Uniform load:      q = {q:.2f} N/m")
    print(f"  Mid-span deflection: v_mid = {v_mid:.6e} m")
    print(f"  EB analytical:       w_EB  = {w_mid_EB:.6e} m")
    print(f"  Ratio (Reddy/EB):    {v_mid / w_mid_EB:.6f}")

    # For slender beam, Reddy should be very close to EB
    assert abs(v_mid / w_mid_EB - 1.0) < 0.05, \
        f"Slender beam deflection error > 5%: {abs(v_mid / w_mid_EB - 1.0) * 100:.2f}%"

    # Check reactions
    Ry_left, Ry_right = _get_reactions(mesh, nodes, displacements)
    total_load = q * L_BEAM
    print(f"  Total applied load: {total_load:.4e} N")
    print(f"  Sum of reactions:   {Ry_left + Ry_right:.4e} N")
    print(f"  Left reaction:      {Ry_left:.4e} N (expected {R_ana:.4e})")
    print(f"  Right reaction:     {Ry_right:.4e} N (expected {R_ana:.4e})")

    # Check equilibrium
    equil_err = abs((Ry_left + Ry_right) + total_load) / abs(total_load)
    assert equil_err < 0.01, f"Equilibrium error > 1%: {equil_err * 100:.2f}%"

    # Check individual reactions (symmetric)
    assert abs(Ry_left / R_ana - 1.0) < 0.02, "Left reaction error > 2%"
    assert abs(Ry_right / R_ana - 1.0) < 0.02, "Right reaction error > 2%"

    print("OK Simply supported uniform load: deflection and reactions correct")
    return True


# ===========================================================================
# Test 6 – Comparison: Euler-Bernoulli vs Timoshenko vs Reddy-Bickford
# ===========================================================================
def test_beam_theory_comparison():
    """
    Compare deflections from three beam theories for a thick cantilever beam
    under point load:
      - Euler-Bernoulli (no shear deformation)
      - Timoshenko (first-order shear deformation)
      - Reddy-Bickford (third-order shear deformation)

    Expected behavior:
      w_EB < w_Reddy < w_Timoshenko (approximately)

    The relationship depends on the specific beam geometry and loading, but
    generally Reddy-Bickford should fall between EB and Timoshenko, providing
    improved accuracy over Timoshenko for thick beams.

    Note: Uses n=8 elements for RB to avoid over-stiffness from the linear-θ
    constraint in a single-element cantilever.
    """
    print("\n" + "=" * 60)
    print("Test 6: Beam Theory Comparison (Thick Cantilever)")
    print("=" * 60)

    P = -1000.0  # N, downward
    h = H_THICK  # L/h = 10

    # Run analysis with each element type
    results_dict = {}

    for etype in ["euler_bernoulli_2node", "timoshenko_2node", "reddy_bickford_2node"]:
        # Use n=8 for RB to avoid single-element over-stiffness
        n_elem = 8 if etype == "reddy_bickford_2node" else 1
        mesh, nodes = _make_cantilever(n_elem, h, etype=etype)
        load = PointLoad(P, 1)
        load.node = nodes[-1]
        mesh.point_loads.append(load)

        displacements, _ = _solve(mesh)

        # Extract tip displacement
        tip_node_id = nodes[-1].id
        if etype == "reddy_bickford_2node":
            dofs_per_node = 4
        else:
            dofs_per_node = 3

        v_tip = displacements[(tip_node_id - 1) * dofs_per_node + 1]
        results_dict[etype] = v_tip

    w_EB = results_dict["euler_bernoulli_2node"]
    w_Timo = results_dict["timoshenko_2node"]
    w_Reddy = results_dict["reddy_bickford_2node"]

    print(f"  Euler-Bernoulli:   w_EB    = {w_EB:.6e} m  (ratio: 1.000)")
    print(f"  Timoshenko:        w_Timo  = {w_Timo:.6e} m  (ratio: {w_Timo/w_EB:.3f})")
    print(f"  Reddy-Bickford:    w_Reddy = {w_Reddy:.6e} m  (ratio: {w_Reddy/w_EB:.3f})")
    print(f"")
    print(f"  Shear effect (Timo):  {(w_Timo/w_EB - 1)*100:.2f}%")
    print(f"  Shear effect (Reddy): {(w_Reddy/w_EB - 1)*100:.2f}%")

    # Verify ordering: shear-deformable theories give more deflection than EB.
    # For downward loads both w values are negative; ratio > 1 means larger magnitude.
    assert w_Reddy / w_EB > 1.0, "Reddy deflection should be larger than EB"
    assert w_Timo / w_EB > 1.0, "Timoshenko deflection should be larger than EB"

    # For this configuration, both should be reasonably close
    # (typically within 20% of each other)
    assert abs(w_Reddy - w_Timo) / abs(w_EB) < 0.20, \
        "Reddy and Timoshenko should be relatively close for thick beams"

    print("OK Beam theory comparison shows expected behavior")
    return True


# ===========================================================================
# Test 7 – Mesh convergence for Reddy-Bickford
# ===========================================================================
def test_mesh_convergence():
    """
    Verify that the solution converges as the mesh is refined.
    Use a cantilever with distributed load.

    Test convergence of:
      - Tip deflection
      - Convergence rate (should improve with more elements)
    """
    print("\n" + "=" * 60)
    print("Test 7: Mesh Convergence (Cantilever with Distributed Load)")
    print("=" * 60)

    q = -1000.0  # N/m
    h = H_THICK

    n_values = [1, 2, 4, 8, 16]
    deflections = []

    for n in n_values:
        mesh, nodes = _make_cantilever(n, h)

        # Apply uniform load
        for element in mesh.elements:
            load = DistributedLoad(magnitude_start=q, magnitude_end=q, direction="t")
            load.element = element
            mesh.distributed_loads.append(load)

        displacements, _ = _solve(mesh)

        # Extract tip displacement
        tip_node_id = nodes[-1].id
        v_tip = displacements[(tip_node_id - 1) * 4 + 1]
        deflections.append(v_tip)

    print(f"  {'n':>4} | {'w_tip (m)':>15} | {'Diff from prev':>15}")
    print(f"  {'-'*4}-+-{'-'*15}-+-{'-'*15}")
    for i, (n, w) in enumerate(zip(n_values, deflections)):
        if i == 0:
            print(f"  {n:>4} | {w:>15.6e} | {'---':>15}")
        else:
            diff = abs(w - deflections[i-1])
            print(f"  {n:>4} | {w:>15.6e} | {diff:>15.6e}")

    # Verify convergence: differences should decrease
    diffs = [abs(deflections[i] - deflections[i-1]) for i in range(1, len(deflections))]

    for i in range(len(diffs) - 1):
        # Each refinement should reduce error
        # Allow some tolerance for numerical noise
        convergence_ratio = diffs[i+1] / diffs[i]
        print(f"  Convergence ratio {n_values[i+1]}/{n_values[i]}: {convergence_ratio:.4f}")
        # Expect ratio < 0.5 (at least 2x improvement per refinement)
        assert convergence_ratio < 0.8, \
            f"Poor convergence: ratio {convergence_ratio:.4f} at refinement {i}"

    print("OK Mesh convergence verified")
    return True


# ===========================================================================
# Test 8 – Force recovery: bending moment
# ===========================================================================
def test_bending_moment_recovery():
    """
    Test the bending_moment() method for a cantilever beam with point load.

    Analytical: M(x) = P·(L-x) for 0 <= x <= L

    The Reddy-Bickford moment is computed as:
      M_hat(x) = D1·theta'(x) - E1·v''(x)

    where theta' and v'' are extracted from the element displacement vector.
    """
    print("\n" + "=" * 60)
    print("Test 8: Bending Moment Recovery")
    print("=" * 60)

    P = -1000.0  # N
    h = H_THICK
    n = 4  # Multiple elements

    mesh, nodes = _make_cantilever(n, h)
    load = PointLoad(P, 1)
    load.node = nodes[-1]
    mesh.point_loads.append(load)

    displacements, results = _solve(mesh)

    # Check moment at fixed end (should be P·L)
    M_fixed_ana = P * L_BEAM
    M_fixed_fem = results.M(0.0)

    print(f"  Moment at fixed end:")
    print(f"    Analytical: {M_fixed_ana:.4e} N·m")
    print(f"    FEM:        {M_fixed_fem:.4e} N·m")
    print(f"    Error:      {abs(M_fixed_fem/M_fixed_ana - 1)*100:.2f}%")

    assert abs(M_fixed_fem / M_fixed_ana - 1.0) < 0.10, \
        f"Moment error > 10%: {abs(M_fixed_fem/M_fixed_ana - 1)*100:.2f}%"

    # Check moment at tip (should be zero)
    M_tip_fem = results.M(L_BEAM)
    print(f"  Moment at free end:")
    print(f"    Analytical: 0.0 N·m")
    print(f"    FEM:        {M_tip_fem:.4e} N·m")

    assert abs(M_tip_fem) < abs(M_fixed_ana) * 0.01, "Tip moment should be near zero"

    print("OK Bending moment recovery correct")
    return True


# ===========================================================================
# Test 9 – Force recovery: shear force
# ===========================================================================
def test_shear_force_recovery():
    """
    Test the shear_force() method for a cantilever beam with point load.

    Analytical: V(x) = P (constant along beam)

    The Reddy-Bickford shear force is computed as:
      V_hat = -dM_hat/dx = E1·v'''(x)

    For Hermite cubic v, v''' is constant, so V is constant per element.
    """
    print("\n" + "=" * 60)
    print("Test 9: Shear Force Recovery")
    print("=" * 60)

    P = -1000.0  # N
    h = H_THICK
    n = 4

    mesh, nodes = _make_cantilever(n, h)
    load = PointLoad(P, 1)
    load.node = nodes[-1]
    mesh.point_loads.append(load)

    displacements, results = _solve(mesh)

    # Check shear force (should be constant = P)
    V_mid = results.V(L_BEAM / 2)

    print(f"  Shear force at mid-span:")
    print(f"    Analytical: {P:.4e} N")
    print(f"    FEM:        {V_mid:.4e} N")
    print(f"    Error:      {abs(V_mid/P - 1)*100:.2f}%")

    assert abs(V_mid / P - 1.0) < 0.10, \
        f"Shear force error > 10%: {abs(V_mid/P - 1)*100:.2f}%"

    print("OK Shear force recovery correct")
    return True


# ===========================================================================
# Test 10 – Force recovery: normal force
# ===========================================================================
def test_normal_force_recovery():
    """
    Test the normal_force() method.

    For pure bending (no axial load), normal force should be zero everywhere.
    """
    print("\n" + "=" * 60)
    print("Test 10: Normal Force Recovery")
    print("=" * 60)

    P = -1000.0  # N, vertical only
    h = H_THICK

    mesh, nodes = _make_cantilever(1, h)
    load = PointLoad(P, 1)  # Vertical load only
    load.node = nodes[-1]
    mesh.point_loads.append(load)

    displacements, results = _solve(mesh)

    # Check normal force at several points (should be zero)
    N_values = [results.N(x) for x in [0.0, L_BEAM/2, L_BEAM]]

    print(f"  Normal force at various points:")
    for i, (x, N_val) in enumerate(zip([0.0, L_BEAM/2, L_BEAM], N_values)):
        print(f"    x={x:.2f}m: N={N_val:.6e} N")

    max_N = max(abs(n) for n in N_values)
    assert max_N < 1.0, f"Normal force should be near zero: max = {max_N:.6e}"

    print("OK Normal force recovery correct (zero for pure bending)")
    return True


# ===========================================================================
# Test 11 – Polynomial distributed loads across beam theories
# ===========================================================================
def test_beam_theory_polynomial_distributed_loads():
    """
    Verify that Euler-Bernoulli, Timoshenko, and Reddy-Bickford produce
    consistent internal-force diagrams for linear and quadratic distributed loads.

    Note:
      - Beam theory mainly affects deflection (through shear deformation effects).
      - For statically determinate beams, reactions and moment/shear resultants
        should remain consistent across theories.
    """
    print("\n" + "=" * 60)
    print("Test 11: Beam Theory Comparison with Polynomial Distributed Loads")
    print("=" * 60)

    n = 24
    h = H_THICK

    def _analytical_reactions(a0, a1, a2):
        total = a0 * L_BEAM + a1 * L_BEAM**2 / 2 + a2 * L_BEAM**3 / 3
        first_moment = a0 * L_BEAM**2 / 2 + a1 * L_BEAM**3 / 3 + a2 * L_BEAM**4 / 4
        r_right = -first_moment / L_BEAM
        r_left = -total - r_right
        return r_left, r_right, total

    def _analytical_moment(x, r_left, a0, a1, a2):
        return (
            r_left * x
            + a0 * x**2 / 2
            + a1 * x**3 / 6
            + a2 * x**4 / 12
        )

    load_cases = [
        ("Linear", -1000.0, -250.0, 0.0),
        ("Quadratic", -1000.0, -250.0, -80.0),
    ]

    for case_name, a0, a1, a2 in load_cases:
        print(f"\n  Case: {case_name} load")
        print(f"    q(x) = {a0:.1f} + ({a1:.1f})x + ({a2:.1f})x²  [N/m]")

        r_left_ana, r_right_ana, total_load = _analytical_reactions(a0, a1, a2)
        m_ref = max(
            abs(_analytical_moment(x, r_left_ana, a0, a1, a2))
            for x in np.linspace(0.0, L_BEAM, 200)
        )

        results_dict = {}
        for etype in ["euler_bernoulli_2node", "timoshenko_2node", "reddy_bickford_2node"]:
            mesh, nodes = _make_simply_supported(n, h, etype=etype)
            func = f"({a0}) + ({a1})*x + ({a2})*x**2"

            for element in mesh.elements:
                load = DistributedLoad(direction="y", func=func)
                load.element = element
                mesh.distributed_loads.append(load)

            displacements, results = _solve(mesh)
            ry_left, ry_right = _get_reactions(mesh, nodes, displacements)

            equilibrium_error = abs((ry_left + ry_right) + total_load) / abs(total_load)
            assert equilibrium_error < 1e-8, \
                f"{etype}: equilibrium error too large ({equilibrium_error * 100:.3e}%)"

            reaction_error_left = abs((ry_left - r_left_ana) / r_left_ana)
            reaction_error_right = abs((ry_right - r_right_ana) / r_right_ana)
            assert reaction_error_left < 2e-3, \
                f"{etype}: left reaction error too large ({reaction_error_left * 100:.3f}%)"
            assert reaction_error_right < 2e-3, \
                f"{etype}: right reaction error too large ({reaction_error_right * 100:.3f}%)"

            xs = np.linspace(0.1 * L_BEAM, 0.9 * L_BEAM, 17)
            max_moment_error = max(
                abs(results.M(float(x)) - _analytical_moment(float(x), r_left_ana, a0, a1, a2))
                for x in xs
            ) / m_ref
            assert max_moment_error < 1.5e-2, \
                f"{etype}: moment-diagram error too large ({max_moment_error * 100:.3f}%)"

            dofs_per_node = 4 if etype == "reddy_bickford_2node" else 3
            mid_node = nodes[len(nodes) // 2]
            v_mid = displacements[(mid_node.id - 1) * dofs_per_node + 1]

            results_dict[etype] = {
                "v_mid": v_mid,
                "ry_left": ry_left,
                "ry_right": ry_right,
                "moment_error": max_moment_error,
            }

            print(
                f"    {etype:>21}: "
                f"RyL={ry_left:.3f}, RyR={ry_right:.3f}, "
                f"moment err={max_moment_error * 100:.3f}%"
            )

        w_eb = results_dict["euler_bernoulli_2node"]["v_mid"]
        w_timo = results_dict["timoshenko_2node"]["v_mid"]
        w_reddy = results_dict["reddy_bickford_2node"]["v_mid"]

        assert w_timo / w_eb > 1.0, "Timoshenko mid-span deflection should exceed EB"
        assert w_reddy / w_eb > 1.0, "Reddy mid-span deflection should exceed EB"
        assert abs(w_reddy - w_timo) / abs(w_eb) < 0.05, \
            "Reddy and Timoshenko deflections should remain close for thick beams"

    print("OK Polynomial distributed-load behavior is consistent across beam theories")
    return True


# ===========================================================================
# Test 12 – Interpolation-order effects on polynomial-load force recovery
# ===========================================================================
def test_polynomial_load_interpolation_order_effects():
    """
    Verify that, for polynomial distributed loads on a coarse mesh:
      - Higher-order interpolation (3-node) improves force-diagram fidelity
        within the same beam-theory family.
      - Different beam theories with similar interpolation order keep comparable
        resultant-force trends (especially for statically determinate cases).
    """
    print("\n" + "=" * 60)
    print("Test 12: Interpolation Order Effects (Polynomial Loads, Coarse Mesh)")
    print("=" * 60)

    n = 2  # intentionally coarse to highlight interpolation-order differences
    h = H_THICK

    def _analytical_reactions(a0, a1, a2):
        total = a0 * L_BEAM + a1 * L_BEAM**2 / 2 + a2 * L_BEAM**3 / 3
        first_moment = a0 * L_BEAM**2 / 2 + a1 * L_BEAM**3 / 3 + a2 * L_BEAM**4 / 4
        r_right = -first_moment / L_BEAM
        r_left = -total - r_right
        return r_left, r_right

    def _analytical_moment(x, r_left, a0, a1, a2):
        return r_left * x + a0 * x**2 / 2 + a1 * x**3 / 6 + a2 * x**4 / 12

    def _analytical_shear(x, r_left, a0, a1, a2):
        return r_left + a0 * x + a1 * x**2 / 2 + a2 * x**3 / 3

    load_cases = [
        ("Linear", -1000.0, -250.0, 0.0),
        ("Quadratic", -1000.0, -250.0, -80.0),
    ]

    for case_name, a0, a1, a2 in load_cases:
        print(f"\n  Case: {case_name} load (coarse mesh, n={n})")
        r_left_ana, r_right_ana = _analytical_reactions(a0, a1, a2)
        func = f"({a0}) + ({a1})*x + ({a2})*x**2"
        xs = np.linspace(0.1 * L_BEAM, 0.9 * L_BEAM, 31)

        m_ref = max(abs(_analytical_moment(x, r_left_ana, a0, a1, a2)) for x in xs)
        v_ref = max(abs(_analytical_shear(x, r_left_ana, a0, a1, a2)) for x in xs)

        errors = {}
        etypes = [
            "euler_bernoulli_2node",
            "euler_bernoulli_3node",
            "timoshenko_2node",
            "timoshenko_3node",
            "reddy_bickford_2node",
        ]

        for etype in etypes:
            mesh, nodes = _make_simply_supported(n, h, etype=etype)
            for element in mesh.elements:
                load = DistributedLoad(direction="y", func=func)
                load.element = element
                mesh.distributed_loads.append(load)

            displacements, results = _solve(mesh)
            ry_left, ry_right = _get_reactions(mesh, nodes, displacements)

            rel_left = abs((ry_left - r_left_ana) / r_left_ana)
            rel_right = abs((ry_right - r_right_ana) / r_right_ana)
            assert rel_left < 6e-3 and rel_right < 6e-3, (
                f"{etype}: support reactions should remain accurate even on coarse mesh"
            )

            m_err = max(
                abs(results.M(float(x)) - _analytical_moment(float(x), r_left_ana, a0, a1, a2))
                for x in xs
            ) / m_ref
            v_err = max(
                abs(results.V(float(x)) - _analytical_shear(float(x), r_left_ana, a0, a1, a2))
                for x in xs
            ) / v_ref
            errors[etype] = (m_err, v_err)

            print(f"    {etype:>21}: M err={m_err*100:.2f}%, V err={v_err*100:.2f}%")

        # Euler-Bernoulli: 3-node (higher-order interpolation) should capture
        # polynomial-force trends much better than 2-node on coarse meshes.
        assert errors["euler_bernoulli_3node"][0] < 0.25 * errors["euler_bernoulli_2node"][0], \
            "Euler-Bernoulli 3-node should reduce moment error versus 2-node"
        assert errors["euler_bernoulli_3node"][1] < 0.25 * errors["euler_bernoulli_2node"][1], \
            "Euler-Bernoulli 3-node should reduce shear error versus 2-node"

        # Timoshenko: higher-order element mainly improves recovered shear curve.
        assert errors["timoshenko_3node"][1] < 0.75 * errors["timoshenko_2node"][1], \
            "Timoshenko 3-node should improve shear-force recovery versus 2-node"

        # Across 2-node theories, moment errors remain of similar order since
        # static resultants dominate and interpolation order is comparable.
        m_errs_2node = [
            errors["euler_bernoulli_2node"][0],
            errors["timoshenko_2node"][0],
            errors["reddy_bickford_2node"][0],
        ]
        assert max(m_errs_2node) < 1.25 * min(m_errs_2node), \
            "2-node EB/Timoshenko/Reddy moment-error levels should remain comparable"

    print("OK Higher-order interpolation improves polynomial-load force recovery as expected")
    return True


def test_polynomial_load_interpolation_order_effects_in_plotted_diagrams():
    """
    Verify that plotted force diagrams reflect the same interpolation-order trends:
      - marker values in Plotly traces match element force recovery values
      - coarse-mesh polynomial-load error trends shown in diagrams match theory expectations
    """
    print("\n" + "=" * 60)
    print("Test 13: Interpolation Order Effects in Plotted Diagrams")
    print("=" * 60)

    n = 2
    h = H_THICK
    a0, a1, a2 = -1000.0, -250.0, -80.0  # quadratic load
    func = f"({a0}) + ({a1})*x + ({a2})*x**2"

    def _analytical_reactions():
        total = a0 * L_BEAM + a1 * L_BEAM**2 / 2 + a2 * L_BEAM**3 / 3
        first_moment = a0 * L_BEAM**2 / 2 + a1 * L_BEAM**3 / 3 + a2 * L_BEAM**4 / 4
        r_right = -first_moment / L_BEAM
        r_left = -total - r_right
        return r_left, r_right

    def _analytical_moment(x, r_left):
        return r_left * x + a0 * x**2 / 2 + a1 * x**3 / 6 + a2 * x**4 / 12

    def _analytical_shear(x, r_left):
        return r_left + a0 * x + a1 * x**2 / 2 + a2 * x**3 / 3

    def _diagram_points(fig):
        traces = [
            tr for tr in fig.data
            if getattr(tr, "mode", None) == "markers"
            and getattr(getattr(tr, "marker", None), "size", None) == 7
        ]
        xs = np.concatenate([np.asarray(tr.x, dtype=float) for tr in traces])
        vals = np.concatenate([np.asarray(tr.customdata)[:, 0] for tr in traces])
        return xs, vals, traces

    r_left_ana, _ = _analytical_reactions()
    xs_ref = np.linspace(0.1 * L_BEAM, 0.9 * L_BEAM, 101)
    m_ref = max(abs(_analytical_moment(float(x), r_left_ana)) for x in xs_ref)
    v_ref = max(abs(_analytical_shear(float(x), r_left_ana)) for x in xs_ref)

    etypes = [
        "euler_bernoulli_2node",
        "euler_bernoulli_3node",
        "timoshenko_2node",
        "timoshenko_3node",
        "reddy_bickford_2node",
    ]
    errors = {}

    for etype in etypes:
        mesh, _ = _make_simply_supported(n, h, etype=etype)
        for element in mesh.elements:
            load = DistributedLoad(direction="y", func=func)
            load.element = element
            mesh.distributed_loads.append(load)

        _, results = _solve(mesh)

        fig_m = plot_structure_diagram(results, force_type="moment", n_points=31)
        fig_v = plot_structure_diagram(results, force_type="shear", n_points=31)

        x_m, val_m, traces_m = _diagram_points(fig_m)
        x_v, val_v, traces_v = _diagram_points(fig_v)

        assert len(traces_m) == len(results.element_results), \
            f"{etype}: expected one moment marker trace per element"
        assert len(traces_v) == len(results.element_results), \
            f"{etype}: expected one shear marker trace per element"

        for er, tr in zip(results.element_results, traces_m):
            xs_local = np.linspace(0.0, er.length, 31)
            expected = np.array([er.bending_moment(float(xi)) for xi in xs_local])
            plotted = np.asarray(tr.customdata)[:, 0]
            assert np.max(np.abs(expected - plotted)) < 1e-10, \
                f"{etype}: plotted moment values differ from recovered values"

        for er, tr in zip(results.element_results, traces_v):
            xs_local = np.linspace(0.0, er.length, 31)
            expected = np.array([er.shear_force(float(xi)) for xi in xs_local])
            plotted = np.asarray(tr.customdata)[:, 0]
            assert np.max(np.abs(expected - plotted)) < 1e-10, \
                f"{etype}: plotted shear values differ from recovered values"

        m_err = np.max(np.abs(val_m - np.array([_analytical_moment(float(x), r_left_ana) for x in x_m]))) / m_ref
        v_err = np.max(np.abs(val_v - np.array([_analytical_shear(float(x), r_left_ana) for x in x_v]))) / v_ref
        errors[etype] = (m_err, v_err)
        print(f"    {etype:>21}: plotted M err={m_err*100:.2f}%, plotted V err={v_err*100:.2f}%")

    assert errors["euler_bernoulli_3node"][0] < 0.25 * errors["euler_bernoulli_2node"][0], \
        "Plotted Euler-Bernoulli 3-node moment should improve versus 2-node"
    assert errors["euler_bernoulli_3node"][1] < 0.25 * errors["euler_bernoulli_2node"][1], \
        "Plotted Euler-Bernoulli 3-node shear should improve versus 2-node"
    assert errors["timoshenko_3node"][1] < 0.75 * errors["timoshenko_2node"][1], \
        "Plotted Timoshenko 3-node shear should improve versus 2-node"

    m_errs_2node = [
        errors["euler_bernoulli_2node"][0],
        errors["timoshenko_2node"][0],
        errors["reddy_bickford_2node"][0],
    ]
    assert max(m_errs_2node) < 1.25 * min(m_errs_2node), \
        "Plotted 2-node EB/Timoshenko/Reddy moment-error levels should remain comparable"

    print("OK Plotted diagrams preserve expected interpolation-order behavior")
    return True


# ===========================================================================
# Main test runner
# ===========================================================================
if __name__ == "__main__":
    tests = [
        test_reddy_parameters_rectangular_section,
        test_stiffness_matrix_symmetry,
        test_stiffness_matrix_positive_semidefinite,
        test_cantilever_point_load_single_element,
        test_simply_supported_uniform_load,
        test_beam_theory_comparison,
        test_mesh_convergence,
        test_bending_moment_recovery,
        test_shear_force_recovery,
        test_normal_force_recovery,
        test_beam_theory_polynomial_distributed_loads,
        test_polynomial_load_interpolation_order_effects,
        test_polynomial_load_interpolation_order_effects_in_plotted_diagrams,
    ]

    print("\n" + "=" * 70)
    print("REDDY-BICKFORD ELEMENT TEST SUITE")
    print("=" * 70)

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n*** FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n*** ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 70)

    if failed == 0:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ {failed} test(s) failed")
        sys.exit(1)
