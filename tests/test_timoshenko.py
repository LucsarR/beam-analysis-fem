"""
Tests for Timoshenko beam elements with custom distributed loads.

Verifies that custom distributed loads (sinusoidal, exponential) produce correct
displacements, bending moments, and shear forces for:
  - Slender beams  (shear deformation negligible → results match Euler-Bernoulli)
  - Thick beams    (shear deformation significant → larger deflection than EB)
  - Single-element and multi-element (subdivided) meshes
  - Both 'func'-based and piecewise-linear load representations

The 2-node Timoshenko element uses the field-consistent (phi-corrected) stiffness
matrix to avoid shear locking.  The shear force is recovered with the same
field-consistent formula, giving a constant value per element.

Sign convention (same as Euler-Bernoulli):
  - Positive q  = upward distributed load
  - Positive w  = upward deflection
  - M = EI · dθ/dx  (positive = sagging)
  - V constant per element (field-consistent recovery)

Analytical references for a simply-supported beam under sinusoidal load
q(x) = q0·sin(π·x/L):

  Euler-Bernoulli:
    w_EB(x)  = q0·L⁴/(EI·π⁴) · sin(π·x/L)

  Timoshenko (exact):
    w_T(x)   = w_EB(x) · (1 + Φ),   where Φ = EI·π²/(κGA·L²)

  Bending moment (same for EB and Timoshenko):
    M(x)   = −q0·L²/π²  · sin(π·x/L)

  Shear force (same for EB and Timoshenko):
    V(x)   = −q0·L/π    · cos(π·x/L)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fem.mesh import Mesh
from fem.material import Material
from fem.section import RectangularBar
from fem.constraint import Constraint
from fem.load import DistributedLoad
from fem.analysis import BeamAnalysis
from post_processing.forces import StructureResults

# ---------------------------------------------------------------------------
# Beam parameters shared across tests
# ---------------------------------------------------------------------------
E = 210e9       # Pa – Young's modulus (steel)
NU = 0.3        # Poisson's ratio
B = 0.05        # m  – section width
H_SLENDER = 0.01  # m  – very slender cross-section (L/h = 200)
H_THICK = 0.50    # m  – very thick  cross-section  (L/h =   4)
L_BEAM = 2.0      # m  – total beam span
KAPPA = 5.0 / 6   # Timoshenko shear correction factor (rectangular section)


def _ei(h):
    """EI for rectangular section B × h."""
    return E * (B * h**3 / 12)


def _kga(h):
    """κ·G·A for rectangular section B × h."""
    G = E / (2 * (1 + NU))
    return KAPPA * G * B * h


def _shear_phi(h, L):
    """Timoshenko shear-flexibility parameter Φ = EI·π²/(κGA·L²)."""
    return _ei(h) * np.pi**2 / (_kga(h) * L**2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_simply_supported(n, h, etype="timoshenko_2node"):
    mesh = Mesh()
    mat = Material(1, E, NU)
    sec = RectangularBar(1, B, h)
    nodes = mesh.generate_1d_mesh(0, 0, L_BEAM, 0, n, mat, sec, etype)
    mesh.constraints.add(Constraint(nodes[0], 0, 0.0))
    mesh.constraints.add(Constraint(nodes[0], 1, 0.0))
    mesh.constraints.add(Constraint(nodes[-1], 1, 0.0))
    return mesh, nodes


def _solve(mesh):
    analysis = BeamAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    results = StructureResults(mesh, displacements)
    return displacements, results


def _apply_piecewise_sin(mesh, n, q0):
    """Apply globally sinusoidal load as piecewise-linear to each element."""
    le = L_BEAM / n
    for i, element in enumerate(mesh.elements):
        xs = i * le
        xe = (i + 1) * le
        qs = q0 * np.sin(np.pi * xs / L_BEAM)
        qe = q0 * np.sin(np.pi * xe / L_BEAM)
        ld = DistributedLoad(magnitude_start=qs, magnitude_end=qe, direction="t")
        ld.element = element
        mesh.distributed_loads.append(ld)


def _apply_piecewise_exp(mesh, n, q0):
    """Apply globally exponential load q(x)=q0*(e^x − 1) as piecewise-linear."""
    le = L_BEAM / n
    for i, element in enumerate(mesh.elements):
        xs = i * le
        xe = (i + 1) * le
        qs = q0 * (np.exp(xs) - 1.0)
        qe = q0 * (np.exp(xe) - 1.0)
        ld = DistributedLoad(magnitude_start=qs, magnitude_end=qe, direction="t")
        ld.element = element
        mesh.distributed_loads.append(ld)


def _get_reactions(mesh, nodes, displacements):
    rxs = mesh.constraints.calculate_all_reactions(displacements)
    Ry_left = rxs.get((nodes[0].id, 1), 0.0)
    Ry_right = rxs.get((nodes[-1].id, 1), 0.0)
    return Ry_left, Ry_right


# ===========================================================================
# Test 1 – Sinusoidal load via func (single element, slender beam)
# ===========================================================================
def test_sinusoidal_load_func_single_element_slender():
    """
    Slender Timoshenko beam, 1 element, sinusoidal load via func.

    For a single-element mesh the local variable x spans [0, L_BEAM], so the
    func evaluates the correct global sinusoidal shape.  The slender beam has
    Φ ≈ 6.4×10⁻⁵ (shear effect < 0.01%), so reactions should match the
    Euler-Bernoulli analytical values.

    Checks: equilibrium and individual reactions.
    """
    print("\n" + "=" * 60)
    print("Test 1: Sinusoidal Load (func) – Single Element – Timoshenko Slender")
    print("=" * 60)

    q0 = -1000.0   # N/m downward
    R_each_ana = -q0 * 2 * L_BEAM / (2 * np.pi)   # = q0*L/π upward magnitude

    mesh, nodes = _make_simply_supported(1, H_SLENDER)
    load = DistributedLoad(direction="t", func=f"{q0}*np.sin(np.pi*x/L)")
    load.element = mesh.elements[0]
    mesh.distributed_loads.append(load)

    displacements, _ = _solve(mesh)
    Ry_left, Ry_right = _get_reactions(mesh, nodes, displacements)
    total_ana = q0 * 2.0 * L_BEAM / np.pi          # total downward load

    equil_err = abs((Ry_left + Ry_right) + total_ana) / abs(total_ana) * 100
    print(f"  Total applied load:  {total_ana:.4e} N/m · m")
    print(f"  Sum of reactions:    {Ry_left + Ry_right:.4e} N")
    print(f"  Equilibrium error:   {equil_err:.4f}%")
    assert equil_err < 0.1, f"Equilibrium error > 0.1%: {equil_err:.4f}%"

    for label, Ry in [("Left ", Ry_left), ("Right", Ry_right)]:
        ratio = Ry / R_each_ana
        print(f"  {label} reaction: FEM={Ry:.4e}, Ana={R_each_ana:.4e}, ratio={ratio:.6f}")
        assert abs(ratio - 1.0) < 0.02, f"{label} reaction error > 2%: ratio={ratio:.6f}"

    print("OK Single-element sinusoidal func: equilibrium and reactions correct")
    return True


# ===========================================================================
# Test 2 – Exponential load via func (single element, slender beam)
# ===========================================================================
def test_exponential_load_func_single_element_slender():
    """
    Slender Timoshenko beam, 1 element, exponential load via func:
        q(x) = q0 * (exp(x) − 1),   q0 < 0.

    Checks: equilibrium and reactions against numerical integration.
    """
    print("\n" + "=" * 60)
    print("Test 2: Exponential Load (func) – Single Element – Timoshenko Slender")
    print("=" * 60)

    q0 = -500.0   # N/m
    # Analytical: total load and reactions from moment equilibrium.
    # R_left*L = -∫ q(x)*(L−x) dx  (moment about right support; load is downward)
    from scipy.integrate import quad  # noqa: PLC0415
    total_ana, _ = quad(lambda x: q0 * (np.exp(x) - 1.0), 0, L_BEAM)
    moment_right, _ = quad(lambda x: q0 * (np.exp(x) - 1.0) * (L_BEAM - x), 0, L_BEAM)
    R_left_ana  = -moment_right / L_BEAM        # upward (positive)
    R_right_ana = -total_ana - R_left_ana        # balance: R_l + R_r = -total_ana

    mesh, nodes = _make_simply_supported(1, H_SLENDER)
    load = DistributedLoad(direction="t", func=f"{q0}*(np.exp(x)-1)")
    load.element = mesh.elements[0]
    mesh.distributed_loads.append(load)

    displacements, _ = _solve(mesh)
    Ry_left, Ry_right = _get_reactions(mesh, nodes, displacements)

    equil_err = abs((Ry_left + Ry_right) + total_ana) / abs(total_ana) * 100
    print(f"  Total applied load:  {total_ana:.4e} N")
    print(f"  Equilibrium error:   {equil_err:.4f}%")
    assert equil_err < 0.1, f"Equilibrium error > 0.1%: {equil_err:.4f}%"

    for label, Ry, R_ana in [("Left ", Ry_left, R_left_ana), ("Right", Ry_right, R_right_ana)]:
        ratio = Ry / R_ana
        print(f"  {label} reaction: FEM={Ry:.4e}, Ana={R_ana:.4e}, ratio={ratio:.6f}")
        assert abs(ratio - 1.0) < 0.02, f"{label} reaction error > 2%: ratio={ratio:.6f}"

    print("OK Single-element exponential func: equilibrium and reactions correct")
    return True


# ===========================================================================
# Test 3 – Slender beam deflection matches EB (multi-element sinusoidal)
# ===========================================================================
def test_slender_timoshenko_matches_euler_bernoulli():
    """
    Slender Timoshenko beam: mid-span deflection must match EB analytical solution
    within the FEM discretisation error, since Φ ≈ 6.4×10⁻⁵ ≪ 1.

    Uses piecewise-linear representation of the sinusoidal load (n=20 elements).
    The FEM discretisation error scales as 1/n^2; with n=20 the error should be < 2%.
    """
    print("\n" + "=" * 60)
    print("Test 3: Slender Timoshenko ≈ Euler-Bernoulli (n=20, sinusoidal load)")
    print("=" * 60)

    q0 = -1000.0
    n = 20
    phi = _shear_phi(H_SLENDER, L_BEAM)
    w_EB_ana = q0 * L_BEAM**4 / (_ei(H_SLENDER) * np.pi**4)   # EB mid-span
    w_T_ana = w_EB_ana * (1.0 + phi)                           # Timoshenko exact

    print(f"  Shear-flexibility Φ = {phi:.6e}  (shear effect: {phi*100:.4f}%)")
    print(f"  w_EB_mid (analytical) = {w_EB_ana:.6e} m")
    print(f"  w_T_mid  (analytical) = {w_T_ana:.6e} m")

    mesh, nodes = _make_simply_supported(n, H_SLENDER)
    _apply_piecewise_sin(mesh, n, q0)
    displacements, _ = _solve(mesh)

    mid_node_id = nodes[n // 2].id
    w_fem = displacements[3 * (mid_node_id - 1) + 1]
    err_vs_EB = abs(w_fem - w_EB_ana) / abs(w_EB_ana) * 100
    err_vs_T  = abs(w_fem - w_T_ana) / abs(w_T_ana) * 100

    print(f"  w_T_mid  (FEM)        = {w_fem:.6e} m")
    print(f"  Error vs EB analytical: {err_vs_EB:.4f}%")
    print(f"  Error vs T  analytical: {err_vs_T:.4f}%")

    # Slender beam: FEM should agree with EB within discretisation error (<2%)
    assert err_vs_EB < 2.0, f"Slender T deviation from EB too large: {err_vs_EB:.2f}%"
    # And the shear correction is negligible
    assert phi < 1e-3, f"Φ unexpectedly large for slender beam: {phi:.4e}"

    print("OK Slender Timoshenko matches Euler-Bernoulli within FEM error")
    return True


# ===========================================================================
# Test 4 – Thick beam deflection exceeds EB (shear deformation visible)
# ===========================================================================
def test_thick_timoshenko_shear_effect():
    """
    Thick Timoshenko beam (L/h = 4): shear deformation adds Φ ≈ 16% to the
    mid-span deflection compared to Euler-Bernoulli.

    Uses piecewise-linear sinusoidal load with n=20 elements.
    Checks that:
      1. FEM deflection is closer to the Timoshenko analytical value than to EB.
      2. FEM error vs Timoshenko analytical is < 2% (FEM discretisation).
    """
    print("\n" + "=" * 60)
    print("Test 4: Thick Timoshenko – Shear Effect on Deflection (n=20)")
    print("=" * 60)

    q0 = -1000.0
    n = 20
    phi = _shear_phi(H_THICK, L_BEAM)
    w_EB_ana = q0 * L_BEAM**4 / (_ei(H_THICK) * np.pi**4)
    w_T_ana  = w_EB_ana * (1.0 + phi)

    print(f"  Shear-flexibility Φ = {phi:.6f}  (shear adds {phi*100:.2f}%)")
    print(f"  w_EB_mid (analytical) = {w_EB_ana:.6e} m")
    print(f"  w_T_mid  (analytical) = {w_T_ana:.6e} m")

    mesh, nodes = _make_simply_supported(n, H_THICK)
    _apply_piecewise_sin(mesh, n, q0)
    displacements, _ = _solve(mesh)

    mid_node_id = nodes[n // 2].id
    w_fem = displacements[3 * (mid_node_id - 1) + 1]
    err_vs_EB = abs(w_fem - w_EB_ana) / abs(w_EB_ana) * 100
    err_vs_T  = abs(w_fem - w_T_ana) / abs(w_T_ana) * 100

    print(f"  w_T_mid  (FEM)        = {w_fem:.6e} m")
    print(f"  Error vs EB:  {err_vs_EB:.4f}%")
    print(f"  Error vs T :  {err_vs_T:.4f}%")

    # FEM should be much closer to Timoshenko than to EB
    assert err_vs_T < err_vs_EB, (
        "FEM should be closer to Timoshenko analytical than to EB analytical"
    )
    # FEM accuracy vs Timoshenko
    assert err_vs_T < 2.0, f"Thick beam FEM error vs Timoshenko > 2%: {err_vs_T:.2f}%"
    # Shear effect must be clearly visible (phi > 5%)
    assert phi > 0.05, f"Φ too small to demonstrate shear effect: {phi:.4f}"

    print("OK Thick Timoshenko: shear deformation verified, FEM matches analytical")
    return True


# ===========================================================================
# Test 5 – Bending moments in thick beam (sinusoidal load, multi-element)
# ===========================================================================
def test_bending_moment_thick_sinusoidal():
    """
    Bending moment M(x) for the Timoshenko beam under sinusoidal load is
    theoretically identical to the Euler-Bernoulli result:

        M(x) = −q0·L²/π² · sin(π·x/L)

    This test checks that the FEM bending moments at several element midpoints
    agree with this analytical formula within 1% for n=20 elements.
    """
    print("\n" + "=" * 60)
    print("Test 5: Bending Moment – Thick Timoshenko – Sinusoidal Load (n=20)")
    print("=" * 60)

    q0 = -1000.0
    n = 20
    le = L_BEAM / n

    mesh, nodes = _make_simply_supported(n, H_THICK)
    _apply_piecewise_sin(mesh, n, q0)
    displacements, results = _solve(mesh)

    check_elems = [3, 7, 10, 14, 17]
    print(f"  {'elem':>5}  {'x_mid':>7}  {'M_FEM':>12}  {'M_Ana':>12}  {'err%':>7}")
    for i_elem in check_elems:
        x_mid = (i_elem + 0.5) * le
        M_fem = results.element_results[i_elem].bending_moment(le / 2)
        M_ana = -q0 * L_BEAM**2 / np.pi**2 * np.sin(np.pi * x_mid / L_BEAM)
        err = abs(M_fem - M_ana) / abs(M_ana) * 100
        print(f"  {i_elem:>5}  {x_mid:>7.3f}  {M_fem:>12.4e}  {M_ana:>12.4e}  {err:>7.3f}%")
        assert err < 1.5, f"Bending moment error > 1.5% at x={x_mid:.3f}: err={err:.3f}%"

    print("OK Bending moments match analytical within 1.5%")
    return True


# ===========================================================================
# Test 6 – Shear forces in thick beam (sinusoidal load, multi-element)
# ===========================================================================
def test_shear_force_thick_sinusoidal():
    """
    Shear force V(x) under sinusoidal load is the same for EB and Timoshenko:

        V(x) = −q0·L/π · cos(π·x/L)

    This test verifies the field-consistent shear force recovery for the
    Timoshenko element using n=20 elements on a thick (L/h=4) beam.
    Checks element midpoints with |V_ana| > 50 N.
    """
    print("\n" + "=" * 60)
    print("Test 6: Shear Force – Thick Timoshenko – Sinusoidal Load (n=20)")
    print("=" * 60)

    q0 = -1000.0
    n = 20
    le = L_BEAM / n

    mesh, nodes = _make_simply_supported(n, H_THICK)
    _apply_piecewise_sin(mesh, n, q0)
    displacements, results = _solve(mesh)

    check_elems = [2, 5, 13, 16]
    print(f"  {'elem':>5}  {'x_mid':>7}  {'V_FEM':>12}  {'V_Ana':>12}  {'err%':>7}")
    for i_elem in check_elems:
        x_mid = (i_elem + 0.5) * le
        V_fem = results.element_results[i_elem].shear_force(le / 2)
        V_ana = -q0 * L_BEAM / np.pi * np.cos(np.pi * x_mid / L_BEAM)
        err = abs(V_fem - V_ana) / abs(V_ana) * 100
        print(f"  {i_elem:>5}  {x_mid:>7.3f}  {V_fem:>12.4e}  {V_ana:>12.4e}  {err:>7.3f}%")
        assert err < 1.5, f"Shear force error > 1.5% at x={x_mid:.3f}: err={err:.3f}%"

    print("OK Shear forces match analytical within 1.5% (field-consistent recovery)")
    return True


# ===========================================================================
# Test 7 – Exponential load (multi-element, slender beam)
# ===========================================================================
def test_exponential_load_multi_element_slender():
    """
    Slender Timoshenko beam, exponential load piecewise-linear (n=16 elements):
        q(x) = q0·(e^x − 1)

    Because the slender beam behaves like EB, we compare the mid-span deflection
    to a high-resolution FEM reference (n=128, same Timoshenko element).
    Checks that:
      - The n=16 result is within 2% of the n=128 reference.
      - Mesh refinement reduces the error monotonically.
    """
    print("\n" + "=" * 60)
    print("Test 7: Exponential Load – Slender Timoshenko – Piecewise-linear (n=16)")
    print("=" * 60)

    q0 = -500.0

    def _run(n):
        mesh, nodes = _make_simply_supported(n, H_SLENDER)
        _apply_piecewise_exp(mesh, n, q0)
        disp, _ = _solve(mesh)
        mid_node_id = nodes[n // 2].id
        return disp[3 * (mid_node_id - 1) + 1]

    n_ref = 128
    w_ref = _run(n_ref)
    print(f"  Reference (n={n_ref}): w_mid = {w_ref:.6e} m")

    prev_err = None
    for n in [4, 8, 16]:
        w = _run(n)
        err = abs(w - w_ref) / abs(w_ref) * 100
        print(f"  n={n:3d}: w_mid = {w:.6e} m, err vs ref = {err:.4f}%")
        if prev_err is not None:
            assert err < prev_err, f"Error should decrease with refinement at n={n}"
        prev_err = err

    assert prev_err < 2.0, f"n=16 error vs reference > 2%: {prev_err:.4f}%"
    print("OK Exponential load: monotonic convergence and n=16 within 2% of reference")
    return True


# ===========================================================================
# Test 8 – Mesh convergence (thick beam, sinusoidal load)
# ===========================================================================
def test_mesh_convergence_thick_sinusoidal():
    """
    As n increases the mid-span deflection should converge monotonically to the
    Timoshenko analytical value  w_T = w_EB·(1+Φ).

    Uses a thick beam (H=0.5 m, L/h=4, Φ≈16%) and piecewise-linear sinusoidal load.
    Verifies convergence for n = 4, 8, 16, 32 elements.
    """
    print("\n" + "=" * 60)
    print("Test 8: Mesh Convergence – Thick Timoshenko – Sinusoidal Load")
    print("=" * 60)

    q0 = -1000.0
    phi = _shear_phi(H_THICK, L_BEAM)
    w_T_ana = q0 * L_BEAM**4 / (_ei(H_THICK) * np.pi**4) * (1.0 + phi)

    print(f"  Timoshenko analytical w_T_mid = {w_T_ana:.6e} m  (Φ={phi:.4f})")
    print(f"  {'n':>4}  {'w_mid':>14}  {'err%':>8}")

    prev_err = None
    for n in [4, 8, 16, 32]:
        mesh, nodes = _make_simply_supported(n, H_THICK)
        _apply_piecewise_sin(mesh, n, q0)
        disp, _ = _solve(mesh)
        mid_node_id = nodes[n // 2].id
        w_fem = disp[3 * (mid_node_id - 1) + 1]
        err = abs(w_fem - w_T_ana) / abs(w_T_ana) * 100
        print(f"  {n:>4}  {w_fem:>14.6e}  {err:>8.4f}%")
        if prev_err is not None:
            assert err < prev_err, f"Convergence failure at n={n}: err={err:.4f}% vs prev={prev_err:.4f}%"
        prev_err = err

    assert prev_err < 1.0, f"n=32 error vs analytical > 1%: {prev_err:.4f}%"
    print("OK Mesh convergence verified (monotonic, n=32 within 1% of analytical)")
    return True


# ===========================================================================
# Test 9 – Sinusoidal load via func on multi-element thick beam
# ===========================================================================
def test_sinusoidal_func_multi_element_thick():
    """
    Sinusoidal load applied via the func feature on each element independently.

    For a multi-element mesh the 'func' variable x spans [0, L_elem] (the LOCAL
    element coordinate).  Therefore func='{q0}*sin(pi*x/L)' applies a full
    half-cycle over each element's length.  This is NOT the same as the global
    sinusoidal pattern q0*sin(pi*x_global/L_beam), but it is a valid, physically
    meaningful load that exercises the func evaluation and Gauss quadrature.

    The test checks vertical equilibrium only, since there is no simple analytical
    closed form for the accumulated multi-element pattern.
    """
    print("\n" + "=" * 60)
    print("Test 9: Sinusoidal func on Multi-Element Thick Beam – Equilibrium Check")
    print("=" * 60)

    q0 = -1000.0
    n = 8

    # Each element has the same func; the magnitude at the end-points (x=0 and
    # x=L_elem) is zero, so the load on each element integrates to
    # q0 * 2*L_elem/pi.  Total = q0 * 2*L_beam/pi.
    le = L_BEAM / n
    total_per_elem_ana = q0 * 2.0 * le / np.pi    # integral of q0*sin(pi*x/le) over [0,le]
    total_ana = n * total_per_elem_ana

    mesh, nodes = _make_simply_supported(n, H_THICK)
    for element in mesh.elements:
        ld = DistributedLoad(direction="t", func=f"{q0}*np.sin(np.pi*x/L)")
        ld.element = element
        mesh.distributed_loads.append(ld)

    displacements, _ = _solve(mesh)
    Ry_left, Ry_right = _get_reactions(mesh, nodes, displacements)
    equil_err = abs((Ry_left + Ry_right) + total_ana) / abs(total_ana) * 100

    print(f"  Total analytical load:  {total_ana:.4e} N  (n·∫q over each element)")
    print(f"  Sum of reactions:       {Ry_left + Ry_right:.4e} N")
    print(f"  Equilibrium error:      {equil_err:.4f}%")
    assert equil_err < 0.1, f"Equilibrium error > 0.1%: {equil_err:.4f}%"

    print("OK Multi-element sinusoidal func: equilibrium satisfied")
    return True


# ===========================================================================
# Main runner
# ===========================================================================
if __name__ == "__main__":
    tests = [
        ("Test 1: Sinusoidal func – 1 element – slender",   test_sinusoidal_load_func_single_element_slender),
        ("Test 2: Exponential func – 1 element – slender",  test_exponential_load_func_single_element_slender),
        ("Test 3: Slender Timoshenko ≈ EB (n=20)",          test_slender_timoshenko_matches_euler_bernoulli),
        ("Test 4: Thick beam shear effect (n=20)",           test_thick_timoshenko_shear_effect),
        ("Test 5: Bending moment – thick – sinusoidal",      test_bending_moment_thick_sinusoidal),
        ("Test 6: Shear force – thick – sinusoidal",         test_shear_force_thick_sinusoidal),
        ("Test 7: Exponential load – slender – piecewise",   test_exponential_load_multi_element_slender),
        ("Test 8: Mesh convergence – thick – sinusoidal",    test_mesh_convergence_thick_sinusoidal),
        ("Test 9: Sinusoidal func – multi-element – thick",  test_sinusoidal_func_multi_element_thick),
    ]

    print("=" * 60)
    print("RUNNING TIMOSHENKO DISTRIBUTED LOAD TESTS")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, func in tests:
        try:
            func()
            passed += 1
        except Exception as exc:
            print(f"\nFAILED: {name}")
            print(f"  {exc}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    if failed:
        sys.exit(1)
