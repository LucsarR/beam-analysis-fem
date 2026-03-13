"""
Tests for Euler-Bernoulli beam elements with custom distributed loads.

Verifies that custom distributed loads (sinusoidal, exponential) produce correct
displacements, bending moments, and shear forces compared to analytical solutions
for:
  - Simply supported beams (Euler-Bernoulli, 2-node elements)
  - Cantilever beams (Euler-Bernoulli, 2-node elements)
  - Single-element and multi-element (subdivided) meshes

Sign convention used throughout (standard Euler-Bernoulli):
  - Positive q  = upward distributed load
  - Positive w  = upward deflection
  - M = EI * w'' (positive = sagging, i.e. concave upward)
  - V = EI * w''' (shear force)

Analytical references:
  Sinusoidal load q(x) = q0*sin(pi*x/L)  on simply supported beam:
    w(x)   = q0*L^4 / (EI*pi^4) * sin(pi*x/L)
    M(x)   = -q0*L^2/pi^2        * sin(pi*x/L)
    V(x)   = -q0*L/pi            * cos(pi*x/L)

  Uniform load q0 on cantilever (fixed at x=0, free at x=L):
    w_tip  = q0*L^4 / (8*EI)
    M(x)   = q0*(x-L)^2/2              [M(0) = q0*L^2/2]

Notes on single-element accuracy:
  A single Hermite cubic element exactly recovers nodal displacements for
  polynomial loads up to degree 3 but cannot represent the exact shape of
  higher-order or transcendental displacement fields (e.g. sinusoidal).
  Consequently, *internal* bending moments from a one-element model may have
  significant errors even when the global equilibrium is satisfied.
  Tests that verify internal force accuracy therefore use multiple elements.
"""

import sys
import os
import numpy as np
from scipy.integrate import quad

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fem.mesh import Mesh
from fem.material import Material
from fem.section import RectangularBar
from fem.constraint import Constraint
from fem.load import DistributedLoad
from fem.analysis import BeamAnalysis
from post_processing.forces import StructureResults

# ---------------------------------------------------------------------------
# Beam parameters (used across tests)
# ---------------------------------------------------------------------------
E = 210e9       # Pa  – Young's modulus (steel)
NU = 0.3        # Poisson's ratio
B = 0.05        # m   – section width
H = 0.10        # m   – section height
I_SEC = B * H**3 / 12   # m^4 – second moment of area
L_BEAM = 2.0    # m   – total beam length
EI = E * I_SEC  # N*m^2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_simply_supported(n_elements, element_type="euler_bernoulli_2node"):
    mesh = Mesh()
    mat = Material(1, E, NU)
    sec = RectangularBar(1, B, H)
    nodes = mesh.generate_1d_mesh(0, 0, L_BEAM, 0, n_elements, mat, sec, element_type)
    mesh.constraints.add(Constraint(nodes[0], 0, 0.0))
    mesh.constraints.add(Constraint(nodes[0], 1, 0.0))
    mesh.constraints.add(Constraint(nodes[-1], 1, 0.0))
    return mesh, nodes


def _make_cantilever(n_elements, element_type="euler_bernoulli_2node"):
    mesh = Mesh()
    mat = Material(1, E, NU)
    sec = RectangularBar(1, B, H)
    nodes = mesh.generate_1d_mesh(0, 0, L_BEAM, 0, n_elements, mat, sec, element_type)
    mesh.constraints.add(Constraint(nodes[0], 0, 0.0))
    mesh.constraints.add(Constraint(nodes[0], 1, 0.0))
    mesh.constraints.add(Constraint(nodes[0], 2, 0.0))
    return mesh, nodes


def _solve(mesh):
    analysis = BeamAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    results = StructureResults(mesh, displacements)
    return displacements, results


def _apply_piecewise_sin(mesh, n, q0):
    """Apply globally sinusoidal load piecewise-linearly to each element."""
    le = L_BEAM / n
    for i, element in enumerate(mesh.elements):
        xs = i * le
        xe = (i + 1) * le
        qs = q0 * np.sin(np.pi * xs / L_BEAM)
        qe = q0 * np.sin(np.pi * xe / L_BEAM)
        ld = DistributedLoad(magnitude_start=qs, magnitude_end=qe, direction="t")
        ld.element = element
        mesh.distributed_loads.append(ld)


def _get_reactions(mesh, nodes, displacements):
    """Return (Ry_left, Ry_right) vertical reactions for a simply-supported beam."""
    rxs = mesh.constraints.calculate_all_reactions(displacements)
    Ry_left = rxs.get((nodes[0].id, 1), 0.0)
    Ry_right = rxs.get((nodes[-1].id, 1), 0.0)
    return Ry_left, Ry_right


# ===========================================================================
# Test 1 – Uniform constant load via func  (simply supported, single element)
# ===========================================================================
def test_uniform_load_single_element_func():
    """Baseline: constant load via func on a 1-element simply-supported EB beam.

    For a uniform load the consistent nodal forces are exact, so the reactions
    must exactly equal q0*L/2 each.
    """
    print("\n" + "=" * 60)
    print("Test 1: Uniform Load (func) – Single Element – EB")
    print("=" * 60)

    q0 = -1000.0   # N/m (downward)

    mesh, nodes = _make_simply_supported(1)
    load = DistributedLoad(direction="t", func=str(q0))
    load.element = mesh.elements[0]
    mesh.distributed_loads.append(load)

    displacements, _ = _solve(mesh)

    Ry_left, Ry_right = _get_reactions(mesh, nodes, displacements)
    expected = -q0 * L_BEAM / 2   # upward (positive)

    for label, Ry in [("Left", Ry_left), ("Right", Ry_right)]:
        ratio = Ry / expected
        print(f"  {label} reaction: FEM={Ry:.2f}, Expected={expected:.2f}, ratio={ratio:.6f}")
        assert abs(ratio - 1.0) < 0.01, f"{label} reaction error > 1%: ratio={ratio:.6f}"

    print("OK Uniform load via func: reactions correct within 1%")
    return True


# ===========================================================================
# Test 2 – Sinusoidal load, single element – equilibrium check
# ===========================================================================
def test_sinusoidal_load_single_element_eb():
    """
    Simply-supported EB beam, single element, sinusoidal transverse load:
        q(x) = q0 * sin(pi*x/L)

    For a single-element mesh, L_elem = L_BEAM so the func variable x spans
    the full beam length, yielding the correct sinusoidal load shape.

    Checks:
    - Vertical equilibrium (reactions balance the integrated applied load).
    - Individual reactions match analytical values (sum of moments).

    NOTE: Internal bending moments are NOT checked for the single-element case
    because a cubic Hermite element cannot represent the sinusoidal displacement
    field exactly; moment errors of 30-40% are expected and acceptable here.
    Moment accuracy with multiple elements is tested in test 6.
    """
    print("\n" + "=" * 60)
    print("Test 2: Sinusoidal Load – Single Element – EB  (equilibrium)")
    print("=" * 60)

    q0 = -1000.0   # N/m downward

    # Analytical: integral of q0*sin(pi*x/L) dx over [0,L] = -q0*2L/pi
    total_load_ana = q0 * 2 * L_BEAM / np.pi         # negative (downward)
    # Symmetric load => equal reactions
    R_each_ana = -total_load_ana / 2                  # upward (positive)

    mesh, nodes = _make_simply_supported(1)
    load = DistributedLoad(direction="t", func=f"{q0}*np.sin(np.pi*x/L)")
    load.element = mesh.elements[0]
    mesh.distributed_loads.append(load)

    displacements, _ = _solve(mesh)
    Ry_left, Ry_right = _get_reactions(mesh, nodes, displacements)
    total_reaction = Ry_left + Ry_right

    # Equilibrium
    equil_err = abs(total_reaction + total_load_ana) / abs(total_load_ana) * 100
    print(f"  Total load: {total_load_ana:.4e}, Reaction sum: {total_reaction:.4e}, "
          f"equil err: {equil_err:.4f}%")
    assert equil_err < 1.0, f"Equilibrium error > 1%: {equil_err:.4f}%"

    # Individual reactions
    for label, Ry in [("Left", Ry_left), ("Right", Ry_right)]:
        ratio = Ry / R_each_ana
        print(f"  {label} reaction: FEM={Ry:.4e}, Ana={R_each_ana:.4e}, ratio={ratio:.6f}")
        assert abs(ratio - 1.0) < 0.02, f"{label} reaction error > 2%: ratio={ratio:.6f}"

    print("OK Sinusoidal load (single element): equilibrium correct within 1%, "
          "reactions within 2%")
    return True


# ===========================================================================
# Test 3 – Exponential load, single element – equilibrium check
# ===========================================================================
def test_exponential_load_single_element_eb():
    """
    Simply-supported EB beam, single element, exponential transverse load:
        q(x) = q0 * exp(alpha * x / L),  alpha = 1.

    Checks vertical equilibrium and individual reactions against statics.
    """
    print("\n" + "=" * 60)
    print("Test 3: Exponential Load – Single Element – EB  (equilibrium)")
    print("=" * 60)

    q0 = -500.0
    alpha = 1.0

    # Analytical reactions from statics (moment equilibrium):
    # R_left = -integral(q(x)*(L-x)/L dx, 0, L)  (upward, positive)
    int_q_Lx, _ = quad(
        lambda x: q0 * np.exp(alpha * x / L_BEAM) * (L_BEAM - x) / L_BEAM, 0, L_BEAM
    )
    R_left_ana = -int_q_Lx    # upward: negate the integral (q < 0, integral < 0, R > 0)

    total_load, _ = quad(lambda x: q0 * np.exp(alpha * x / L_BEAM), 0, L_BEAM)
    R_right_ana = -total_load - R_left_ana  # from sum Fy = 0

    mesh, nodes = _make_simply_supported(1)
    load = DistributedLoad(direction="t", func=f"{q0}*np.exp({alpha}*x/L)")
    load.element = mesh.elements[0]
    mesh.distributed_loads.append(load)

    displacements, _ = _solve(mesh)
    Ry_left, Ry_right = _get_reactions(mesh, nodes, displacements)
    total_reaction = Ry_left + Ry_right

    # Equilibrium
    equil_err = abs(total_reaction + total_load) / abs(total_load) * 100
    print(f"  Total load: {total_load:.4e}, Reaction sum: {total_reaction:.4e}, "
          f"equil err: {equil_err:.4f}%")
    assert equil_err < 1.0, f"Equilibrium error > 1%: {equil_err:.4f}%"

    for label, Ry, R_ana in [("Left",  Ry_left,  R_left_ana),
                              ("Right", Ry_right, R_right_ana)]:
        ratio = Ry / R_ana
        print(f"  {label}: FEM={Ry:.4e}, Ana={R_ana:.4e}, ratio={ratio:.6f}")
        assert abs(ratio - 1.0) < 0.02, f"{label} reaction error > 2%: ratio={ratio:.6f}"

    print("OK Exponential load (single element): equilibrium and reactions correct")
    return True


# ===========================================================================
# Test 4 – Sinusoidal load, multi-element (piecewise-linear per element)
# ===========================================================================
def test_sinusoidal_load_multi_element_eb():
    """
    Simply-supported EB beam, N elements, globally sinusoidal load
    represented as piecewise-linear per element.

    Analytical mid-span deflection:  w_max = q0*L^4 / (EI*pi^4)

    As N increases the piecewise-linear representation of q(x) becomes more
    accurate and the FEM deflection converges to the analytical value.
    """
    print("\n" + "=" * 60)
    print("Test 4: Sinusoidal Load – Multi-Element (piecewise-linear) – EB")
    print("=" * 60)

    q0 = -1000.0
    w_ana = q0 * L_BEAM**4 / (EI * np.pi**4)

    errors = []
    for n in [2, 4, 8, 16]:
        mesh, nodes = _make_simply_supported(n)
        _apply_piecewise_sin(mesh, n, q0)
        displacements, _ = _solve(mesh)

        mid_id = nodes[n // 2].id
        w_mid = displacements[3 * (mid_id - 1) + 1]
        err = abs(w_mid - w_ana) / abs(w_ana) * 100
        errors.append(err)
        print(f"  n={n:2d}: w_mid={w_mid:.6e}, ana={w_ana:.6e}, err={err:.4f}%")

    # Monotonic convergence
    for k in range(len(errors) - 1):
        assert errors[k + 1] <= errors[k] * 1.05, \
            f"Non-monotonic convergence: {errors}"

    assert errors[-1] < 0.5, f"16-element sinusoidal error {errors[-1]:.4f}% > 0.5%"
    print("OK Piecewise-sinusoidal load: converges monotonically with mesh refinement")
    return True


# ===========================================================================
# Test 5 – Exponential load, multi-element (piecewise-linear per element)
# ===========================================================================
def test_exponential_load_multi_element_eb():
    """
    Simply-supported EB beam, N elements, globally exponential load:
        q(x) = q0 * exp(alpha * x / L_beam),  alpha = 1.

    Uses a 128-element reference solution and verifies convergence.
    """
    print("\n" + "=" * 60)
    print("Test 5: Exponential Load – Multi-Element (piecewise-linear) – EB")
    print("=" * 60)

    q0 = -500.0
    alpha = 1.0

    def _apply_exp(mesh, n):
        le = L_BEAM / n
        for i, element in enumerate(mesh.elements):
            xs = i * le
            xe = (i + 1) * le
            qs = q0 * np.exp(alpha * xs / L_BEAM)
            qe = q0 * np.exp(alpha * xe / L_BEAM)
            ld = DistributedLoad(magnitude_start=qs, magnitude_end=qe, direction="t")
            ld.element = element
            mesh.distributed_loads.append(ld)

    # Reference: 128 elements
    mesh_ref, nodes_ref = _make_simply_supported(128)
    _apply_exp(mesh_ref, 128)
    disp_ref, _ = _solve(mesh_ref)
    w_ref = disp_ref[3 * (nodes_ref[64].id - 1) + 1]
    print(f"  Reference (128 elem): w_mid = {w_ref:.6e}")

    errors = []
    for n in [4, 8, 16, 32]:
        mesh, nodes = _make_simply_supported(n)
        _apply_exp(mesh, n)
        displacements, _ = _solve(mesh)
        mid_id = nodes[n // 2].id
        w_mid = displacements[3 * (mid_id - 1) + 1]
        err = abs(w_mid - w_ref) / abs(w_ref) * 100
        errors.append(err)
        print(f"  n={n:2d}: w_mid={w_mid:.6e}, err vs ref={err:.4f}%")

    # Monotonic convergence
    for k in range(len(errors) - 1):
        assert errors[k + 1] <= errors[k] * 1.05, \
            f"Non-monotonic convergence: {errors}"

    assert errors[-1] < 0.5, f"32-element exponential error {errors[-1]:.4f}% > 0.5%"
    print("OK Piecewise-exponential load: converges monotonically with mesh refinement")
    return True


# ===========================================================================
# Test 6 – Bending moment and shear force diagrams (sinusoidal, 20 elements)
# ===========================================================================
def test_forces_inside_beam_sinusoidal_eb():
    """
    Verify bending-moment and shear-force values at several interior positions
    for a globally sinusoidal load applied with 20 elements (piecewise-linear).

    Analytical (simply supported, q(x) = q0*sin(pi*x/L)):
        M(x) = -q0*L^2/pi^2  * sin(pi*x/L)
        V(x) = -q0*L/pi      * cos(pi*x/L)

    Shear forces are evaluated at ELEMENT MIDPOINTS (not at element boundaries)
    where the constant-per-element EB shear force best approximates the true
    value.  Bending moments are checked at element midpoints as well.
    """
    print("\n" + "=" * 60)
    print("Test 6: Forces Inside Beam (sinusoidal, 20 elements) – EB")
    print("=" * 60)

    q0 = -1000.0
    n = 20
    le = L_BEAM / n

    def M_ana(x):
        return -q0 * L_BEAM**2 / np.pi**2 * np.sin(np.pi * x / L_BEAM)

    def V_ana(x):
        return -q0 * L_BEAM / np.pi * np.cos(np.pi * x / L_BEAM)

    mesh, nodes = _make_simply_supported(n)
    _apply_piecewise_sin(mesh, n, q0)
    displacements, results = _solve(mesh)

    # Choose element indices whose midpoints are well away from V=0 (x=L/2)
    check_elems = [3, 5, 13, 15]   # element indices (0-based) in a 20-element mesh
    tol_M = 2.0   # %
    tol_V = 2.0   # % (shear at midpoints converges well)

    header = (f"{'elem':>5}  {'x_mid':>6}  {'M_FEM':>12}  {'M_Ana':>12}  {'errM%':>7}  "
              f"{'V_FEM':>12}  {'V_Ana':>12}  {'errV%':>7}")
    print(header)
    for i_elem in check_elems:
        x_mid = (i_elem + 0.5) * le          # midpoint of element in global coords
        x_loc = le / 2                        # local position = midpoint of element

        er = results.element_results[i_elem]
        M_fem = er.bending_moment(x_loc)
        V_fem = er.shear_force(x_loc)
        M_a = M_ana(x_mid)
        V_a = V_ana(x_mid)

        err_M = abs(M_fem - M_a) / abs(M_a) * 100 if abs(M_a) > 1e-10 else 0.0
        err_V = abs(V_fem - V_a) / abs(V_a) * 100 if abs(V_a) > 1e-10 else 0.0

        print(f"{i_elem:5d}  {x_mid:6.3f}  {M_fem:12.4e}  {M_a:12.4e}  {err_M:7.3f}%  "
              f"{V_fem:12.4e}  {V_a:12.4e}  {err_V:7.3f}%")

        if abs(M_a) > 1e-10:
            assert err_M < tol_M, f"Moment error at elem {i_elem}: {err_M:.3f}% > {tol_M}%"
        if abs(V_a) > abs(q0) * L_BEAM / np.pi * 0.1:
            assert err_V < tol_V, f"Shear error at elem {i_elem}: {err_V:.3f}% > {tol_V}%"

    print("OK Moment and shear values match analytical within tolerance")
    return True


# ===========================================================================
# Test 7 – Cantilever with uniform load (multi-element for moment accuracy)
# ===========================================================================
def test_cantilever_uniform_load_eb():
    """
    Cantilever EB beam (fixed at x=0, free at x=L).

    Uniform load q0 applied via func.

    For this load case the FEM solution converges to:
        w_tip  = q0*L^4 / (8*EI)            (exact for any number of elements)
        M(0)   = q0*L^2/2                    (M = EI*w''; negative for downward load)

    The tip deflection is checked with a single element (exact).
    Root moment accuracy is tested with increasing number of elements.
    """
    print("\n" + "=" * 60)
    print("Test 7: Cantilever – Uniform Load – EB")
    print("=" * 60)

    q0 = -1000.0   # N/m downward

    # Analytical values
    w_tip_ana = q0 * L_BEAM**4 / (8 * EI)
    M_root_ana = q0 * L_BEAM**2 / 2      # = EI*w''(0), negative (hogging)

    # --- Single element: tip deflection is exact ---
    mesh1, nodes1 = _make_cantilever(1)
    load1 = DistributedLoad(direction="t", func=str(q0))
    load1.element = mesh1.elements[0]
    mesh1.distributed_loads.append(load1)
    disp1, res1 = _solve(mesh1)

    tip_v = disp1[3 * (nodes1[-1].id - 1) + 1]
    ratio_w = tip_v / w_tip_ana
    print(f"  [n=1] Tip deflection: FEM={tip_v:.6e}, Ana={w_tip_ana:.6e}, ratio={ratio_w:.6f}")
    assert abs(ratio_w - 1.0) < 0.01, f"Tip deflection error > 1%: ratio={ratio_w:.6f}"

    # --- Multi-element: root moment convergence ---
    print(f"  Expected root moment (analytical): {M_root_ana:.4e}")
    for n in [4, 8, 16]:
        mesh, nodes = _make_cantilever(n)
        for el in mesh.elements:
            ld = DistributedLoad(direction="t", func=str(q0))
            ld.element = el
            mesh.distributed_loads.append(ld)
        disp, res = _solve(mesh)

        M_root_fem = res.element_results[0].bending_moment(0.0)
        err_M = abs(M_root_fem - M_root_ana) / abs(M_root_ana) * 100
        print(f"  [n={n:2d}] M(0): FEM={M_root_fem:.4e}, Ana={M_root_ana:.4e}, err={err_M:.2f}%")

        if n >= 8:
            assert err_M < 1.0, f"n={n}: Root moment error {err_M:.2f}% > 1%"

    print("OK Cantilever: tip deflection exact (n=1), root moment converges with refinement")
    return True


# ===========================================================================
# Test 8 – Cantilever with sinusoidal load (single element)
# ===========================================================================
def test_cantilever_sinusoidal_load_eb():
    """
    Cantilever EB beam, sinusoidal load:
        q(x) = q0 * sin(pi*x / (2*L))   (zero at root, maximum at free tip)

    Analytical reference via the unit-load (virtual-work) theorem:
        M(s) = integral_s^L q(t)*(t-s) dt   (moment from free-end equilibrium)
        w_tip = integral_0^L (L-s) * M(s) / EI ds

    Single-element test:
        - Custom ``func`` evaluated in local x ∈ [0, L_BEAM] gives the exact
          load shape on a one-element beam.
        - Tip deflection is exact (Gauss quadrature + exact nodal solution).
        - Root moment M(0) from cubic interpolation has ~11% single-element
          error and is NOT checked here.

    Multi-element test (piecewise-linear per element):
        - With 8 elements the root moment converges to < 1% of analytical.
    """
    print("\n" + "=" * 60)
    print("Test 8: Cantilever – Sinusoidal Load – EB")
    print("=" * 60)

    q0 = -1000.0   # N/m downward

    def M_statics(x0):
        """M(x0) from moment equilibrium of free-body [x0, L]."""
        val, _ = quad(lambda s: q0 * np.sin(np.pi * s / (2 * L_BEAM)) * (s - x0),
                      x0, L_BEAM)
        return val   # negative for downward load (hogging)

    # Analytical values
    w_tip_ana, _ = quad(lambda s: (L_BEAM - s) * M_statics(s) / EI, 0, L_BEAM)
    M_root_ana = M_statics(0.0)
    print(f"  Analytical tip deflection: {w_tip_ana:.6e}")
    print(f"  Analytical root moment:    {M_root_ana:.6e}")

    # ------------------------------------------------------------------
    # Single-element: tip deflection (exact thanks to Gauss quadrature)
    # ------------------------------------------------------------------
    mesh, nodes = _make_cantilever(1)
    load = DistributedLoad(direction="t", func=f"{q0}*np.sin(np.pi*x/(2*L))")
    load.element = mesh.elements[0]
    mesh.distributed_loads.append(load)

    displacements, _ = _solve(mesh)

    tip_v = displacements[3 * (nodes[-1].id - 1) + 1]
    ratio_w = tip_v / w_tip_ana
    print(f"  [n=1] Tip deflection: FEM={tip_v:.6e}, Ana={w_tip_ana:.6e}, "
          f"ratio={ratio_w:.6f}")
    assert abs(ratio_w - 1.0) < 0.01, f"Tip deflection error > 1%: ratio={ratio_w:.6f}"

    # ------------------------------------------------------------------
    # Multi-element: root moment convergence (piecewise-linear load)
    # ------------------------------------------------------------------
    print(f"  Expected root moment: {M_root_ana:.4e}")
    for n in [4, 8]:
        mesh, nodes = _make_cantilever(n)
        le = L_BEAM / n
        for i, el in enumerate(mesh.elements):
            xs = i * le
            xe = (i + 1) * le
            qs = q0 * np.sin(np.pi * xs / (2 * L_BEAM))
            qe = q0 * np.sin(np.pi * xe / (2 * L_BEAM))
            ld = DistributedLoad(magnitude_start=qs, magnitude_end=qe, direction="t")
            ld.element = el
            mesh.distributed_loads.append(ld)
        disp, res = _solve(mesh)

        M_root_fem = res.element_results[0].bending_moment(0.0)
        err_M = abs(M_root_fem - M_root_ana) / abs(M_root_ana) * 100
        print(f"  [n={n:2d}] M(0): FEM={M_root_fem:.4e}, err={err_M:.2f}%")

        if n >= 8:
            assert err_M < 1.0, f"n={n}: Root moment error {err_M:.2f}% > 1%"

    print("OK Cantilever sinusoidal: tip deflection exact (n=1), "
          "root moment converges with refinement")
    return True


# ===========================================================================
# Test 9 – Mesh convergence: sinusoidal load (mid-span deflection)
# ===========================================================================
def test_mesh_convergence_sinusoidal_eb():
    """
    Demonstrate convergence of the mid-span deflection to the analytical value
    as the number of elements increases for a globally sinusoidal load
    applied as piecewise-linear per element.

    Analytical: w_max = q0*L^4 / (EI*pi^4).
    """
    print("\n" + "=" * 60)
    print("Test 9: Mesh Convergence – Sinusoidal Load – EB")
    print("=" * 60)

    q0 = -1000.0
    w_ana = q0 * L_BEAM**4 / (EI * np.pi**4)

    prev_err = None
    final_err = None
    for n in [2, 4, 8, 16]:
        mesh, nodes = _make_simply_supported(n)
        _apply_piecewise_sin(mesh, n, q0)
        displacements, _ = _solve(mesh)

        mid_id = nodes[n // 2].id
        w_mid = displacements[3 * (mid_id - 1) + 1]
        err = abs(w_mid - w_ana) / abs(w_ana) * 100
        print(f"  n={n:2d}: w_mid={w_mid:.6e}, ana={w_ana:.6e}, err={err:.4f}%")

        if prev_err is not None:
            assert err <= prev_err * 1.1, \
                f"Non-monotonic convergence at n={n}: {err:.4f}% > {prev_err:.4f}%"
        prev_err = err
        final_err = err

    assert final_err < 1.0, f"16-element error {final_err:.4f}% > 1.0%"
    print("OK Mesh convergence verified: error decreases with refinement")
    return True


# ===========================================================================
# Main runner
# ===========================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RUNNING EULER-BERNOULLI DISTRIBUTED LOAD TESTS")
    print("=" * 60)

    tests = [
        ("Test 1: Uniform Load (func) – Single Elem",       test_uniform_load_single_element_func),
        ("Test 2: Sinusoidal Load – Single Elem",           test_sinusoidal_load_single_element_eb),
        ("Test 3: Exponential Load – Single Elem",          test_exponential_load_single_element_eb),
        ("Test 4: Sinusoidal Load – Multi-Elem",            test_sinusoidal_load_multi_element_eb),
        ("Test 5: Exponential Load – Multi-Elem",           test_exponential_load_multi_element_eb),
        ("Test 6: Forces (Moment & Shear, 20 elem)",        test_forces_inside_beam_sinusoidal_eb),
        ("Test 7: Cantilever – Uniform Load",               test_cantilever_uniform_load_eb),
        ("Test 8: Cantilever – Sinusoidal Load",            test_cantilever_sinusoidal_load_eb),
        ("Test 9: Mesh Convergence – Sinusoidal",           test_mesh_convergence_sinusoidal_eb),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"FAILED [{name}]: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"ERROR  [{name}]: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    if failed:
        sys.exit(1)
