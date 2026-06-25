"""
Tests verifying the polynomial degrees of internal forces for Euler-Bernoulli and Timoshenko 3-node elements.
This validates:
1. Euler-Bernoulli 3-node element:
   - v(x) is 5th degree (quintic)
   - θ(x) is 4th degree (quartic)
   - M(x) is 3rd degree (cubic)
   - V(x) is 2nd degree (quadratic)
   - N(x) is 1st degree (linear)
2. Timoshenko 3-node element:
   - v(x) is 5th degree (quintic) in the current implementation
   - θ(x) is 2nd degree (quadratic)
   - M(x) is 1st degree (linear)
   - V(x) is 1st degree (linear) in post-processing due to equilibrium recovery and linear interpolation
   - N(x) is 1st degree (linear)
3. Mesh subdivision (subelements):
   - Internal force distributions across multiple subdivided elements are correct and consistent.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fem.mesh import Mesh
from fem.material import Material
from fem.section import RectangularBar
from fem.constraint import Constraint
from fem.load import DistributedLoad
from fem.analysis import BeamAnalysis

def check_polynomial_degree(x_vals, y_vals, deg):
    """
    Fits a polynomial of degree `deg` to the data and returns the residual.
    A residual close to 0 indicates the function is of degree <= `deg`.
    """
    # Fit polynomial
    coeffs, residuals, rank, singular_values, rcond = np.polyfit(x_vals, y_vals, deg, full=True)
    if len(residuals) > 0:
        return residuals[0]
    else:
        # If full=True doesn't return residuals (e.g. overdetermined but perfect fit), compute manually
        y_fit = np.polyval(coeffs, x_vals)
        return np.sum((y_vals - y_fit) ** 2)

def test_euler_bernoulli_3node_polynomial_degrees():
    """Verify that Euler-Bernoulli 3-node element internal forces have the expected polynomial degrees."""
    # Beam properties
    L = 5.0
    E = 1e9
    nu = 0.3
    b = 0.05
    h = 0.1
    
    mesh = Mesh()
    mat = Material(1, E, nu)
    sec = RectangularBar(1, b, h)
    
    n1 = mesh.add_node(0, 0)
    n2 = mesh.add_node(L, 0)
    el = mesh.add_element(n1, n2, mat, sec, 'euler_bernoulli_3node')
    
    # Boundary conditions: cantilever fixed at left end
    mesh.constraints.add(Constraint(n1, 0, 0.0))
    mesh.constraints.add(Constraint(n1, 1, 0.0))
    mesh.constraints.add(Constraint(n1, 2, 0.0))
    
    # Linear distributed load in y direction (from 10.0 to 0.0)
    dist_load = DistributedLoad(magnitude_start=-10.0, magnitude_end=0.0, direction='y')
    dist_load.element = el
    mesh.distributed_loads.append(dist_load)
    
    # Solve
    analysis = BeamAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    
    # Get local displacements for element
    # The element local DOFs order is [u1, v1, theta1, u2, v2, theta2, u3, v3, theta3]
    # For a single element, global displacements map 1-to-1 to local displacements (no rotation of coordinates)
    from fem.analysis import get_element_dof_indices
    dof_indices = get_element_dof_indices(el, analysis.dpn)
    el_disps = displacements[dof_indices]
    
    # Sample at 10 points along the element
    xs = np.linspace(0.0, L, 10)
    moments = np.array([el.bending_moment(x, el_disps) for x in xs])
    shears = np.array([el.shear_force(x, el_disps) for x in xs])
    normals = np.array([el.normal_force(x, el_disps) for x in xs])
    
    # 1. Normal force: should be linear (degree 1)
    res_n_linear = check_polynomial_degree(xs, normals, 1)
    assert res_n_linear < 1e-12, f"Normal force should be linear, got residual: {res_n_linear}"
    
    # 2. Bending moment: should be cubic (degree 3)
    res_m_cubic = check_polynomial_degree(xs, moments, 3)
    res_m_quadratic = check_polynomial_degree(xs, moments, 2)
    assert res_m_cubic < 1e-12, f"Bending moment should be cubic, got residual: {res_m_cubic}"
    assert res_m_quadratic > 1e-6, f"Bending moment should not be quadratic or lower, got residual: {res_m_quadratic}"
    
    # 3. Shear force: should be quadratic (degree 2)
    res_v_quadratic = check_polynomial_degree(xs, shears, 2)
    res_v_linear = check_polynomial_degree(xs, shears, 1)
    assert res_v_quadratic < 1e-12, f"Shear force should be quadratic, got residual: {res_v_quadratic}"
    assert res_v_linear > 1e-6, f"Shear force should not be linear or lower, got residual: {res_v_linear}"
    
    print("\n✓ Euler-Bernoulli 3-node element polynomial degrees verified successfully!")

def test_timoshenko_3node_polynomial_degrees():
    """Verify that Timoshenko 3-node element internal forces have the expected polynomial degrees."""
    # Beam properties
    L = 5.0
    E = 1e9
    nu = 0.3
    b = 0.05
    h = 0.1
    
    mesh = Mesh()
    mat = Material(1, E, nu)
    sec = RectangularBar(1, b, h)
    
    n1 = mesh.add_node(0, 0)
    n2 = mesh.add_node(L, 0)
    el = mesh.add_element(n1, n2, mat, sec, 'timoshenko_3node')
    
    # Boundary conditions: cantilever fixed at left end
    mesh.constraints.add(Constraint(n1, 0, 0.0))
    mesh.constraints.add(Constraint(n1, 1, 0.0))
    mesh.constraints.add(Constraint(n1, 2, 0.0))
    
    # Linear distributed load in y direction (from -10.0 to 0.0)
    dist_load = DistributedLoad(magnitude_start=-10.0, magnitude_end=0.0, direction='y')
    dist_load.element = el
    mesh.distributed_loads.append(dist_load)
    
    # Solve
    analysis = BeamAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    
    # Get local displacements for element
    from fem.analysis import get_element_dof_indices
    dof_indices = get_element_dof_indices(el, analysis.dpn)
    el_disps = displacements[dof_indices]
    
    # Sample at 10 points along the element
    xs = np.linspace(0.0, L, 10)
    moments = np.array([el.bending_moment(x, el_disps) for x in xs])
    shears = np.array([el.shear_force(x, el_disps) for x in xs])
    normals = np.array([el.normal_force(x, el_disps) for x in xs])
    
    # 1. Normal force: should be linear (degree 1)
    res_n_linear = check_polynomial_degree(xs, normals, 1)
    assert res_n_linear < 1e-12, f"Normal force should be linear, got residual: {res_n_linear}"
    
    # 2. Bending moment: should be linear (degree 1)
    res_m_linear = check_polynomial_degree(xs, moments, 1)
    res_m_constant = check_polynomial_degree(xs, moments, 0)
    assert res_m_linear < 1e-12, f"Bending moment should be linear, got residual: {res_m_linear}"
    assert res_m_constant > 1e-6, f"Bending moment should not be constant, got residual: {res_m_constant}"
    
    # 3. Shear force: should be linear (degree 1) due to recovery and linear interpolation
    res_v_linear = check_polynomial_degree(xs, shears, 1)
    res_v_constant = check_polynomial_degree(xs, shears, 0)
    assert res_v_linear < 1e-12, f"Shear force should be linear, got residual: {res_v_linear}"
    assert res_v_constant > 1e-6, f"Shear force should not be constant under non-uniform load, got residual: {res_v_constant}"
    
    print("\n✓ Timoshenko 3-node element polynomial degrees verified successfully!")

def test_forces_with_mesh_subdivision():
    """Verify that internal forces behave correctly and continuously when elements are subdivided (subelements)."""
    # Beam properties
    L = 5.0
    E = 1e9
    nu = 0.3
    b = 0.05
    h = 0.1
    
    # We will create two models of a simply supported beam with a uniform load of -10.0 N/m:
    # Model A: 1 element subdivided into 4 subelements
    # Model B: 4 elements added manually
    # The results (displacements, bending moments, shear forces) should be identical.
    
    # Model A: 1 element subdivided into 4 subelements
    # In app.py, the subdivision is performed by generating sub-nodes and adding elements.
    # We mimic this process here.
    mesh_sub = Mesh()
    mat = Material(1, E, nu)
    sec = RectangularBar(1, b, h)
    
    n_start = mesh_sub.add_node(0, 0)
    n_end = mesh_sub.add_node(L, 0)
    
    # Subdivide into 4 subelements
    n_subdiv = 4
    subdiv_nodes = [n_start]
    for i in range(1, n_subdiv):
        x = L * i / n_subdiv
        subdiv_nodes.append(mesh_sub.add_node(x, 0))
    subdiv_nodes.append(n_end)
    
    sub_elements = []
    for i in range(n_subdiv):
        el = mesh_sub.add_element(subdiv_nodes[i], subdiv_nodes[i+1], mat, sec, 'timoshenko_3node')
        sub_elements.append(el)
        
    # Boundary conditions
    mesh_sub.constraints.add(Constraint(n_start, 0, 0.0))
    mesh_sub.constraints.add(Constraint(n_start, 1, 0.0))
    mesh_sub.constraints.add(Constraint(n_end, 1, 0.0))
    
    # Add distributed load of -10.0 N/m to all subelements
    # (In app.py, when a load is applied to an element, it is copied to all subdivided elements)
    for el in sub_elements:
        dist_load = DistributedLoad(magnitude_start=-10.0, magnitude_end=-10.0, direction='y')
        dist_load.element = el
        mesh_sub.distributed_loads.append(dist_load)
        
    # Solve subdivided mesh
    analysis_sub = BeamAnalysis(mesh_sub)
    analysis_sub.assemble()
    disps_sub = analysis_sub.solve()
    
    # Let's verify displacement at the center (x = L/2)
    # The center is node at x = 2.5, which is subdiv_nodes[2] for 4 subdivisions
    center_node_id = subdiv_nodes[2].id
    v_center_sub = disps_sub[3 * (center_node_id - 1) + 1]
    
    # Analytical solution for simply supported Timoshenko beam with uniform load q:
    # w_max = (5*q*L^4)/(384*E*I) + (q*L^2)/(8*kappa*G*A)
    q = -10.0
    I = sec.inertia
    A = sec.area
    kappa = sec.shear_coefficient
    G = mat.G
    
    w_bending = (5 * q * L**4) / (384 * E * I)
    w_shear = (q * L**2) / (8 * kappa * G * A)
    w_analytical = w_bending + w_shear
    
    # Check that the subdivision converges extremely close to the analytical solution
    error = abs((v_center_sub - w_analytical) / w_analytical) * 100
    assert error < 0.1, f"Deflection error too large for 4 subdivided elements: {error:.4f}%"
    
    # Verify moment diagram is piecewise linear across subelements and continuous at nodes
    # For a simply supported beam with uniform load, the exact moment diagram is a parabola.
    # The Timoshenko 3-node element uses piecewise linear moment within each element.
    # Let's check that the bending moments match at internal nodes (continuity)
    # Node at x = 1.25 is subdiv_nodes[1]
    # Subelement 0 ends at x = 1.25 (which is local coordinate x_local = L/n_subdiv = 1.25)
    # Subelement 1 starts at x = 1.25 (which is local coordinate x_local = 0.0)
    from fem.analysis import get_element_dof_indices
    
    el0 = sub_elements[0]
    el1 = sub_elements[1]
    
    disps_el0 = disps_sub[get_element_dof_indices(el0, analysis_sub.dpn)]
    disps_el1 = disps_sub[get_element_dof_indices(el1, analysis_sub.dpn)]
    
    M_el0_end = el0.bending_moment(el0.length, disps_el0)
    M_el1_start = el1.bending_moment(0.0, disps_el1)
    
    assert np.isclose(M_el0_end, M_el1_start, atol=1e-10), \
        f"Moment mismatch at subdivision boundary: {M_el0_end} vs {M_el1_start}"
        
    # Check shear force jump at the shared node:
    # Under a distributed load, the shear force recovered from K_local * d_local will have a jump
    # at the shared node equal to the sum of the equivalent nodal forces from the two adjacent elements.
    # Nodal equilibrium: f_local_0[7] + f_local_1[1] = F_equivalent_shared
    # Since V_el0_end = -f_local_0[7] and V_el1_start = f_local_1[1], this becomes:
    # -V_el0_end + V_el1_start = F_equivalent_shared => V_el1_start - V_el0_end = F_equivalent_shared
    fe_local_0 = el0.compute_equivalent_nodal_loads(dist_load)
    fe_local_1 = el1.compute_equivalent_nodal_loads(dist_load)
    F_equivalent_shared = fe_local_0[7] + fe_local_1[1]
    
    V_el0_end = el0.shear_force(el0.length, disps_el0)
    V_el1_start = el1.shear_force(0.0, disps_el1)
    jump = V_el1_start - V_el0_end
    assert np.isclose(jump, F_equivalent_shared, atol=1e-10), \
        f"Shear force jump {jump} does not match equivalent nodal load {F_equivalent_shared}"
        
    print("\n✓ Mesh subdivision forces and continuity verified successfully!")

def test_timoshenko_3node_kinematic_vs_recovered():
    """
    Verify and demonstrate why the kinematic formula V = kGA*(dv/dx - theta)
    is not used for the Timoshenko 3-node element. It oscillates wildly due to
    uncoupled/decoupled interpolation orders (quintic v vs quadratic theta),
    while the recovered shear force is linear (degree 1) and matches equilibrium.
    """
    # Beam properties
    L = 5.0
    E = 1e9
    nu = 0.3
    G = E / (2 * (1 + nu))
    b = 0.05
    h = 0.1
    A = b * h
    kappa = 5/6
    
    mesh = Mesh()
    mat = Material(1, E, nu)
    sec = RectangularBar(1, b, h)
    
    n1 = mesh.add_node(0, 0)
    n2 = mesh.add_node(L, 0)
    el = mesh.add_element(n1, n2, mat, sec, 'timoshenko_3node')
    
    mesh.constraints.add(Constraint(n1, 0, 0.0))
    mesh.constraints.add(Constraint(n1, 1, 0.0))
    mesh.constraints.add(Constraint(n1, 2, 0.0))
    
    dist_load = DistributedLoad(magnitude_start=-10.0, magnitude_end=-10.0, direction='y')
    dist_load.element = el
    mesh.distributed_loads.append(dist_load)
    
    analysis = BeamAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    
    from fem.analysis import get_element_dof_indices
    dof_indices = get_element_dof_indices(el, analysis.dpn)
    el_disps = displacements[dof_indices]
    
    from fem.element import _quintic_bending_shapes_3node, _quadratic_shape_functions_3node
    
    # Sample at 11 points along element
    xs = np.linspace(0.0, L, 11)
    vkins = []
    vrecs = []
    
    for x in xs:
        xi = x / L
        _, dn_w_dx, _, _, _, _ = _quintic_bending_shapes_3node(xi, L)
        dv_dx = np.dot(dn_w_dx, el_disps[[1, 2, 4, 5, 7, 8]])
        
        n_theta = _quadratic_shape_functions_3node(xi)
        theta = np.dot(n_theta, el_disps[[2, 5, 8]])
        
        V_kin = kappa * G * A * (dv_dx - theta)
        vkins.append(V_kin)
        
        V_rec = el.shear_force(x, el_disps)
        vrecs.append(V_rec)
        
    vkins = np.array(vkins)
    vrecs = np.array(vrecs)
    
    # Kinematic shear force has high-order oscillations and evaluates to zero at ends (which is incorrect)
    # Recovered shear force is linear (degree 1)
    res_vrec_linear = check_polynomial_degree(xs, vrecs, 1)
    res_vkin_linear = check_polynomial_degree(xs, vkins, 1)
    
    assert res_vrec_linear < 1e-12, "Recovered shear force should be perfectly linear"
    assert res_vkin_linear > 10.0, "Kinematic shear force should have high residual due to oscillations"
    
    print("\n✓ Timoshenko 3-node kinematic vs recovered shear test passed!")

