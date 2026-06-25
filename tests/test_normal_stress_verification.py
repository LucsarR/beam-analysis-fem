import numpy as np
import pytest

from fem.mesh import Mesh
from fem.material import Material
from fem.section import RectangularBar
from fem.constraint import Constraint
from fem.load import PointLoad, DistributedLoad
from fem.analysis import BeamAnalysis
from post_processing.forces import StructureResults

def fit_residual(x_vals, y_vals, deg):
    """Fit a polynomial of degree `deg` and return the sum of squared residuals."""
    coeffs = np.polyfit(x_vals, y_vals, deg)
    y_fit = np.polyval(coeffs, x_vals)
    return float(np.sum((y_vals - y_fit) ** 2))

def solve_beam(etype):
    # Setup standard parameters
    L = 2.0
    B = 0.05
    H = 0.1
    E = 210e9
    NU = 0.3
    
    mesh = Mesh()
    mat = Material(1, E, NU)
    sec = RectangularBar(1, B, H)
    
    # 1 element mesh
    nodes = mesh.generate_1d_mesh(0, 0, L, 0, 1, mat, sec, etype)
    
    # Fix left end
    left_node = nodes[0]
    mesh.constraints.add(Constraint(left_node, 0, 0.0))  # u
    mesh.constraints.add(Constraint(left_node, 1, 0.0))  # v
    mesh.constraints.add(Constraint(left_node, 2, 0.0))  # theta
    if "reddy" in etype:
        mesh.constraints.add(Constraint(left_node, 3, 0.0))  # dv/dx
        
    # Apply combined tension (point load) and transverse bending (distributed load)
    tip_node = nodes[-1]
    load_x = PointLoad(1.0e5, 0)
    load_x.node = tip_node
    mesh.point_loads.append(load_x)
    
    for el in mesh.elements:
        dist_load = DistributedLoad(magnitude_start=-1.0e4, magnitude_end=0.0, direction='y')
        dist_load.element = el
        mesh.distributed_loads.append(dist_load)
        
    # Solve
    analysis = BeamAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    
    struct_res = StructureResults(mesh, displacements)
    return struct_res.element_results[0], L, H

@pytest.mark.parametrize("etype, expected_y, expected_x", [
    ("euler_bernoulli_2node", "LINEAR", "LINEAR"),
    ("euler_bernoulli_3node", "LINEAR", "CUBIC"),
    ("timoshenko_2node",      "LINEAR", "LINEAR"),
    ("timoshenko_3node",      "LINEAR", "LINEAR"),
    ("reddy_bickford_2node",  "CUBIC",  "LINEAR"),
])
def test_normal_stress_distributions(etype, expected_y, expected_x):
    """
    Verify the polynomial degrees of kinematic normal stress distributions
    along both the cross-section height (y) and the beam span (x).
    """
    er, L, H = solve_beam(etype)
    
    # 1. Verify distribution along height y (at x = L/2)
    x_mid = L / 2.0
    ys = np.linspace(-H/2.0, H/2.0, 11)
    sigma_ys = np.array([er.kinematic_normal_stress(x_mid, y) for y in ys])
    
    res_y_deg1 = fit_residual(ys, sigma_ys, 1)
    res_y_deg3 = fit_residual(ys, sigma_ys, 3)
    
    if expected_y == "LINEAR":
        assert res_y_deg1 < 1e-10, f"{etype}: expected linear stress in y (residual={res_y_deg1:.1e})"
    elif expected_y == "CUBIC":
        assert res_y_deg1 > 1e-2, f"{etype}: expected nonlinear stress in y (residual={res_y_deg1:.1e})"
        assert res_y_deg3 < 1e-10, f"{etype}: expected cubic stress in y (residual={res_y_deg3:.1e})"
        
    # 2. Verify distribution along length x (at y = H/2)
    y_top = H / 2.0
    xs = np.linspace(0.0, L, 11)
    sigma_xs = np.array([er.kinematic_normal_stress(x, y_top) for x in xs])
    
    res_x_deg1 = fit_residual(xs, sigma_xs, 1)
    res_x_deg3 = fit_residual(xs, sigma_xs, 3)
    
    if expected_x == "LINEAR":
        assert res_x_deg1 < 1e-10, f"{etype}: expected linear stress in x (residual={res_x_deg1:.1e})"
    elif expected_x == "CUBIC":
        assert res_x_deg1 > 1.0, f"{etype}: expected nonlinear stress in x (residual={res_x_deg1:.1e})"
        assert res_x_deg3 < 1e-10, f"{etype}: expected cubic stress in x (residual={res_x_deg3:.1e})"
