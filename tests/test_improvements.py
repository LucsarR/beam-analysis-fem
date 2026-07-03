import pytest
import numpy as np

from fem.mesh import Mesh
from fem.material import Material
from fem.section import RectangularBar
from fem.constraint import Constraint
from fem.load import PointLoad
from fem.analysis import BeamAnalysis
from post_processing.forces import StructureResults
from post_processing.plotter import (
    plot_deformed_shape,
    plot_normal_stress_distribution,
    plot_shear_stress_distribution,
    plot_reddy_shear_stress_distribution,
    plot_normal_stress_side_view
)

# Test properties
E = 210e9
NU = 0.3
B = 0.05
h = 0.20
L_BEAM = 2.0

def test_deformed_shape_central_node_displacement():
    """Verify that the central node of a 3-node element has its displacement correctly retrieved in plot_deformed_shape."""
    mesh = Mesh()
    mat = Material(1, E, NU)
    sec = RectangularBar(1, B, h)
    
    # Generate 1D mesh with 3-node elements
    nodes = mesh.generate_1d_mesh(0, 0, L_BEAM, 0, 2, mat, sec, "euler_bernoulli_3node")
    
    # Boundary conditions: Clamped at left node (Node 1)
    mesh.constraints.add(Constraint(nodes[0], 0, 0.0))  # u
    mesh.constraints.add(Constraint(nodes[0], 1, 0.0))  # v
    mesh.constraints.add(Constraint(nodes[0], 2, 0.0))  # theta
    
    # Downward point load at right tip (Node 3)
    load = PointLoad(-1000.0, 1)
    load.node = nodes[-1]
    mesh.point_loads.append(load)
    
    analysis = BeamAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    results = StructureResults(mesh, displacements)
    
    # Generate plot_deformed_shape (returns fig, scale_factor)
    fig, sf = plot_deformed_shape(results, show_original=True)
    
    # Find trace for Node 2 (central node)
    node2_trace = None
    for trace in fig.data:
        if getattr(trace, 'name', None) == 'Node 2':
            node2_trace = trace
            break
            
    assert node2_trace is not None, "Node 2 trace should be found"
    
    # Deformed position of Node 2
    # The central node should be deflected downwards under downward load
    yd = node2_trace.y[0]
    assert yd < 0.0, f"Central node deformed y-coordinate should be negative (deformed), got {yd}"
    
    # And it should not be exactly 0 (which would be undeformed)
    assert not np.isclose(yd, 0.0, atol=1e-12)

def test_reddy_jourawski_sign_alignment():
    """Verify that Reddy and Jourawski shear stresses have the same sign convention."""
    mesh = Mesh()
    mat = Material(1, E, NU)
    sec = RectangularBar(1, B, h)
    nodes = mesh.generate_1d_mesh(0, 0, L_BEAM, 0, 4, mat, sec, "reddy_bickford_2node")
    
    mesh.constraints.add(Constraint(nodes[0], 0, 0.0))  # u
    mesh.constraints.add(Constraint(nodes[0], 1, 0.0))  # v
    mesh.constraints.add(Constraint(nodes[0], 2, 0.0))  # theta
    mesh.constraints.add(Constraint(nodes[0], 3, 0.0))  # dv/dx
    
    # Downward point load at tip
    load = PointLoad(-1000.0, 1)
    load.node = nodes[-1]
    mesh.point_loads.append(load)
    
    analysis = BeamAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    results = StructureResults(mesh, displacements)
    
    er = results.element_results[0]
    
    # Evaluate at x = L/2 (midpoint of first element)
    x_eval = er.length * 0.5
    tau_j = er.jourawski_shear_stress(x_eval, 0.0)
    tau_r = er.reddy_shear_stress(x_eval, 0.0)
    
    assert tau_j > 0, f"Jourawski shear stress at neutral axis under positive shear should be positive, got {tau_j}"
    assert tau_r > 0, f"Reddy shear stress at neutral axis under positive shear should be positive, got {tau_r}"
    assert np.sign(tau_j) == np.sign(tau_r), f"Signs should match: Jourawski={tau_j}, Reddy={tau_r}"

def test_jourawski_caching():
    """Verify that jourawski_shear_stress correctly caches cross-section grid properties."""
    mesh = Mesh()
    mat = Material(1, E, NU)
    sec = RectangularBar(1, B, h)
    nodes = mesh.generate_1d_mesh(0, 0, L_BEAM, 0, 2, mat, sec, "euler_bernoulli_2node")
    
    mesh.constraints.add(Constraint(nodes[0], 0, 0.0))  # u
    mesh.constraints.add(Constraint(nodes[0], 1, 0.0))  # v
    mesh.constraints.add(Constraint(nodes[0], 2, 0.0))  # theta
    
    analysis = BeamAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    results = StructureResults(mesh, displacements)
    er = results.element_results[0]
    
    # Ensure cache attribute is not present initially
    cache_key = "_jourawski_cache_100"
    assert not hasattr(er, cache_key)
    
    # First call - populates cache
    val1 = er.jourawski_shear_stress(0.0, 0.0, n_points=100)
    assert hasattr(er, cache_key)
    
    # Second call - retrieves from cache
    val2 = er.jourawski_shear_stress(0.0, 0.0, n_points=100)
    assert np.isclose(val1, val2)

def test_cross_section_axes_ranges():
    """Verify that cross-section stress distribution plots have explicit centered ranges set."""
    mesh = Mesh()
    mat = Material(1, E, NU)
    sec = RectangularBar(1, B, h)
    nodes = mesh.generate_1d_mesh(0, 0, L_BEAM, 0, 2, mat, sec, "reddy_bickford_2node")
    
    mesh.constraints.add(Constraint(nodes[0], 0, 0.0))  # u
    mesh.constraints.add(Constraint(nodes[0], 1, 0.0))  # v
    mesh.constraints.add(Constraint(nodes[0], 2, 0.0))  # theta
    mesh.constraints.add(Constraint(nodes[0], 3, 0.0))  # dv/dx
    
    analysis = BeamAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    results = StructureResults(mesh, displacements)
    er = results.element_results[0]
    
    # Plot normal stress
    fig_n = plot_normal_stress_distribution(er, x=0.0, query_y=0.0)
    assert fig_n.layout.xaxis.range is not None
    assert fig_n.layout.yaxis.range is not None
    
    # Verify that the ranges are close to the cross-section bounds
    y_range = fig_n.layout.yaxis.range
    # Height of RectangularBar is 0.20, so y bounds are -0.10 and 0.10
    # Expected y range with 15% margin is roughly [-0.13, 0.13]
    assert np.isclose(y_range[0], -0.13, atol=0.01)
    assert np.isclose(y_range[1], 0.13, atol=0.01)
    
    # Plot Jourawski shear stress
    fig_s = plot_shear_stress_distribution(er, x=0.0, query_y=0.0)
    assert fig_s.layout.xaxis.range is not None
    assert fig_s.layout.yaxis.range is not None
    
    # Plot Reddy shear stress
    fig_r = plot_reddy_shear_stress_distribution(er, x=0.0, query_y=0.0)
    assert fig_r.layout.xaxis.range is not None
    assert fig_r.layout.yaxis.range is not None

def test_normal_stress_side_view_arrow_annotations():
    """Verify that plot_normal_stress_side_view sets pixel-based arrowheads."""
    mesh = Mesh()
    mat = Material(1, E, NU)
    sec = RectangularBar(1, B, h)
    nodes = mesh.generate_1d_mesh(0, 0, L_BEAM, 0, 2, mat, sec, "euler_bernoulli_2node")
    
    mesh.constraints.add(Constraint(nodes[0], 0, 0.0))  # u
    mesh.constraints.add(Constraint(nodes[0], 1, 0.0))  # v
    mesh.constraints.add(Constraint(nodes[0], 2, 0.0))  # theta
    
    # Downward point load at tip
    load = PointLoad(-1000.0, 1)
    load.node = nodes[-1]
    mesh.point_loads.append(load)
    
    analysis = BeamAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    results = StructureResults(mesh, displacements)
    er = results.element_results[0]
    
    fig = plot_normal_stress_side_view(er, x=0.0, display_x=0.0, display_length=2.0)
    
    # Find arrow annotations (exclude text labels for Top and Bottom fibers which have showarrow=False)
    arrow_annotations = [ann for ann in fig.layout.annotations if ann.showarrow]
    
    assert len(arrow_annotations) > 0, "Should have arrow annotations"
    for ann in arrow_annotations:
        assert ann.axref == 'pixel'
        assert ann.ayref == 'pixel'
        assert abs(ann.ax) > 0
        assert ann.ay == 0
