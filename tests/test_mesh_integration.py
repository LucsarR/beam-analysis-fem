"""
Integration tests for the Mesh class to verify it is correctly integrated with
element classes and other components of the FEM framework.

This test suite verifies:
1. Mesh creation and basic operations
2. Node and element management
3. Integration with different element types (EulerBernoulli, Timoshenko)
4. Integration with Material and Section classes
5. Integration with Constraint and Load classes
6. Integration with Analysis classes
7. Integration with post-processing (StructureResults, ElementResults)
8. Mesh generation utilities (generate_1d_mesh)
9. Mesh query and export methods
"""

import numpy as np
from streamlit.testing.v1 import AppTest
from fem.mesh import Mesh
from fem.material import Material
from fem.section import RectangularBar, CircularBar
from fem.constraint import Constraint
from fem.load import PointLoad, DistributedLoad
from fem.analysis import EulerBernoulliAnalysis
from fem.spring import Spring
from post_processing.forces import StructureResults


def test_mesh_creation():
    """Test basic mesh creation and attributes."""
    mesh = Mesh()
    assert mesh.nodes == []
    assert mesh.elements == []
    assert mesh.node_id_counter == 1
    assert mesh.element_id_counter == 1
    assert mesh.point_loads == []
    assert mesh.distributed_loads == []
    assert mesh.constraints is not None
    print("✓ test_mesh_creation passed")


def test_add_nodes():
    """Test adding nodes to the mesh."""
    mesh = Mesh()
    n1 = mesh.add_node(0, 0)
    n2 = mesh.add_node(1, 0)
    n3 = mesh.add_node(2, 0.5)
    
    assert len(mesh.nodes) == 3
    assert n1.id == 1
    assert n2.id == 2
    assert n3.id == 3
    assert n1.x == 0 and n1.y == 0
    assert n2.x == 1 and n2.y == 0
    assert n3.x == 2 and n3.y == 0.5
    assert mesh.node_id_counter == 4
    print("✓ test_add_nodes passed")


def test_add_elements():
    """Test adding elements to the mesh with different types."""
    mesh = Mesh()
    mat = Material(1, 210e9, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)
    
    n1 = mesh.add_node(0, 0)
    n2 = mesh.add_node(1, 0)
    n3 = mesh.add_node(2, 0)
    
    # Add Euler-Bernoulli 2-node element
    el1 = mesh.add_element(n1, n2, mat, sec, 'euler_bernoulli_2node')
    assert el1.id == 1
    assert el1.node_start == n1
    assert el1.node_end == n2
    assert el1.material == mat
    assert el1.section == sec
    
    # Add Timoshenko 2-node element
    el2 = mesh.add_element(n2, n3, mat, sec, 'timoshenko_2node')
    assert el2.id == 2
    assert len(mesh.elements) == 2
    assert mesh.element_id_counter == 3

    # Add Euler-Bernoulli 2-node element with numerical stiffness integration
    el3 = mesh.add_element(
        n1, n3, mat, sec, 'euler_bernoulli_2node',
        stiffness_integration='numerical'
    )
    assert el3.stiffness_integration == 'numerical'
    
    # Test invalid element type
    try:
        mesh.add_element(n1, n2, mat, sec, 'invalid_type')
        assert False, "Should raise NotImplementedError"
    except NotImplementedError:
        pass
    
    print("✓ test_add_elements passed")


def test_euler_bernoulli_numerical_stiffness_matches_analytical():
    """Numerical integration option should remain consistent with analytical EB stiffness."""
    mesh = Mesh()
    mat = Material(1, 210e9, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)

    n1 = mesh.add_node(0, 0)
    n2 = mesh.add_node(2, 0)

    el_analytical = mesh.add_element(
        n1, n2, mat, sec, 'euler_bernoulli_2node',
        stiffness_integration='analytical'
    )
    el_numerical = mesh.add_element(
        n1, n2, mat, sec, 'euler_bernoulli_2node',
        stiffness_integration='numerical'
    )

    k_analytical = el_analytical.stiffness_matrix()
    k_numerical = el_numerical.stiffness_matrix()

    assert np.allclose(k_analytical, k_numerical, rtol=1e-10, atol=1e-8)
    print("✓ test_euler_bernoulli_numerical_stiffness_matches_analytical passed")


def test_element_attributes():
    """Test that all element types have consistent attributes."""
    mesh = Mesh()
    mat = Material(1, 210e9, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)
    
    n1 = mesh.add_node(0, 0)
    n2 = mesh.add_node(1, 0)
    
    # Test EulerBernoulli element
    el_eb = mesh.add_element(n1, n2, mat, sec, 'euler_bernoulli_2node')
    assert hasattr(el_eb, 'R'), "EulerBernoulli element missing R attribute"
    assert hasattr(el_eb, 'c'), "EulerBernoulli element missing c attribute"
    assert hasattr(el_eb, 's'), "EulerBernoulli element missing s attribute"
    assert hasattr(el_eb, 'length'), "EulerBernoulli element missing length attribute"
    
    # Test Timoshenko element
    mesh2 = Mesh()
    n3 = mesh2.add_node(0, 0)
    n4 = mesh2.add_node(1, 0)
    el_tim = mesh2.add_element(n3, n4, mat, sec, 'timoshenko_2node')
    assert hasattr(el_tim, 'R'), "Timoshenko element missing R attribute"
    assert hasattr(el_tim, 'c'), "Timoshenko element missing c attribute"
    assert hasattr(el_tim, 's'), "Timoshenko element missing s attribute"
    assert hasattr(el_tim, 'length'), "Timoshenko element missing length attribute"
    
    print("✓ test_element_attributes passed")


def test_generate_1d_mesh():
    """Test automatic 1D mesh generation."""
    mesh = Mesh()
    mat = Material(1, 210e9, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)
    
    # Generate mesh with 5 elements
    nodes = mesh.generate_1d_mesh(0, 0, 5, 0, 5, mat, sec, 'euler_bernoulli_2node')
    
    assert len(nodes) == 6  # n_elements + 1
    assert len(mesh.nodes) == 6
    assert len(mesh.elements) == 5
    
    # Verify node positions
    for i, node in enumerate(nodes):
        assert abs(node.x - i) < 1e-10
        assert abs(node.y - 0) < 1e-10
        assert node.id == i + 1
    
    # Verify element connectivity
    for i, element in enumerate(mesh.elements):
        assert element.node_start.id == i + 1
        assert element.node_end.id == i + 2
        assert element.id == i + 1
    
    print("✓ test_generate_1d_mesh passed")


def test_constraint_integration():
    """Test integration with constraint system."""
    mesh = Mesh()
    n1 = mesh.add_node(0, 0)
    n2 = mesh.add_node(1, 0)
    
    # Add constraints through mesh.constraints
    mesh.constraints.add(Constraint(n1, 0, 0.0))  # Fix x
    mesh.constraints.add(Constraint(n1, 1, 0.0))  # Fix y
    mesh.constraints.add(Constraint(n1, 2, 0.0))  # Fix rotation
    
    assert len(mesh.constraints.constraints) == 3
    print("✓ test_constraint_integration passed")


def test_point_load_integration():
    """Test integration with point loads."""
    mesh = Mesh()
    mat = Material(1, 210e9, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)
    
    n1 = mesh.add_node(0, 0)
    n2 = mesh.add_node(1, 0)
    mesh.add_element(n1, n2, mat, sec)
    
    # Add constraints
    mesh.constraints.add(Constraint(n1, 0, 0.0))
    mesh.constraints.add(Constraint(n1, 1, 0.0))
    mesh.constraints.add(Constraint(n1, 2, 0.0))
    
    # Add point load
    load = PointLoad(-1000, 1)  # -1000 N in y direction
    load.node = n2
    mesh.point_loads.append(load)
    
    # Run analysis
    analysis = EulerBernoulliAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    
    assert displacements is not None
    assert len(displacements) == 6  # 2 nodes * 3 DOF
    # Verify boundary conditions are satisfied
    assert abs(displacements[0]) < 1e-9  # x displacement at n1
    assert abs(displacements[1]) < 1e-9  # y displacement at n1
    assert abs(displacements[2]) < 1e-9  # rotation at n1
    # Free end should have displacement
    assert abs(displacements[4]) > 1e-6  # y displacement at n2
    
    print("✓ test_point_load_integration passed")


def test_distributed_load_integration():
    """Test integration with distributed loads."""
    mesh = Mesh()
    mat = Material(1, 210e9, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)
    
    n1 = mesh.add_node(0, 0)
    n2 = mesh.add_node(2, 0)
    el = mesh.add_element(n1, n2, mat, sec)
    
    # Add constraints
    mesh.constraints.add(Constraint(n1, 0, 0.0))
    mesh.constraints.add(Constraint(n1, 1, 0.0))
    mesh.constraints.add(Constraint(n1, 2, 0.0))
    
    # Add constant distributed load
    dist_load = DistributedLoad(magnitude_start=-1000, direction='y')
    dist_load.element = el
    mesh.distributed_loads.append(dist_load)
    
    # Compute equivalent nodal loads
    fe = el.compute_equivalent_nodal_loads(dist_load)
    assert len(fe) == 6
    
    # Run analysis
    analysis = EulerBernoulliAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    
    assert displacements is not None
    assert abs(displacements[4]) > 1e-6  # Should have significant displacement
    
    print("✓ test_distributed_load_integration passed")


def test_mixed_element_types():
    """Test mesh with mixed element types."""
    mesh = Mesh()
    mat = Material(1, 210e9, 0.3)
    sec1 = RectangularBar(1, 0.05, 0.1)
    sec2 = CircularBar(2, 0.08)
    
    n1 = mesh.add_node(0, 0)
    n2 = mesh.add_node(1, 0)
    n3 = mesh.add_node(2, 0)
    
    el1 = mesh.add_element(n1, n2, mat, sec1, 'euler_bernoulli_2node')
    el2 = mesh.add_element(n2, n3, mat, sec2, 'timoshenko_2node')
    
    assert len(mesh.elements) == 2
    assert el1.__class__.__name__ == 'EulerBernoulliElement2Node'
    assert el2.__class__.__name__ == 'TimoshenkoElement2Node'
    
    print("✓ test_mixed_element_types passed")


def test_get_node_by_id():
    """Test get_node_by_id method."""
    mesh = Mesh()
    n1 = mesh.add_node(0, 0)
    n2 = mesh.add_node(1, 0)
    n3 = mesh.add_node(2, 0)
    
    node = mesh.get_node_by_id(2)
    assert node is not None
    assert node.id == 2
    assert node.x == 1
    assert node.y == 0
    
    # Test non-existent node
    node = mesh.get_node_by_id(999)
    assert node is None
    
    print("✓ test_get_node_by_id passed")


def test_get_element_by_id():
    """Test get_element_by_id method."""
    mesh = Mesh()
    mat = Material(1, 210e9, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)
    
    n1 = mesh.add_node(0, 0)
    n2 = mesh.add_node(1, 0)
    n3 = mesh.add_node(2, 0)
    
    el1 = mesh.add_element(n1, n2, mat, sec)
    el2 = mesh.add_element(n2, n3, mat, sec)
    
    element = mesh.get_element_by_id(2)
    assert element is not None
    assert element.id == 2
    assert element.node_start == n2
    assert element.node_end == n3
    
    # Test non-existent element
    element = mesh.get_element_by_id(999)
    assert element is None
    
    print("✓ test_get_element_by_id passed")


def test_export_mesh():
    """Test mesh export functionality."""
    mesh = Mesh()
    mat = Material(1, 210e9, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)
    
    n1 = mesh.add_node(0, 0)
    n2 = mesh.add_node(1, 0)
    n3 = mesh.add_node(2, 0)
    
    mesh.add_element(n1, n2, mat, sec)
    mesh.add_element(n2, n3, mat, sec)
    
    export_data = mesh.export_mesh()
    
    assert 'nodes' in export_data
    assert 'elements' in export_data
    assert len(export_data['nodes']) == 3
    assert len(export_data['elements']) == 2
    
    # Verify node data format
    node_data = export_data['nodes'][0]
    assert len(node_data) == 3  # (id, x, y)
    
    # Verify element data format
    elem_data = export_data['elements'][0]
    assert len(elem_data) == 3  # (id, node_start_id, node_end_id)
    
    print("✓ test_export_mesh passed")


def test_node_attributes():
    """Test that nodes can store references to loads and springs."""
    mesh = Mesh()
    n1 = mesh.add_node(0, 0)
    
    assert hasattr(n1, 'loads'), "Node missing loads attribute"
    assert hasattr(n1, 'springs'), "Node missing springs attribute"
    assert n1.loads == []
    assert n1.springs == []
    
    # Test adding references
    load = PointLoad(-1000, 1)
    n1.loads.append(load)
    assert len(n1.loads) == 1
    
    spring = Spring(n1, 1e6, 1)
    n1.springs.append(spring)
    assert len(n1.springs) == 1
    
    print("✓ test_node_attributes passed")


def test_linear_spring_support():
    """Test translational spring support contribution and displacement response."""
    mesh = Mesh()
    n1 = mesh.add_node(0, 0)

    k_linear = 2.0e5
    p_axial = 500.0

    mesh.constraints.add(Constraint(n1, 1, 0.0))
    mesh.constraints.add(Constraint(n1, 2, 0.0))

    load = PointLoad(p_axial, 0)
    load.node = n1
    mesh.point_loads.append(load)

    n1.springs.append(Spring(n1, k_linear, 0))

    analysis = EulerBernoulliAnalysis(mesh)
    analysis.assemble()
    assert np.isclose(analysis.K_global[0, 0], k_linear), "Linear spring stiffness not assembled"

    displacements = analysis.solve()
    expected_u = p_axial / k_linear
    assert np.isclose(displacements[0], expected_u), (
        f"Axial displacement mismatch: {displacements[0]} vs {expected_u}"
    )

    print("✓ test_linear_spring_support passed")


def test_torsional_spring_support():
    """Test torsional spring support contribution and rotation response."""
    mesh = Mesh()
    n1 = mesh.add_node(0, 0)

    k_torsion = 4.0e4
    m_applied = 120.0

    mesh.constraints.add(Constraint(n1, 0, 0.0))
    mesh.constraints.add(Constraint(n1, 1, 0.0))

    load = PointLoad(m_applied, 2)
    load.node = n1
    mesh.point_loads.append(load)

    n1.springs.append(Spring(n1, k_torsion, 2))

    analysis = EulerBernoulliAnalysis(mesh)
    analysis.assemble()
    assert np.isclose(analysis.K_global[2, 2], k_torsion), "Torsional spring stiffness not assembled"

    displacements = analysis.solve()
    expected_theta = m_applied / k_torsion
    assert np.isclose(displacements[2], expected_theta), (
        f"Rotation mismatch: {displacements[2]} vs {expected_theta}"
    )

    print("✓ test_torsional_spring_support passed")


def test_element_geometry():
    """Test element geometry calculations for inclined elements."""
    mesh = Mesh()
    mat = Material(1, 210e9, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)
    
    # 3-4-5 right triangle
    n1 = mesh.add_node(0, 0)
    n2 = mesh.add_node(3, 4)
    el = mesh.add_element(n1, n2, mat, sec)
    
    assert abs(el.length - 5.0) < 1e-10
    assert abs(el.c - 0.6) < 1e-10  # cos(theta) = 3/5
    assert abs(el.s - 0.8) < 1e-10  # sin(theta) = 4/5
    
    print("✓ test_element_geometry passed")


def test_structure_results_integration():
    """Test integration with StructureResults post-processing."""
    mesh = Mesh()
    mat = Material(1, 210e9, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)
    
    n1 = mesh.add_node(0, 0)
    n2 = mesh.add_node(1, 0)
    n3 = mesh.add_node(2, 0)
    
    mesh.add_element(n1, n2, mat, sec, 'euler_bernoulli_2node')
    mesh.add_element(n2, n3, mat, sec, 'euler_bernoulli_2node')
    
    # Add constraints
    mesh.constraints.add(Constraint(n1, 0, 0.0))
    mesh.constraints.add(Constraint(n1, 1, 0.0))
    mesh.constraints.add(Constraint(n1, 2, 0.0))
    
    # Add load
    load = PointLoad(-1000, 1)
    load.node = n3
    mesh.point_loads.append(load)
    
    # Analyze
    analysis = EulerBernoulliAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    
    # Create StructureResults
    results = StructureResults(mesh, displacements)
    assert results is not None
    assert len(results.element_results) == 2
    
    # Test force calculations
    for er in results.element_results:
        x = er.length / 2
        M = er.bending_moment(x)
        V = er.shear_force(x)
        N = er.normal_force(x)
        assert M is not None
        assert V is not None
        assert N is not None
    
    print("✓ test_structure_results_integration passed")


def test_timoshenko_structure_results():
    """Test StructureResults with Timoshenko elements."""
    mesh = Mesh()
    mat = Material(1, 210e9, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)
    
    n1 = mesh.add_node(0, 0)
    n2 = mesh.add_node(1, 0)
    
    mesh.add_element(n1, n2, mat, sec, 'timoshenko_2node')
    
    # Add constraints
    mesh.constraints.add(Constraint(n1, 0, 0.0))
    mesh.constraints.add(Constraint(n1, 1, 0.0))
    mesh.constraints.add(Constraint(n1, 2, 0.0))
    
    # Add load
    load = PointLoad(-1000, 1)
    load.node = n2
    mesh.point_loads.append(load)
    
    # Analyze
    analysis = EulerBernoulliAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    
    # Create StructureResults - should work with Timoshenko elements now
    results = StructureResults(mesh, displacements)
    assert results is not None
    assert len(results.element_results) == 1
    
    # Test force calculations
    er = results.element_results[0]
    x = er.length / 2
    M = er.bending_moment(x)
    V = er.shear_force(x)
    N = er.normal_force(x)
    assert M is not None
    assert V is not None
    assert N is not None
    
    print("✓ test_timoshenko_structure_results passed")


def build_mixed_dof_test_model(element_type):
    """Build and solve a small mixed-DOF model for analysis-pipeline regression tests."""
    mesh = Mesh()
    mat = Material(1, 210e9, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)

    for x in (0.0, 1.0, 2.0):
        mesh.add_node(x, 0.0)

    original_elements = [
        (1, 2, "reddy_bickford_2node", 1),
        (2, 3, element_type, 1),
    ]
    original_to_mesh_elements = {}

    for orig_idx, (n1, n2, etype, n_subdiv) in enumerate(original_elements, start=1):
        node_start = mesh.get_node_by_id(n1)
        node_end = mesh.get_node_by_id(n2)

        if n_subdiv == 1:
            el = mesh.add_element(node_start, node_end, mat, sec, element_type=etype)
            original_to_mesh_elements[orig_idx] = [el.id]
        else:
            subdiv_nodes = [node_start]
            for i in range(1, n_subdiv):
                x = node_start.x + (node_end.x - node_start.x) * i / n_subdiv
                y = node_start.y + (node_end.y - node_start.y) * i / n_subdiv
                existing = next((n for n in mesh.nodes if np.isclose(n.x, x) and np.isclose(n.y, y)), None)
                subdiv_nodes.append(existing if existing else mesh.add_node(x, y))
            subdiv_nodes.append(node_end)

            subdiv_ids = []
            for i in range(n_subdiv):
                el = mesh.add_element(
                    subdiv_nodes[i], subdiv_nodes[i + 1], mat, sec, element_type=etype
                )
                subdiv_ids.append(el.id)
            original_to_mesh_elements[orig_idx] = subdiv_ids

    for direction in range(4):
        mesh.constraints.add(Constraint(mesh.get_node_by_id(1), direction, 0.0))

    load = PointLoad(-1000.0, 1)
    load.node = mesh.get_node_by_id(3)
    mesh.point_loads.append(load)

    for mesh_el_id in original_to_mesh_elements[2]:
        dist_load = DistributedLoad(-100.0, None, "y")
        dist_load.element = mesh.get_element_by_id(mesh_el_id)
        mesh.distributed_loads.append(dist_load)

    analysis = EulerBernoulliAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    results = StructureResults(mesh, displacements, analysis.get_reactions(), analysis.dpn)
    return mesh, analysis, displacements, results


def test_app_pipeline_with_euler_bernoulli_3node():
    """App-style analysis flow should handle Euler-Bernoulli 3-node mixed with 4-DOF elements."""
    mesh, analysis, displacements, results = build_mixed_dof_test_model("euler_bernoulli_3node")

    assert analysis.dpn == 4
    assert len(displacements) == 4 * len(mesh.nodes)
    assert len(results.element_results) == len(mesh.elements)
    assert any(getattr(el, "node_center", None) is not None for el in mesh.elements)

    three_node_result = next(
        er for er in results.element_results if getattr(er.element, "node_center", None) is not None
    )
    assert np.isfinite(three_node_result.bending_moment(three_node_result.length / 2))
    assert np.isfinite(three_node_result.shear_force(three_node_result.length / 2))
    assert np.isfinite(three_node_result.normal_force(three_node_result.length / 2))

    print("✓ test_app_pipeline_with_euler_bernoulli_3node passed")


def test_app_pipeline_with_timoshenko_3node():
    """App-style analysis flow should handle Timoshenko 3-node mixed with 4-DOF elements."""
    mesh, analysis, displacements, results = build_mixed_dof_test_model("timoshenko_3node")

    assert analysis.dpn == 4
    assert len(displacements) == 4 * len(mesh.nodes)
    assert len(results.element_results) == len(mesh.elements)
    assert any(getattr(el, "node_center", None) is not None for el in mesh.elements)

    three_node_result = next(
        er for er in results.element_results if getattr(er.element, "node_center", None) is not None
    )
    assert np.isfinite(three_node_result.bending_moment(three_node_result.length / 2))
    assert np.isfinite(three_node_result.shear_force(three_node_result.length / 2))
    assert np.isfinite(three_node_result.normal_force(three_node_result.length / 2))

    print("✓ test_app_pipeline_with_timoshenko_3node passed")


def test_internal_force_recovery_across_element_types():
    """Internal force recovery should vary correctly across all supported beam elements."""
    element_types = [
        "euler_bernoulli_2node",
        "euler_bernoulli_3node",
        "timoshenko_2node",
        "timoshenko_3node",
        "reddy_bickford_2node",
    ]

    E = 210e9
    nu = 0.3
    L = 2.0
    n_elements = 4
    P = -1000.0

    for element_type in element_types:
        mesh = Mesh()
        mat = Material(1, E, nu)
        sec = RectangularBar(1, 0.05, 0.1)
        nodes = mesh.generate_1d_mesh(0, 0, L, 0, n_elements, mat, sec, element_type)

        mesh.constraints.add(Constraint(nodes[0], 0, 0.0))
        mesh.constraints.add(Constraint(nodes[0], 1, 0.0))
        mesh.constraints.add(Constraint(nodes[0], 2, 0.0))
        if element_type in ["reddy_bickford_2node", "mrbt_2node"]:
            mesh.constraints.add(Constraint(nodes[0], 3, 0.0))

        tip_load = PointLoad(P, 1)
        tip_load.node = nodes[-1]
        mesh.point_loads.append(tip_load)

        analysis = EulerBernoulliAnalysis(mesh)
        analysis.assemble()
        displacements = analysis.solve()
        results = StructureResults(mesh, displacements)

        m_start = results.M(0.0)
        m_mid = results.M(L / 2)
        m_end = results.M(L)
        m_start_expected = P * L
        m_mid_expected = P * (L / 2)
        assert abs(m_start - m_start_expected) < 1e-3 * abs(m_start_expected), (
            f"{element_type}: expected M(0)≈{m_start_expected}, got {m_start}"
        )
        assert abs(m_mid - m_mid_expected) < 1e-3 * abs(m_mid_expected), (
            f"{element_type}: expected M(L/2)≈{m_mid_expected}, got {m_mid}"
        )
        assert abs(m_end) < 1e-3 * abs(P * L), (
            f"{element_type}: free-end bending moment should be near zero, got M(L)={m_end}"
        )

        for v in (results.V(0.0), results.V(L / 2), results.V(L)):
            assert abs(abs(v) - abs(P)) < 1e-3 * abs(P), (
                f"{element_type}: expected |V|≈|P|={abs(P)}, got {v}"
            )

        for x in (0.0, L / 2, L):
            n_val = results.N(x)
            assert abs(n_val) < 1e-6, f"{element_type}: expected N≈0 for pure bending, got {n_val} at x={x}"

    print("✓ test_internal_force_recovery_across_element_types passed")


def test_streamlit_app_run_analysis_with_mixed_dofs():
    """The Streamlit Run Analysis flow should complete without UI errors for mixed DOF meshes."""
    mat = Material(1, 210e9, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)

    for element_type in ("euler_bernoulli_3node", "timoshenko_3node"):
        at = AppTest.from_file("app.py")
        at.session_state["nodes"] = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        at.session_state["properties"] = [{
            "name": "Property_1",
            "material": mat,
            "mat_input_mode": "Calculate G (from E and ν)",
            "section": sec,
            "section_type": "rectangular_bar",
            "section_kwargs": {"width": 0.05, "height": 0.1}
        }]
        at.session_state["elements"] = [
            (1, 2, "reddy_bickford_2node", "Property_1", 1),
            (2, 3, element_type, "Property_1", 1),
        ]
        at.session_state["constraints"] = [(1, 0, 0.0), (1, 1, 0.0), (1, 2, 0.0), (1, 3, 0.0)]
        at.session_state["point_loads"] = [(3, 1, -1000.0)]
        at.session_state["distributed_loads"] = []

        at.run(timeout=60)
        run_analysis_button = next(btn for btn in at.button if btn.label == "Run Analysis")
        run_analysis_button.click()
        at.run(timeout=60)

        errors = [err.value for err in at.error]
        assert not any(msg.startswith("❌ Analysis failed") for msg in errors), (
            f"Run Analysis failed for {element_type}: {errors}"
        )
        assert "displacements" in at.session_state

    print("✓ test_streamlit_app_run_analysis_with_mixed_dofs passed")


def test_streamlit_app_numerical_integration_option_applied_to_elements():
    """Run Analysis should pass the selected numerical integration mode to 2-node elements."""
    mat = Material(1, 210e9, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)

    at = AppTest.from_file("app.py")
    at.session_state["nodes"] = [(0.0, 0.0), (2.0, 0.0)]
    at.session_state["properties"] = [{
        "name": "Property_1",
        "material": mat,
        "mat_input_mode": "Calculate G (from E and ν)",
        "section": sec,
        "section_type": "rectangular_bar",
        "section_kwargs": {"width": 0.05, "height": 0.1}
    }]
    at.session_state["elements"] = [(1, 2, "euler_bernoulli_2node", "Property_1", 1)]
    at.session_state["constraints"] = [(1, 0, 0.0), (1, 1, 0.0), (1, 2, 0.0)]
    at.session_state["point_loads"] = [(2, 1, -1000.0)]
    at.session_state["distributed_loads"] = []
    at.session_state["stiffness_integration_mode"] = "numerical"

    at.run(timeout=60)
    run_analysis_button = next(btn for btn in at.button if btn.label == "Run Analysis")
    run_analysis_button.click()
    at.run(timeout=60)

    errors = [err.value for err in at.error]
    assert not any(msg.startswith("❌ Analysis failed") for msg in errors), (
        f"Run Analysis failed with numerical integration mode: {errors}"
    )
    assert at.session_state["mesh"].elements[0].stiffness_integration == "numerical"
    print("✓ test_streamlit_app_numerical_integration_option_applied_to_elements passed")


def test_streamlit_app_applies_linear_and_torsional_springs():
    """Run Analysis should transfer spring definitions from session state to mesh nodes."""
    mat = Material(1, 210e9, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)

    at = AppTest.from_file("app.py")
    at.session_state["nodes"] = [(0.0, 0.0), (1.0, 0.0)]
    at.session_state["properties"] = [{
        "name": "Property_1",
        "material": mat,
        "mat_input_mode": "Calculate G (from E and ν)",
        "section": sec,
        "section_type": "rectangular_bar",
        "section_kwargs": {"width": 0.05, "height": 0.1}
    }]
    at.session_state["elements"] = [(1, 2, "euler_bernoulli_2node", "Property_1", 1)]
    at.session_state["constraints"] = [(1, 0, 0.0), (1, 1, 0.0), (1, 2, 0.0)]
    at.session_state["springs"] = [(2, 1, 1.5e5), (2, 2, 8.0e4)]
    at.session_state["point_loads"] = [(2, 1, -1000.0)]
    at.session_state["distributed_loads"] = []

    at.run(timeout=60)
    run_analysis_button = next(btn for btn in at.button if btn.label == "Run Analysis")
    run_analysis_button.click()
    at.run(timeout=60)

    errors = [err.value for err in at.error]
    assert not any(msg.startswith("❌ Analysis failed") for msg in errors), (
        f"Run Analysis failed with springs: {errors}"
    )

    mesh_node = at.session_state["mesh"].get_node_by_id(2)
    assert len(mesh_node.springs) == 2
    assert {(sp.direction, sp.stiffness) for sp in mesh_node.springs} == {(1, 1.5e5), (2, 8.0e4)}
    print("✓ test_streamlit_app_applies_linear_and_torsional_springs passed")


def test_structural_behavior_modes_supported_across_element_types():
    """Truss, beam, and frame behavior modes should solve across all element formulations."""
    mat = Material(1, 210e9, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)
    element_types = (
        "euler_bernoulli_2node",
        "euler_bernoulli_3node",
        "timoshenko_2node",
        "timoshenko_3node",
        "reddy_bickford_2node",
    )

    for element_type in element_types:
        for behavior in ("truss", "beam", "frame"):
            mesh = Mesh()
            n1 = mesh.add_node(0.0, 0.0)
            n2 = mesh.add_node(1.0, 0.0)
            mesh.add_element(n1, n2, mat, sec, element_type=element_type)

            if behavior == "truss":
                mesh.constraints.add(Constraint(n1, 0, 0.0))
                p = PointLoad(1000.0, 0)
            else:
                mesh.constraints.add(Constraint(n1, 0, 0.0))
                mesh.constraints.add(Constraint(n1, 1, 0.0))
                mesh.constraints.add(Constraint(n1, 2, 0.0))
                if element_type in ["reddy_bickford_2node", "mrbt_2node"]:
                    mesh.constraints.add(Constraint(n1, 3, 0.0))
                p = PointLoad(-1000.0, 1)
            p.node = n2
            mesh.point_loads.append(p)

            analysis = EulerBernoulliAnalysis(mesh, structural_behavior=behavior)
            analysis.assemble()
            displacements = analysis.solve()

            assert np.all(np.isfinite(displacements)), (
                f"Non-finite displacement for element={element_type}, behavior={behavior}"
            )

    print("✓ test_structural_behavior_modes_supported_across_element_types passed")


def test_streamlit_app_structural_behavior_option_applied():
    """Run Analysis should apply selected structural behavior mode."""
    mat = Material(1, 210e9, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)

    for behavior in ("truss", "beam", "frame"):
        at = AppTest.from_file("app.py")
        at.session_state["nodes"] = [(0.0, 0.0), (2.0, 0.0)]
        at.session_state["properties"] = [{
            "name": "Property_1",
            "material": mat,
            "mat_input_mode": "Calculate G (from E and ν)",
            "section": sec,
            "section_type": "rectangular_bar",
            "section_kwargs": {"width": 0.05, "height": 0.1}
        }]
        at.session_state["elements"] = [(1, 2, "euler_bernoulli_2node", "Property_1", 1)]
        at.session_state["constraints"] = [(1, 0, 0.0), (1, 1, 0.0), (1, 2, 0.0)]
        if behavior == "truss":
            at.session_state["point_loads"] = [(2, 0, 1000.0)]
        elif behavior == "beam":
            at.session_state["point_loads"] = [(2, 1, -1000.0)]
        else:
            at.session_state["point_loads"] = [(2, 0, 1000.0), (2, 1, -1000.0)]
        at.session_state["distributed_loads"] = []
        at.session_state["structural_behavior_mode"] = behavior

        at.run(timeout=60)
        run_analysis_button = next(btn for btn in at.button if btn.label == "Run Analysis")
        run_analysis_button.click()
        at.run(timeout=60)

        errors = [err.value for err in at.error]
        assert not any(msg.startswith("❌ Analysis failed") for msg in errors), (
            f"Run Analysis failed for behavior={behavior}: {errors}"
        )

        disp = at.session_state["displacements"]
        dpn = at.session_state["dpn"]
        base = dpn * (2 - 1)
        u, v, theta = disp[base], disp[base + 1], disp[base + 2]
        if behavior == "truss":
            assert abs(u) > 1e-16
            assert abs(v) < 1e-14
            assert abs(theta) < 1e-14
        elif behavior == "beam":
            assert abs(u) < 1e-14
            assert abs(v) > 1e-16
        else:
            assert abs(u) > 1e-16
            assert abs(v) > 1e-16

    print("✓ test_streamlit_app_structural_behavior_option_applied passed")


def test_streamlit_app_structural_behavior_filters_inputs_and_outputs():
    """Structural behavior should filter relevant input options and output views."""
    mat = Material(1, 210e9, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)
    expected = {
        "truss": {
            "constraint_dofs": ["X displacement", "Y displacement"],
            "point_dofs": ["X force", "Y force"],
            "diagram_options": ["Normal Force"],
            "disp_cols": {"Node", "X", "Y", "U", "V"},
            "forbidden_disp_cols": {"θ", "dv/dx"},
        },
        "beam": {
            "constraint_dofs": ["Y displacement", "Rotation"],
            "point_dofs": ["Y force", "Moment"],
            "diagram_options": ["Moment", "Shear"],
            "disp_cols": {"Node", "X", "Y", "V", "θ"},
            "forbidden_disp_cols": {"U", "dv/dx"},
        },
        "frame": {
            "constraint_dofs": ["X displacement", "Y displacement", "Rotation"],
            "point_dofs": ["X force", "Y force", "Moment"],
            "diagram_options": ["Moment", "Shear", "Normal Force"],
            "disp_cols": {"Node", "X", "Y", "U", "V", "θ"},
            "forbidden_disp_cols": {"dv/dx"},
        },
    }

    for behavior in ("truss", "beam", "frame"):
        at = AppTest.from_file("app.py")
        at.session_state["nodes"] = [(0.0, 0.0), (2.0, 0.0)]
        at.session_state["properties"] = [{
            "name": "Property_1",
            "material": mat,
            "mat_input_mode": "Calculate G (from E and ν)",
            "section": sec,
            "section_type": "rectangular_bar",
            "section_kwargs": {"width": 0.05, "height": 0.1}
        }]
        at.session_state["elements"] = [(1, 2, "euler_bernoulli_2node", "Property_1", 1)]
        if behavior == "truss":
            at.session_state["constraints"] = [(1, 0, 0.0), (1, 1, 0.0)]
            at.session_state["point_loads"] = [(2, 0, 1000.0)]
        elif behavior == "beam":
            at.session_state["constraints"] = [(1, 1, 0.0), (1, 2, 0.0)]
            at.session_state["point_loads"] = [(2, 1, -1000.0)]
        else:
            at.session_state["constraints"] = [(1, 0, 0.0), (1, 1, 0.0), (1, 2, 0.0)]
            at.session_state["point_loads"] = [(2, 0, 1000.0), (2, 1, -1000.0)]
        at.session_state["distributed_loads"] = []
        at.session_state["structural_behavior_mode"] = behavior

        at.run(timeout=60)

        dof_box = next(sb for sb in at.selectbox if sb.label == "DOF")
        point_box = next(sb for sb in at.selectbox if sb.label == "Direction")

        assert dof_box.options == expected[behavior]["constraint_dofs"]
        assert point_box.options == expected[behavior]["point_dofs"]

        run_analysis_button = next(btn for btn in at.button if btn.label == "Run Analysis")
        run_analysis_button.click()
        at.run(timeout=60)
        errors = [err.value for err in at.error]
        assert not any(msg.startswith("❌ Analysis failed") for msg in errors), (
            f"Run Analysis failed for behavior={behavior}: {errors}"
        )

        diagram_box = next(sb for sb in at.selectbox if sb.label == "Select diagram type")
        assert diagram_box.options == expected[behavior]["diagram_options"]

        disp_df = at.dataframe[0].value
        disp_cols = set(disp_df.columns)
        assert disp_cols == expected[behavior]["disp_cols"], (
            f"{behavior}: unexpected displacement columns {disp_cols}"
        )
        for forbidden_col in expected[behavior]["forbidden_disp_cols"]:
            assert forbidden_col not in disp_cols, f"{behavior}: {forbidden_col} should not be shown"

    print("✓ test_streamlit_app_structural_behavior_filters_inputs_and_outputs passed")


def test_streamlit_app_force_diagram_resolution_slider_removed():
    """Force diagram tab should not expose manual points-per-element slider."""
    mat = Material(1, 210e9, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)

    at = AppTest.from_file("app.py")
    at.session_state["nodes"] = [(0.0, 0.0), (2.0, 0.0)]
    at.session_state["properties"] = [{
        "name": "Property_1",
        "material": mat,
        "mat_input_mode": "Calculate G (from E and ν)",
        "section": sec,
        "section_type": "rectangular_bar",
        "section_kwargs": {"width": 0.05, "height": 0.1}
    }]
    at.session_state["elements"] = [(1, 2, "euler_bernoulli_2node", "Property_1", 1)]
    at.session_state["constraints"] = [(1, 0, 0.0), (1, 1, 0.0), (1, 2, 0.0)]
    at.session_state["point_loads"] = [(2, 1, -1000.0)]
    at.session_state["distributed_loads"] = []

    at.run(timeout=60)
    run_analysis_button = next(btn for btn in at.button if btn.label == "Run Analysis")
    run_analysis_button.click()
    at.run(timeout=60)

    slider_labels = [sl.label for sl in at.slider]
    assert "Diagram resolution (points per element)" not in slider_labels
    print("✓ test_streamlit_app_force_diagram_resolution_slider_removed passed")


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("Running Mesh Integration Tests")
    print("="*60 + "\n")
    
    test_mesh_creation()
    test_add_nodes()
    test_add_elements()
    test_euler_bernoulli_numerical_stiffness_matches_analytical()
    test_element_attributes()
    test_generate_1d_mesh()
    test_constraint_integration()
    test_point_load_integration()
    test_distributed_load_integration()
    test_mixed_element_types()
    test_get_node_by_id()
    test_get_element_by_id()
    test_export_mesh()
    test_node_attributes()
    test_linear_spring_support()
    test_torsional_spring_support()
    test_element_geometry()
    test_structure_results_integration()
    test_timoshenko_structure_results()
    test_app_pipeline_with_euler_bernoulli_3node()
    test_app_pipeline_with_timoshenko_3node()
    test_internal_force_recovery_across_element_types()
    test_streamlit_app_run_analysis_with_mixed_dofs()
    test_streamlit_app_numerical_integration_option_applied_to_elements()
    test_streamlit_app_applies_linear_and_torsional_springs()
    test_structural_behavior_modes_supported_across_element_types()
    test_streamlit_app_structural_behavior_option_applied()
    test_streamlit_app_structural_behavior_filters_inputs_and_outputs()
    test_streamlit_app_force_diagram_resolution_slider_removed()
    
    print("\n" + "="*60)
    print("✅ All Mesh Integration Tests Passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
