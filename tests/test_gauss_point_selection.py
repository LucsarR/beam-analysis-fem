import pytest
import numpy as np
from fem.node import Node
from fem.material import Material
from fem.section import RectangularBar
from fem.element import (
    EulerBernoulliElement2Node,
    EulerBernoulliElement3Node,
    TimoshenkoElement2Node,
    TimoshenkoElement3Node,
    ReddyBickfordElement2Node,
    MRBTElement2Node
)
from fem.mesh import Mesh

def test_euler_bernoulli_2node_gauss_points():
    node1 = Node(1, 0.0, 0.0)
    node2 = Node(2, 2.0, 0.0)
    mat = Material(1, 2.1e11, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)
    
    # 1. Default (n_gauss=None which defaults to 3)
    el_default = EulerBernoulliElement2Node(1, node1, node2, mat, sec, stiffness_integration="numerical")
    assert el_default.n_gauss is None
    
    # 2. Configured n_gauss=3
    el_3 = EulerBernoulliElement2Node(2, node1, node2, mat, sec, stiffness_integration="numerical", n_gauss=3)
    assert el_3.n_gauss == 3
    
    # 3. Configured n_gauss=4
    el_4 = EulerBernoulliElement2Node(3, node1, node2, mat, sec, stiffness_integration="numerical", n_gauss=4)
    assert el_4.n_gauss == 4

    # Stiffness matrices
    k_default = el_default.stiffness_matrix()
    k_3 = el_3.stiffness_matrix()
    k_4 = el_4.stiffness_matrix()
    k_analytical = EulerBernoulliElement2Node(4, node1, node2, mat, sec, stiffness_integration="analytical").stiffness_matrix()
    
    # Numerical with 3 points should match default (which is 3 points)
    assert np.allclose(k_default, k_3)
    
    # 3-point Gauss-Legendre is exact for Hermite cubic bending terms, so it should match analytical stiffness
    assert np.allclose(k_3, k_analytical, atol=1e-5)
    
    # 4-point should also be exact and match analytical stiffness
    assert np.allclose(k_4, k_analytical, atol=1e-5)

def test_euler_bernoulli_3node_gauss_points():
    node1 = Node(1, 0.0, 0.0)
    node2 = Node(2, 2.0, 0.0)
    node_c = Node(3, 1.0, 0.0)
    mat = Material(1, 2.1e11, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)
    
    # Default (5-point)
    el_default = EulerBernoulliElement3Node(1, node1, node2, mat, sec, node_center=node_c)
    assert el_default.n_gauss is None
    
    # Configured n_gauss=5
    el_5 = EulerBernoulliElement3Node(2, node1, node2, mat, sec, node_center=node_c, n_gauss=5)
    
    # Configured n_gauss=3
    el_3 = EulerBernoulliElement3Node(3, node1, node2, mat, sec, node_center=node_c, n_gauss=3)
    
    k_default = el_default.stiffness_matrix()
    k_5 = el_5.stiffness_matrix()
    k_3 = el_3.stiffness_matrix()
    
    assert k_default.shape == (9, 9)
    assert k_5.shape == (9, 9)
    assert k_3.shape == (9, 9)
    assert np.allclose(k_default, k_5)
    assert not np.allclose(k_5, k_3)  # 3-point is not exact for 3-node EB bending (quintic shapes)

def test_timoshenko_2node_gauss_points():
    node1 = Node(1, 0.0, 0.0)
    node2 = Node(2, 2.0, 0.0)
    mat = Material(1, 2.1e11, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)
    
    # Default (2-point axial/bending, 1-point shear)
    el_default = TimoshenkoElement2Node(1, node1, node2, mat, sec, stiffness_integration="numerical")
    assert el_default.n_gauss is None
    
    # Configured n_gauss=2
    el_2 = TimoshenkoElement2Node(2, node1, node2, mat, sec, stiffness_integration="numerical", n_gauss=2)
    
    # Configured n_gauss=3 (3-point axial/bending, 2-point shear)
    el_3 = TimoshenkoElement2Node(3, node1, node2, mat, sec, stiffness_integration="numerical", n_gauss=3)
    
    k_default = el_default.stiffness_matrix()
    k_2 = el_2.stiffness_matrix()
    k_3 = el_3.stiffness_matrix()
    
    assert k_default.shape == (6, 6)
    assert k_2.shape == (6, 6)
    assert k_3.shape == (6, 6)
    
    assert np.allclose(k_default, k_2)
    # n_gauss=3 should yield a different stiffness matrix compared to n_gauss=2
    assert not np.allclose(k_2, k_3)

def test_timoshenko_3node_gauss_points():
    node1 = Node(1, 0.0, 0.0)
    node2 = Node(2, 2.0, 0.0)
    node_c = Node(3, 1.0, 0.0)
    mat = Material(1, 2.1e11, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)
    
    # Default (3-point bending, 2-point shear)
    el_default = TimoshenkoElement3Node(1, node1, node2, mat, sec, node_center=node_c)
    assert el_default.n_gauss is None
    
    # Configured n_gauss=3
    el_3 = TimoshenkoElement3Node(2, node1, node2, mat, sec, node_center=node_c, n_gauss=3)
    
    # Configured n_gauss=4
    el_4 = TimoshenkoElement3Node(3, node1, node2, mat, sec, node_center=node_c, n_gauss=4)
    
    k_default = el_default.stiffness_matrix()
    k_3 = el_3.stiffness_matrix()
    k_4 = el_4.stiffness_matrix()
    
    assert k_default.shape == (9, 9)
    assert k_3.shape == (9, 9)
    assert k_4.shape == (9, 9)
    assert np.allclose(k_default, k_3)

def test_reddy_bickford_mrbt_interfaces():
    node1 = Node(1, 0.0, 0.0)
    node2 = Node(2, 2.0, 0.0)
    mat = Material(1, 2.1e11, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)
    
    # Verify signatures accept n_gauss and don't error
    el_rb = ReddyBickfordElement2Node(1, node1, node2, mat, sec, n_gauss=4)
    el_mrbt = MRBTElement2Node(2, node1, node2, mat, sec, n_gauss=4)
    
    assert el_rb.n_gauss == 4
    assert el_mrbt.n_gauss == 4
    
    k_rb = el_rb.stiffness_matrix()
    k_mrbt = el_mrbt.stiffness_matrix()
    assert k_rb.shape == (8, 8)
    assert k_mrbt.shape == (8, 8)

def test_mesh_integration_gauss_propagation():
    mesh = Mesh()
    mat = Material(1, 2.1e11, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)
    
    # Test Mesh.add_element
    n1 = mesh.add_node(0.0, 0.0)
    n2 = mesh.add_node(2.0, 0.0)
    
    el = mesh.add_element(n1, n2, mat, sec, element_type="euler_bernoulli_2node", stiffness_integration="numerical", n_gauss=5)
    assert el.n_gauss == 5
    
    # Test Mesh.generate_1d_mesh
    nodes = mesh.generate_1d_mesh(2.0, 0.0, 4.0, 0.0, n_elements=2, material=mat, section=sec, element_type="euler_bernoulli_2node", stiffness_integration="numerical", n_gauss=6)
    
    # The generated elements should have ID 2 and 3, check their n_gauss
    assert mesh.elements[1].n_gauss == 6
    assert mesh.elements[2].n_gauss == 6
