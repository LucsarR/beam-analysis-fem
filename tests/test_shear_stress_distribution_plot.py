import numpy as np

from fem.section import RectangularBar
from fem.element import EulerBernoulliElement2Node, ReddyBickfordElement2Node
from fem.material import Material
from post_processing.forces import ElementResults
from post_processing.plotter import (
    plot_shear_stress_distribution,
    plot_reddy_shear_stress_distribution,
    plot_shear_stress_comparison,
    plot_shear_stress_side_view
)


class _Element:
    def __init__(self, section):
        self.section = section


class _ElementResult:
    def __init__(self, section, shear_value):
        self.element = _Element(section)
        self._shear_value = shear_value

    def shear_force(self, x):
        return self._shear_value


def _gradient_trace(fig):
    for trace in fig.data:
        if getattr(trace, "mode", None) == "markers":
            return trace
    raise AssertionError("Could not find shear contour markers trace")


def test_plot_shear_stress_distribution_rectangular_profile():
    sec = RectangularBar(1, width=0.2, height=0.4)
    er = _ElementResult(sec, shear_value=1000.0)
    fig = plot_shear_stress_distribution(er, x=0.0, n_points=61)
    trace = _gradient_trace(fig)

    y = np.asarray(trace.y, dtype=float)
    tau = np.asarray(trace.marker.color, dtype=float)
    h = sec.height

    center = np.mean(np.abs(tau[np.abs(y) < 0.05 * h]))
    edge = np.mean(np.abs(tau[np.abs(y) > 0.45 * h]))

    assert center > edge
    assert "Shear Stress Contour" in fig.layout.title.text
    print("OK test_plot_shear_stress_distribution_rectangular_profile passed")


def test_plot_shear_stress_distribution_zero_shear():
    sec = RectangularBar(1, width=0.2, height=0.4)
    er = _ElementResult(sec, shear_value=0.0)
    fig = plot_shear_stress_distribution(er, x=0.0, n_points=41)
    trace = _gradient_trace(fig)
    tau = np.asarray(trace.marker.color, dtype=float)
    assert np.allclose(tau, 0.0)
    print("OK test_plot_shear_stress_distribution_zero_shear passed")


class MockNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def test_jourawski_shear_stress_extraction():
    material = Material("Steel", E=210e9, nu=0.3)
    section = RectangularBar(1, width=0.2, height=0.4)
    node_start = MockNode(0.0, 0.0)
    node_end = MockNode(2.0, 0.0)
    
    element = EulerBernoulliElement2Node(1, node_start, node_end, material, section)
    displacements = np.array([0.0, 0.0, 0.0, 0.0, -0.01, 0.0])
    
    er = ElementResults(element, displacements)
    V = er.shear_force(0.0)
    
    tau_neutral = er.jourawski_shear_stress(0.0, 0.0)
    # Analytical maximum shear stress for a rectangular section: 1.5 * V / A
    tau_analytical = 1.5 * V / section.area
    
    assert np.isclose(tau_neutral, tau_analytical, rtol=0.03)
    print("OK test_jourawski_shear_stress_extraction passed")


def test_shear_stress_plots_with_query_y():
    material = Material("Steel", E=210e9, nu=0.3)
    section = RectangularBar(1, width=0.2, height=0.4)
    node_start = MockNode(0.0, 0.0)
    node_end = MockNode(2.0, 0.0)
    
    element = ReddyBickfordElement2Node(1, node_start, node_end, material, section)
    displacements = np.zeros(8)
    displacements[5] = -0.01  # cause bending/shear
    
    er = ElementResults(element, displacements)
    
    # Test plot_shear_stress_distribution with query_y
    fig_j = plot_shear_stress_distribution(er, x=0.5, n_points=50, query_y=0.1)
    assert fig_j is not None
    
    # Test plot_reddy_shear_stress_distribution with query_y
    fig_r = plot_reddy_shear_stress_distribution(er, x=0.5, n_points=50, query_y=0.1)
    assert fig_r is not None
    
    # Test plot_shear_stress_comparison with query_y
    fig_comp = plot_shear_stress_comparison(er, x=0.5, n_points=50, query_y=0.1)
    assert fig_comp is not None
    
    print("OK test_shear_stress_plots_with_query_y passed")


def test_plot_shear_stress_side_view():
    material = Material("Steel", E=210e9, nu=0.3)
    section = RectangularBar(1, width=0.2, height=0.4)
    node_start = MockNode(0.0, 0.0)
    node_end = MockNode(2.0, 0.0)
    
    element = ReddyBickfordElement2Node(1, node_start, node_end, material, section)
    displacements = np.zeros(8)
    displacements[5] = -0.01  # cause bending/shear
    
    er = ElementResults(element, displacements)
    
    # Test plot_shear_stress_side_view
    fig = plot_shear_stress_side_view(er, n_x=20, n_y=20, display_x=0.5, query_y=0.1)
    assert fig is not None
    print("OK test_plot_shear_stress_side_view passed")


def test_plot_shear_stress_side_view_subelements():
    material = Material("Steel", E=210e9, nu=0.3)
    section = RectangularBar(1, width=0.2, height=0.4)
    
    # Subelement 1: 0.0 to 1.0
    n1 = MockNode(0.0, 0.0)
    n2 = MockNode(1.0, 0.0)
    el1 = ReddyBickfordElement2Node(1, n1, n2, material, section)
    displacements1 = np.zeros(8)
    displacements1[5] = -0.005
    er1 = ElementResults(el1, displacements1)
    
    # Subelement 2: 1.0 to 2.0
    n3 = MockNode(2.0, 0.0)
    el2 = ReddyBickfordElement2Node(2, n2, n3, material, section)
    displacements2 = np.zeros(8)
    displacements2[5] = -0.01
    er2 = ElementResults(el2, displacements2)
    
    # Test plot_shear_stress_side_view with list of subelements
    fig = plot_shear_stress_side_view([er1, er2], n_x=20, n_y=20, display_x=1.2, query_y=0.05)
    assert fig is not None
    
    # Check that x range matches total length (2.0)
    contour = fig.data[0]
    assert np.isclose(contour.x[0], 0.0)
    assert np.isclose(contour.x[-1], 2.0)
    print("OK test_plot_shear_stress_side_view_subelements passed")


def run_all_tests():
    test_plot_shear_stress_distribution_rectangular_profile()
    test_plot_shear_stress_distribution_zero_shear()
    test_jourawski_shear_stress_extraction()
    test_shear_stress_plots_with_query_y()
    test_plot_shear_stress_side_view()
    test_plot_shear_stress_side_view_subelements()
    print("\nAll shear stress distribution plot tests passed!")


if __name__ == "__main__":
    run_all_tests()
