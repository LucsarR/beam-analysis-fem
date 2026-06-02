import numpy as np

from fem.section import RectangularBar
from fem.element import EulerBernoulliElement2Node, ReddyBickfordElement2Node
from fem.material import Material
from post_processing.forces import ElementResults
from post_processing.plotter import (
    plot_shear_stress_distribution,
    plot_reddy_shear_stress_distribution,
    plot_shear_stress_comparison
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
    print("✓ test_plot_shear_stress_distribution_rectangular_profile passed")


def test_plot_shear_stress_distribution_zero_shear():
    sec = RectangularBar(1, width=0.2, height=0.4)
    er = _ElementResult(sec, shear_value=0.0)
    fig = plot_shear_stress_distribution(er, x=0.0, n_points=41)
    trace = _gradient_trace(fig)
    tau = np.asarray(trace.marker.color, dtype=float)
    assert np.allclose(tau, 0.0)
    print("✓ test_plot_shear_stress_distribution_zero_shear passed")


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
    print("✓ test_jourawski_shear_stress_extraction passed")


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
    
    print("✓ test_shear_stress_plots_with_query_y passed")


def run_all_tests():
    test_plot_shear_stress_distribution_rectangular_profile()
    test_plot_shear_stress_distribution_zero_shear()
    test_jourawski_shear_stress_extraction()
    test_shear_stress_plots_with_query_y()
    print("\n✅ All shear stress distribution plot tests passed!")


if __name__ == "__main__":
    run_all_tests()
