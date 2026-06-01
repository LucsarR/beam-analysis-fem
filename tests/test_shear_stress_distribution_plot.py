import numpy as np

from fem.section import RectangularBar
from post_processing.plotter import plot_shear_stress_distribution


class _Element:
    def __init__(self, section):
        self.section = section


class _Material:
    def __init__(self, G):
        self.G = G


class _ReddyElement(_Element):
    def __init__(self, section, G, theta_minus_slope):
        super().__init__(section)
        self.material = _Material(G)
        self._theta_minus_slope = theta_minus_slope

    def theta_and_slope(self, x, displacements):
        del x, displacements
        return self._theta_minus_slope, 0.0


class _ElementResult:
    def __init__(self, section, shear_value):
        self.element = _Element(section)
        self._shear_value = shear_value

    def shear_force(self, x):
        return self._shear_value


class _ReddyElementResult:
    def __init__(self, section, G=80e9, theta_minus_slope=1e-3):
        self.element = _ReddyElement(section, G, theta_minus_slope)
        self.length = 1.0
        self.displacements = np.zeros(8)

    def shear_force(self, x):
        del x
        return 0.0


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


def test_reddy_bickford_shear_is_zero_on_rectangular_boundaries():
    sec = RectangularBar(1, width=0.2, height=0.4)
    er = _ReddyElementResult(sec, theta_minus_slope=2e-3)
    fig = plot_shear_stress_distribution(er, x=0.0, n_points=81, method="reddy_bickford")
    trace = _gradient_trace(fig)

    y = np.asarray(trace.y, dtype=float)
    tau = np.asarray(trace.marker.color, dtype=float)
    h = sec.height
    edge_mask = np.isclose(np.abs(y), h / 2, atol=1e-12)
    assert np.any(edge_mask), "No boundary points were sampled"
    assert np.allclose(tau[edge_mask], 0.0, atol=1e-6)
    print("✓ test_reddy_bickford_shear_is_zero_on_rectangular_boundaries passed")


def test_compare_shear_methods_returns_both_curves():
    sec = RectangularBar(1, width=0.2, height=0.4)
    er = _ReddyElementResult(sec, theta_minus_slope=2e-3)
    fig = plot_shear_stress_distribution(er, x=0.0, n_points=61, method="compare")
    assert len(fig.data) == 2
    names = {trace.name for trace in fig.data}
    assert {"Jourawski", "Reddy-Bickford"} <= names
    print("✓ test_compare_shear_methods_returns_both_curves passed")


def run_all_tests():
    test_plot_shear_stress_distribution_rectangular_profile()
    test_plot_shear_stress_distribution_zero_shear()
    test_reddy_bickford_shear_is_zero_on_rectangular_boundaries()
    test_compare_shear_methods_returns_both_curves()
    print("\n✅ All shear stress distribution plot tests passed!")


if __name__ == "__main__":
    run_all_tests()
