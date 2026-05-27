import numpy as np

from fem.section import RectangularBar
from post_processing.plotter import plot_shear_stress_distribution


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


def run_all_tests():
    test_plot_shear_stress_distribution_rectangular_profile()
    test_plot_shear_stress_distribution_zero_shear()
    print("\n✅ All shear stress distribution plot tests passed!")


if __name__ == "__main__":
    run_all_tests()
