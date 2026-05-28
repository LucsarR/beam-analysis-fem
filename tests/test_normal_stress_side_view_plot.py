import numpy as np

from fem.section import RectangularBar
from post_processing.plotter import plot_normal_stress_side_view


class _Element:
    def __init__(self, section):
        self.section = section


class _ElementResult:
    def __init__(self, section, length=1.0, normal=100.0, moment=10.0):
        self.element = _Element(section)
        self.length = length
        self._normal = normal
        self._moment = moment

    def normal_force(self, x):
        return self._normal

    def bending_moment(self, x):
        return self._moment


def _trace_by_name(fig, name):
    for trace in fig.data:
        if getattr(trace, "name", None) == name:
            return trace
    raise AssertionError(f"Could not find trace with name '{name}'")


def test_side_view_uses_display_axis_values():
    sec = RectangularBar(1, width=0.2, height=0.4)
    er = _ElementResult(sec, length=1.0)
    fig = plot_normal_stress_side_view(er, x=0.25, display_x=1.5, display_length=3.0)

    beam_trace = _trace_by_name(fig, "Beam Element")
    cut_trace = _trace_by_name(fig, "Cut Position")

    assert np.isclose(float(beam_trace.x[-1]), 3.0)
    assert np.allclose(np.asarray(cut_trace.x, dtype=float), [1.5, 1.5])
    assert "x=1.50" in fig.layout.title.text
    print("✓ test_side_view_uses_display_axis_values passed")


def test_side_view_clips_display_position_to_axis_limits():
    sec = RectangularBar(1, width=0.2, height=0.4)
    er = _ElementResult(sec, length=1.0)
    fig = plot_normal_stress_side_view(er, x=0.25, display_x=4.0, display_length=3.0)
    cut_trace = _trace_by_name(fig, "Cut Position")

    assert np.allclose(np.asarray(cut_trace.x, dtype=float), [3.0, 3.0])
    print("✓ test_side_view_clips_display_position_to_axis_limits passed")


def run_all_tests():
    test_side_view_uses_display_axis_values()
    test_side_view_clips_display_position_to_axis_limits()
    print("\n✅ All normal stress side-view plot tests passed!")


if __name__ == "__main__":
    run_all_tests()
