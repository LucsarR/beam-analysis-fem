from fem.section import RectangularBar
from post_processing.plotter import plot_normal_stress_distribution


class _Element:
    def __init__(self, section):
        self.section = section


class _ElementResult:
    def __init__(self, section, normal=100.0, moment=10.0):
        self.element = _Element(section)
        self._normal = normal
        self._moment = moment

    def normal_force(self, x):
        return self._normal

    def bending_moment(self, x):
        return self._moment


def test_cross_section_title_uses_local_x_by_default():
    sec = RectangularBar(1, width=0.2, height=0.4)
    er = _ElementResult(sec)
    fig = plot_normal_stress_distribution(er, x=0.25)

    assert "x=0.25" in fig.layout.title.text
    print("✓ test_cross_section_title_uses_local_x_by_default passed")


def test_cross_section_title_can_use_element_level_x():
    sec = RectangularBar(1, width=0.2, height=0.4)
    er = _ElementResult(sec)
    fig = plot_normal_stress_distribution(er, x=0.25, display_x=1.75)

    assert "x=1.75" in fig.layout.title.text
    print("✓ test_cross_section_title_can_use_element_level_x passed")


def run_all_tests():
    test_cross_section_title_uses_local_x_by_default()
    test_cross_section_title_can_use_element_level_x()
    print("\n✅ All normal stress cross-section plot tests passed!")


if __name__ == "__main__":
    run_all_tests()
