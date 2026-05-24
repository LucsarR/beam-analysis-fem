from post_processing.plotter import _auto_diagram_points_per_element, plot_structure_diagram


class _Node:
    def __init__(self, node_id, x, y):
        self.id = node_id
        self.x = x
        self.y = y


class _Element:
    def __init__(self, element_id, node_start, node_end):
        self.id = element_id
        self.node_start = node_start
        self.node_end = node_end


class _ElementResult:
    def __init__(self, element):
        self.element = element
        self.length = 1.0

    def bending_moment(self, x):
        return x

    def shear_force(self, x):
        return 1.0

    def normal_force(self, x):
        return -1.0


class _Mesh:
    def __init__(self, nodes):
        self.nodes = nodes


class _StructureResults:
    def __init__(self, nodes, element_results):
        self.mesh = _Mesh(nodes)
        self.element_results = element_results


def _build_mock_structure(n_elements):
    nodes = [_Node(i + 1, float(i), 0.0) for i in range(n_elements + 1)]
    element_results = []
    for i in range(n_elements):
        element = _Element(i + 1, nodes[i], nodes[i + 1])
        element_results.append(_ElementResult(element))
    return _StructureResults(nodes, element_results)


def _first_gradient_trace(fig):
    for trace in fig.data:
        if getattr(trace, "mode", None) == "markers":
            xs = list(trace.x or [])
            if xs and all(x is not None for x in xs):
                return trace
    raise AssertionError("Could not find gradient markers trace")


def test_auto_diagram_points_per_element_scales_by_element_count():
    assert _auto_diagram_points_per_element(1) == 100
    assert _auto_diagram_points_per_element(100) == 10
    assert _auto_diagram_points_per_element(33) == 30
    print("✓ test_auto_diagram_points_per_element_scales_by_element_count passed")


def test_plot_structure_diagram_uses_auto_resolution_when_not_provided():
    one_element_results = _build_mock_structure(1)
    fig_one = plot_structure_diagram(one_element_results, force_type="moment")
    trace_one = _first_gradient_trace(fig_one)
    assert len(trace_one.x) == 100

    hundred_element_results = _build_mock_structure(100)
    fig_hundred = plot_structure_diagram(hundred_element_results, force_type="moment")
    trace_hundred = _first_gradient_trace(fig_hundred)
    assert len(trace_hundred.x) == 10

    print("✓ test_plot_structure_diagram_uses_auto_resolution_when_not_provided passed")


def run_all_tests():
    test_auto_diagram_points_per_element_scales_by_element_count()
    test_plot_structure_diagram_uses_auto_resolution_when_not_provided()
    print("\n✅ All diagram resolution scaling tests passed!")


if __name__ == "__main__":
    run_all_tests()
