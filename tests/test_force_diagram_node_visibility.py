#!/usr/bin/env python3
"""
Regression test for force-diagram rendering:
intermediate sampling points must not be drawn on the element axis as if they were nodes.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fem.mesh import Mesh
from fem.material import Material
from fem.section import RectangularBar
from fem.constraint import Constraint
from fem.load import PointLoad
from fem.analysis import BeamAnalysis
from post_processing.forces import StructureResults
from post_processing.plotter import plot_structure_diagram


def test_force_diagram_points_are_off_axis():
    mesh = Mesh()
    mat = Material(1, 210e9, 0.3)
    sec = RectangularBar(1, 0.05, 0.1)

    n1 = mesh.add_node(0.0, 0.0)
    n2 = mesh.add_node(1.0, 0.0)
    mesh.add_element(n1, n2, mat, sec, "euler_bernoulli_2node")

    mesh.constraints.add(Constraint(n1, 0, 0.0))
    mesh.constraints.add(Constraint(n1, 1, 0.0))
    mesh.constraints.add(Constraint(n1, 2, 0.0))

    load = PointLoad(-1000.0, 1)
    load.node = n2
    mesh.point_loads.append(load)

    analysis = BeamAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    structure_results = StructureResults(mesh, displacements, analysis.get_reactions())

    fig = plot_structure_diagram(
        structure_results,
        force_type="moment",
        n_points=30,
        fill_diagram=False,
    )

    node_traces = [
        tr for tr in fig.data
        if getattr(tr, "mode", None) == "markers+text" and str(getattr(tr, "name", "")).startswith("Node ")
    ]
    assert len(node_traces) == 2, f"Expected 2 node traces, got {len(node_traces)}"

    diagram_traces = [
        tr for tr in fig.data
        if getattr(tr, "mode", None) == "markers" and len(getattr(tr, "x", [])) > 2
    ]
    assert diagram_traces, "Expected a diagram marker trace"

    y_values = [float(y) for y in diagram_traces[0].y]
    assert any(abs(y) > 1e-10 for y in y_values), (
        "Expected force-diagram sampling points to be off the element axis"
    )


if __name__ == "__main__":
    test_force_diagram_points_are_off_axis()
    print("✓ test_force_diagram_points_are_off_axis passed")
