"""
Visual test for the fill vector direction in plot_structure_diagram.

This script creates a simple beam structure and generates a diagram with fill
to verify that the perpendicular vector direction is correct.
"""

import numpy as np
from fem.mesh import Mesh
from fem.material import Material
from fem.section import RectangularBar
from fem.constraint import Constraint
from fem.load import PointLoad
from fem.analysis import BeamAnalysis
from post_processing.forces import StructureResults
from post_processing.plotter import plot_structure_diagram

# Create a simple horizontal cantilever beam
mesh = Mesh()
mat = Material(1, 210e9, 0.3)
sec = RectangularBar(1, 0.05, 0.1)

# Create nodes
n1 = mesh.add_node(0, 0)
n2 = mesh.add_node(1, 0)

# Add element
el = mesh.add_element(n1, n2, mat, sec, 'euler_bernoulli_2node')

# Apply constraints (fixed at left)
mesh.constraints.add(Constraint(n1, 0, 0.0))
mesh.constraints.add(Constraint(n1, 1, 0.0))
mesh.constraints.add(Constraint(n1, 2, 0.0))

# Apply point load (downward at free end)
P = -1000.0  # N (negative = downward)
load = PointLoad(P, 1)
load.node = n2
mesh.point_loads.append(load)

# Run analysis
analysis = BeamAnalysis(mesh)
analysis.assemble()
displacements = analysis.solve()

# Create structure results
structure_results = StructureResults(mesh, displacements, analysis.get_reactions())

# Generate plot with fill
fig = plot_structure_diagram(
    structure_results,
    force_type="moment",
    fill_diagram=True,
    scale=0.3
)

# Save the figure
print("Generating moment diagram with fill...")
fig.write_html("/tmp/moment_diagram_fill_test.html")
print("Saved to: /tmp/moment_diagram_fill_test.html")

# Also test shear diagram
fig_shear = plot_structure_diagram(
    structure_results,
    force_type="shear",
    fill_diagram=True,
    scale=0.3
)
fig_shear.write_html("/tmp/shear_diagram_fill_test.html")
print("Saved to: /tmp/shear_diagram_fill_test.html")

print("\nDiagrams generated successfully!")
print("For a horizontal beam (pointing right):")
print("- The old perpendicular vector [-dy, dx] = [0, 1] pointed UP")
print("- The new perpendicular vector [dy, -dx] = [0, -1] points DOWN")
print("\nThis change was made to fix the TODO item:")
print("'Arrumar direção do vetor de area' (Fix area vector direction)")
print("\nThe fill area should now be drawn in the correct direction relative to the beam.")
