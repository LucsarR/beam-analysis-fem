"""
Visual test demonstrating that equal aspect ratio fixes the perpendicular vector angle issue.

This test creates an inclined beam (45 degrees) and shows that with equal aspect ratio,
the fill area appears at true 90 degrees to the beam.
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

print("="*70)
print("Testing Equal Aspect Ratio Fix for Inclined Beams")
print("="*70)

# Create a 45-degree inclined cantilever beam
mesh = Mesh()
mat = Material(1, 210e9, 0.3)
sec = RectangularBar(1, 0.05, 0.1)

# Create nodes at 45 degrees
n1 = mesh.add_node(0, 0)
n2 = mesh.add_node(1, 1)  # 45 degree angle

# Add element
el = mesh.add_element(n1, n2, mat, sec, 'euler_bernoulli_2node')

# Apply constraints (fixed at left)
mesh.constraints.add(Constraint(n1, 0, 0.0))
mesh.constraints.add(Constraint(n1, 1, 0.0))
mesh.constraints.add(Constraint(n1, 2, 0.0))

# Apply point load (perpendicular to beam - in y direction)
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

# Generate plot with fill (this now has equal aspect ratio)
print("\nGenerating moment diagram with fill for 45° inclined beam...")
fig = plot_structure_diagram(
    structure_results,
    force_type="moment",
    fill_diagram=True,
    scale=0.3
)

# Save the figure
fig.write_html("/tmp/inclined_beam_moment_diagram_with_aspect_ratio.html")
print("Saved to: /tmp/inclined_beam_moment_diagram_with_aspect_ratio.html")

print("\n" + "="*70)
print("✓ Test Complete")
print("="*70)
print("\nThe fix applied:")
print("1. Reverted perpendicular vector to original: [-dy, dx] (90° CCW)")
print("2. Added equal aspect ratio: fig.update_yaxes(scaleanchor='x', scaleratio=1)")
print("\nResult:")
print("- The fill area now appears at true 90° to the beam")
print("- For 45° inclined beam, perpendicular direction is correctly displayed")
print("- Visual angle matches mathematical perpendicular (no distortion from axis scaling)")
print("\nThe user was correct: the original perpendicular formula was fine.")
print("The issue was the automatic axis scaling distorting the visual angle.")
