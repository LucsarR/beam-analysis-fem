import numpy as np
from fem.mesh import Mesh
from fem.material import Material
from fem.section import RectangularBar
from fem.constraint import Constraint
from fem.load import PointLoad
from fem.analysis import BeamAnalysis
from post_processing.forces import StructureResults

L = 1.0
E = 10000000.0
NU = 0.3
b = 0.5
h = 0.1
P = -1030.0
M = 1.03

mesh = Mesh()
mat = Material(1, E, NU)
sec = RectangularBar(1, b, h)
nodes = mesh.generate_1d_mesh(0, 0, L, 0, 40, mat, sec, "reddy_bickford_2node")

mesh.constraints.add(Constraint(nodes[0], 0, 0.0))
mesh.constraints.add(Constraint(nodes[0], 1, 0.0))
mesh.constraints.add(Constraint(nodes[0], 2, 0.0))
mesh.constraints.add(Constraint(nodes[0], 3, 0.0))

pl_v = PointLoad(P, 1)
pl_v.node = nodes[-1]
mesh.point_loads.append(pl_v)

pl_m = PointLoad(M, 2)
pl_m.node = nodes[-1]
mesh.point_loads.append(pl_m)

analysis = BeamAnalysis(mesh)
analysis.assemble()
displacements = analysis.solve()

results = StructureResults(mesh, displacements)

print("Shear stress at nodes near the end (x >= 0.85):")
for i, er in enumerate(results.element_results):
    x_start = i * er.length
    x_end = (i + 1) * er.length
    if x_start >= 0.849:
        tau_start = er.reddy_shear_stress(0.0, 0.0)
        tau_end = er.reddy_shear_stress(er.length, 0.0)
        print(f"Element {i+1:2d} (x = {x_start:.3f} to {x_end:.3f}):")
        print(f"  Start: {tau_start:.4f}")
        print(f"  End:   {tau_end:.4f}")
