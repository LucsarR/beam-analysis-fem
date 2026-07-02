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

def analyze(n_elements):
    mesh = Mesh()
    mat = Material(1, E, NU)
    sec = RectangularBar(1, b, h)
    nodes = mesh.generate_1d_mesh(0, 0, L, 0, n_elements, mat, sec, "reddy_bickford_2node")
    
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
    
    # Let's inspect the last element
    last_er = results.element_results[-1]
    
    # Stress at start of last element (x_local = 0.0)
    tau_start = last_er.reddy_shear_stress(0.0, 0.0)
    # Stress at end of last element (x_local = last_er.length)
    tau_end = last_er.reddy_shear_stress(last_er.length, 0.0)
    
    print(f"Elements: {n_elements:<3} | Last Element Le: {last_er.length:.6f}")
    print(f"  tau at start of last element (x = {L - last_er.length:.4f}): {tau_start:.4f}")
    print(f"  tau at end of last element   (x = {L:.4f}):         {tau_end:.4f}")

for n in [20, 40, 80, 160]:
    analyze(n)
