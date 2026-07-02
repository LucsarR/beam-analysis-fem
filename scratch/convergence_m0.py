import numpy as np
from fem.mesh import Mesh
from fem.material import Material
from fem.section import RectangularBar
from fem.constraint import Constraint
from fem.load import PointLoad
from fem.analysis import BeamAnalysis
from post_processing.forces import StructureResults

# Test properties
L = 1.0
E = 10000000.0
NU = 0.3
b = 0.5
h = 0.1
P = -1030.0

def run_analysis(n_elements, etype, M_val):
    mesh = Mesh()
    mat = Material(1, E, NU)
    sec = RectangularBar(1, b, h)
    
    # Generate 1D mesh
    nodes = mesh.generate_1d_mesh(0, 0, L, 0, n_elements, mat, sec, etype)
    
    # Boundary conditions: Clamped at left node (Node 1)
    mesh.constraints.add(Constraint(nodes[0], 0, 0.0))  # u
    mesh.constraints.add(Constraint(nodes[0], 1, 0.0))  # v
    mesh.constraints.add(Constraint(nodes[0], 2, 0.0))  # theta
    mesh.constraints.add(Constraint(nodes[0], 3, 0.0))  # dv/dx
    
    # Point loads at right node (Node n+1)
    pl_v = PointLoad(P, 1)  # Vertical load P
    pl_v.node = nodes[-1]
    mesh.point_loads.append(pl_v)
    
    if M_val != 0.0:
        pl_m = PointLoad(M_val, 2)  # Applied moment M
        pl_m.node = nodes[-1]
        mesh.point_loads.append(pl_m)
        
    # Solve
    analysis = BeamAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    
    results = StructureResults(mesh, displacements)
    last_er = results.element_results[-1]
    tau = last_er.reddy_shear_stress(last_er.length, 0.0)
    return tau

print("With M = 0.0:")
print(f"{'Elements':<10} | {'RBT':<15} | {'MRBT':<15}")
for n in [1, 2, 4, 10, 20, 40, 80, 160]:
    rbt = run_analysis(n, "reddy_bickford_2node", 0.0)
    mrbt = run_analysis(n, "mrbt_2node", 0.0)
    print(f"{n:<10} | {rbt:<15.2f} | {mrbt:<15.2f}")

print("\nWith M = 1.03:")
print(f"{'Elements':<10} | {'RBT':<15} | {'MRBT':<15}")
for n in [1, 2, 4, 10, 20, 40, 80, 160]:
    rbt = run_analysis(n, "reddy_bickford_2node", 1.03)
    mrbt = run_analysis(n, "mrbt_2node", 1.03)
    print(f"{n:<10} | {rbt:<15.2f} | {mrbt:<15.2f}")
