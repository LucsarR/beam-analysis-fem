import json
import numpy as np
from fem.mesh import Mesh
from fem.material import Material
from fem.section import RectangularBar
from fem.constraint import Constraint
from fem.load import PointLoad, DistributedLoad
from fem.analysis import BeamAnalysis

# Load properties from JSON
E = 10000000.0
NU = 0.3
b = 0.5
h = 0.1

def run_dist_analysis(etype, constrain_slope, roller_support):
    mesh = Mesh()
    mat = Material(1, E, NU)
    sec = RectangularBar(1, b, h)
    
    # Nodes
    n1 = mesh.add_node(0.0, 0.0)
    n2 = mesh.add_node(0.0, 1.0)
    n3 = mesh.add_node(1.0, 1.0)
    
    # Elements (1 division)
    el1 = mesh.add_element(n1, n2, mat, sec, element_type=etype)
    el2 = mesh.add_element(n2, n3, mat, sec, element_type=etype)
    
    # Constraints
    mesh.constraints.add(Constraint(n1, 0, 0.0))  # u1 = 0
    mesh.constraints.add(Constraint(n1, 1, 0.0))  # v1 = 0
    if constrain_slope:
        mesh.constraints.add(Constraint(n1, 3, 0.0))  # dv/dx1 = 0
        
    if not roller_support:
        mesh.constraints.add(Constraint(n3, 0, 0.0))  # u3 = 0
    mesh.constraints.add(Constraint(n3, 1, 0.0))  # v3 = 0
    if constrain_slope:
        mesh.constraints.add(Constraint(n3, 3, 0.0))  # dv/dx3 = 0
        
    # Distributed loads
    dl1 = DistributedLoad(5000.0, 5000.0, direction='x')  # x load on element 1
    dl1.element = el1
    mesh.distributed_loads.append(dl1)
    
    dl2 = DistributedLoad(-10000.0, -10000.0, direction='y')  # y load on element 2
    dl2.element = el2
    mesh.distributed_loads.append(dl2)
    
    # Solve
    analysis = BeamAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    
    # Node 3 displacements
    disp_node3 = displacements[2*4 : 3*4]
    return disp_node3[2], disp_node3[3]

print("With Node 3 as PINNED (u3=0, v3=0):")
t3_rbt, _ = run_dist_analysis("reddy_bickford_2node", constrain_slope=False, roller_support=False)
t3_mrbt, _ = run_dist_analysis("mrbt_2node", constrain_slope=False, roller_support=False)
print(f"  RBT theta = {t3_rbt:.6f} | MRBT theta = {t3_mrbt:.6f}")

print("\nWith Node 3 as ROLLER (u3=free, v3=0):")
t3_rbt, _ = run_dist_analysis("reddy_bickford_2node", constrain_slope=False, roller_support=True)
t3_mrbt, _ = run_dist_analysis("mrbt_2node", constrain_slope=False, roller_support=True)
print(f"  RBT theta = {t3_rbt:.6f} | MRBT theta = {t3_mrbt:.6f}")
