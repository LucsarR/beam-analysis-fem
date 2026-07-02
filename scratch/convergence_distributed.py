import numpy as np
from fem.mesh import Mesh
from fem.material import Material
from fem.section import RectangularBar
from fem.constraint import Constraint
from fem.load import PointLoad, DistributedLoad
from fem.analysis import BeamAnalysis

E = 10000000.0
NU = 0.3
b = 0.5
h = 0.1

def run_dist_analysis(etype, n_subdiv):
    mesh = Mesh()
    mat = Material(1, E, NU)
    sec = RectangularBar(1, b, h)
    
    # Nodes (corner and ends)
    n1 = mesh.add_node(0.0, 0.0)
    n2 = mesh.add_node(0.0, 1.0)
    n3 = mesh.add_node(1.0, 1.0)
    
    # We will build the mesh with n_subdiv elements per member
    nodes_col = [n1]
    for i in range(1, n_subdiv):
        nodes_col.append(mesh.add_node(0.0, i / n_subdiv))
    nodes_col.append(n2)
    
    nodes_beam = [n2]
    for i in range(1, n_subdiv):
        nodes_beam.append(mesh.add_node(i / n_subdiv, 1.0))
    nodes_beam.append(n3)
    
    # Add elements
    for i in range(n_subdiv):
        el = mesh.add_element(nodes_col[i], nodes_col[i+1], mat, sec, element_type=etype)
        # Apply X load of 5000.0
        dl = DistributedLoad(5000.0, 5000.0, direction='x')
        dl.element = el
        mesh.distributed_loads.append(dl)
        
    for i in range(n_subdiv):
        el = mesh.add_element(nodes_beam[i], nodes_beam[i+1], mat, sec, element_type=etype)
        # Apply Y load of -10000.0
        dl = DistributedLoad(-10000.0, -10000.0, direction='y')
        dl.element = el
        mesh.distributed_loads.append(dl)
        
    # Constraints
    mesh.constraints.add(Constraint(n1, 0, 0.0))
    mesh.constraints.add(Constraint(n1, 1, 0.0))
    # mesh.constraints.add(Constraint(n1, 3, 0.0)) # Free slope
    
    mesh.constraints.add(Constraint(n3, 0, 0.0))
    mesh.constraints.add(Constraint(n3, 1, 0.0))
    # mesh.constraints.add(Constraint(n3, 3, 0.0)) # Free slope
    
    # Solve
    analysis = BeamAnalysis(mesh, structural_behavior="frame")
    analysis.assemble()
    displacements = analysis.solve()
    
    # Node 3 is the last node
    # Find its index in the mesh nodes
    n3_idx = mesh.nodes.index(n3)
    theta_3 = displacements[n3_idx * 4 + 2]
    return theta_3

print(f"{'Elements':<10} | {'RBT Theta':<15} | {'RBT Ref':<15} | {'MRBT Theta':<15} | {'MRBT Ref':<15}")
print("-" * 80)

ref_data = {
    1: {"RBT": 0.29398, "MRBT": 0.653515},
    2: {"RBT": 0.58176, "MRBT": 0.640769},
    4: {"RBT": 0.635934, "MRBT": 0.642391},
    10: {"RBT": 0.646368, "MRBT": 0.643111},
    20: {"RBT": 0.644985, "MRBT": 0.643383}
}

for n in [1, 2, 4, 10, 20]:
    rbt = run_dist_analysis("reddy_bickford_2node", n)
    mrbt = run_dist_analysis("mrbt_2node", n)
    ref_r = ref_data[n]["RBT"]
    ref_m = ref_data[n]["MRBT"]
    print(f"{n:<10} | {abs(rbt):<15.6f} | {ref_r:<15.6f} | {abs(mrbt):<15.6f} | {ref_m:<15.6f}")
