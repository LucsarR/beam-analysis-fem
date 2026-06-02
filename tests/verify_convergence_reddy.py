import sys
import os
import json
import numpy as np

# Ensure path includes workspace root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
M = 1.03

# Reference values from table
ref_data = {
    1: {"RBT": 4946, "MRBT": -46320},
    2: {"RBT": 5821, "MRBT": -26440},
    4: {"RBT": 1918, "MRBT": -30480},
    10: {"RBT": -12550, "MRBT": -30640},
    20: {"RBT": -23370, "MRBT": -30400},
    40: {"RBT": -27890, "MRBT": None},
    80: {"RBT": -30330, "MRBT": None},
    160: {"RBT": -30360, "MRBT": None}
}

def run_analysis(n_elements, etype):
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
    
    pl_m = PointLoad(M, 2)  # Applied moment M
    pl_m.node = nodes[-1]
    mesh.point_loads.append(pl_m)
    
    # Solve
    analysis = BeamAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    
    results = StructureResults(mesh, displacements)
    
    # Shear stress assessed at middle of cross-section (y = 0) at the free node (x = L)
    # The last element results:
    last_er = results.element_results[-1]
    
    # Evaluate at free end of last element (local x = last_er.length)
    tau = last_er.reddy_shear_stress(last_er.length, 0.0)
    
    # Let's also evaluate the theta and dv_dx values at x = L to see if there is any shear locking
    local_disps = last_er.displacements
    theta_val = last_er.element.interpolate_theta(last_er.length, local_disps)
    dv_dx_val = last_er.element.interpolate_dv_dx(last_er.length, local_disps)
    
    # Let's also compute the global DOFs
    return tau, theta_val, dv_dx_val

print(f"{'Elements':<10} | {'RBT Tau (FEM)':<15} | {'RBT Tau (Ref)':<15} | {'MRBT Tau (FEM)':<15} | {'MRBT Tau (Ref)':<15}")
print("-" * 80)

for n in sorted(ref_data.keys()):
    rbt_tau, rbt_theta, rbt_dv = run_analysis(n, "reddy_bickford_2node")
    
    # MRBT
    mrbt_tau, mrbt_theta, mrbt_dv = run_analysis(n, "mrbt_2node")
    
    ref_rbt = ref_data[n]["RBT"]
    ref_mrbt = ref_data[n]["MRBT"]
    
    str_ref_mrbt = f"{ref_mrbt}" if ref_mrbt is not None else "-"
    
    print(f"{n:<10} | {rbt_tau:<15.2f} | {ref_rbt:<15} | {mrbt_tau:<15.2f} | {str_ref_mrbt:<15}")