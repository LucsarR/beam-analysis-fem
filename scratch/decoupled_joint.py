import numpy as np
from fem.material import Material
from fem.section import RectangularBar
from fem.element import ReddyBickfordElement2Node, MRBTElement2Node
from fem.constraint import Constraint
from fem.load import DistributedLoad

# Parameters
L = 1.0
E = 10000000.0
NU = 0.3
b = 0.5
h = 0.1

def solve_decoupled(etype):
    mat = Material(1, E, NU)
    sec = RectangularBar(1, b, h)
    
    # We will instantiate elements of type etype
    # Element 1: Column from (0,0) to (0,1)
    # Element 2: Beam from (0,1) to (1,1)
    # We will manually build elements to get their local stiffness matrices
    class DummyNode:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            
    n1 = DummyNode(0.0, 0.0)
    n2 = DummyNode(0.0, 1.0)
    n3 = DummyNode(1.0, 1.0)
    
    if etype == "rbt":
        el1 = ReddyBickfordElement2Node(1, n1, n2, mat, sec)
        el2 = ReddyBickfordElement2Node(2, n2, n3, mat, sec)
    else:
        el1 = MRBTElement2Node(1, n1, n2, mat, sec)
        el2 = MRBTElement2Node(2, n2, n3, mat, sec)
        
    # Local stiffness matrices (8x8)
    # Local DOFs: [u1, v1, theta1, dv_dx1, u2, v2, theta2, dv_dx2]
    # For Element 1: local axes are rotated (c=0, s=1)
    K1_global = el1.stiffness_matrix()
    F1_global = el1.compute_equivalent_nodal_loads(DistributedLoad(5000.0, 5000.0, direction='x'))
    
    # For Element 2: local axes are aligned (c=1, s=0)
    K2_global = el2.stiffness_matrix()
    F2_global = el2.compute_equivalent_nodal_loads(DistributedLoad(-10000.0, -10000.0, direction='y'))
    
    # Custom 13-DOF system
    # DOF mapping:
    # 0: u1
    # 1: v1
    # 2: theta1
    # 3: dv_dx1
    # 4: u2
    # 5: v2
    # 6: theta2
    # 7: dv_dx2_col (from Element 1 node 2)
    # 8: dv_dx2_beam (from Element 2 node 1)
    # 9: u3
    # 10: v3
    # 11: theta3
    # 12: dv_dx3
    
    K = np.zeros((13, 13))
    F = np.zeros(13)
    
    # Element 1 DOFs in global 13-DOF system:
    # [u1, v1, theta1, dv_dx1] -> [0, 1, 2, 3]
    # [u2, v2, theta2, dv_dx2_col] -> [4, 5, 6, 7]
    map1 = [0, 1, 2, 3, 4, 5, 6, 7]
    for i in range(8):
        F[map1[i]] += F1_global[i]
        for j in range(8):
            K[map1[i], map1[j]] += K1_global[i, j]
            
    # Element 2 DOFs in global 13-DOF system:
    # [u2, v2, theta2, dv_dx2_beam] -> [4, 5, 6, 8]
    # [u3, v3, theta3, dv_dx3] -> [9, 10, 11, 12]
    map2 = [4, 5, 6, 8, 9, 10, 11, 12]
    for i in range(8):
        F[map2[i]] += F2_global[i]
        for j in range(8):
            K[map2[i], map2[j]] += K2_global[i, j]
            
    # Boundary conditions: Simply supported
    # u1 = 0, v1 = 0, u3 = 0, v3 = 0
    # Constrained global DOFs: 0, 1, 9, 10
    active_dofs = [2, 3, 4, 5, 6, 7, 8, 11, 12]
    
    K_active = K[np.ix_(active_dofs, active_dofs)]
    F_active = F[active_dofs]
    
    disps_active = np.linalg.solve(K_active, F_active)
    
    disps = np.zeros(13)
    disps[active_dofs] = disps_active
    
    # Node 3 is at index 11 (theta3)
    return disps[11]

t3_rbt = solve_decoupled("rbt")
t3_mrbt = solve_decoupled("mrbt")
print(f"Decoupled Corner Joint Joint Slope:")
print(f"  RBT theta_3 = {abs(t3_rbt):.6f} (Ref: 0.293980)")
print(f"  MRBT theta_3 = {abs(t3_mrbt):.6f} (Ref: 0.653515)")
