import numpy as np
from fem.mesh import Mesh
from fem.material import Material
from fem.section import RectangularBar
from fem.constraint import Constraint
from fem.load import PointLoad, DistributedLoad
from fem.analysis import BeamAnalysis
from post_processing.forces import StructureResults

def test_signs(etype):
    L = 2.0
    B = 0.05
    H = 0.1
    E = 210e9
    NU = 0.3
    
    mesh = Mesh()
    mat = Material(1, E, NU)
    sec = RectangularBar(1, B, H)
    
    nodes = mesh.generate_1d_mesh(0, 0, L, 0, 1, mat, sec, etype)
    
    left_node = nodes[0]
    mesh.constraints.add(Constraint(left_node, 0, 0.0))
    mesh.constraints.add(Constraint(left_node, 1, 0.0))
    mesh.constraints.add(Constraint(left_node, 2, 0.0))
    if "reddy" in etype:
        mesh.constraints.add(Constraint(left_node, 3, 0.0))
        
    # Apply downward tip load
    tip_node = nodes[-1]
    load_y = PointLoad(-1.0e4, 1) # downward point load
    load_y.node = tip_node
    mesh.point_loads.append(load_y)
    
    analysis = BeamAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    
    struct_res = StructureResults(mesh, displacements)
    er = struct_res.element_results[0]
    
    # Evaluate at x = 0 (support) or x = L/2
    x_eval = 0.5
    y_top = H / 2.0
    y_bot = -H / 2.0
    
    stress_top = er.kinematic_normal_stress(x_eval, y_top)
    stress_bot = er.kinematic_normal_stress(x_eval, y_bot)
    
    # Compute fallback stress using section formula
    N = er.normal_force(x_eval)
    M = er.bending_moment(x_eval)
    fallback_top = er.element.section.normal_stress(N, M, y_top)
    fallback_bot = er.element.section.normal_stress(N, M, y_bot)
    
    print(f"--- {etype} ---")
    print(f"Tip displacements: {displacements[-4:] if 'reddy' in etype or 'mrbt' in etype else displacements[-3:]}")
    print(f"Kinematic stress at x={x_eval}:")
    print(f"  Top fiber (y={y_top}): {stress_top:+.4e} (Expected: positive/tension)")
    print(f"  Bottom fiber (y={y_bot}): {stress_bot:+.4e} (Expected: negative/compression)")
    print(f"Fallback section stress at x={x_eval}:")
    print(f"  Top fiber (y={y_top}): {fallback_top:+.4e} (Expected: positive/tension)")
    print(f"  Bottom fiber (y={y_bot}): {fallback_bot:+.4e} (Expected: negative/compression)")

if __name__ == "__main__":
    for etype in ["euler_bernoulli_2node", "timoshenko_2node", "timoshenko_3node", "reddy_bickford_2node", "mrbt_2node"]:
        test_signs(etype)
