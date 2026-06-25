import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fem.mesh import Mesh
from fem.material import Material
from fem.section import RectangularBar
from fem.constraint import Constraint
from fem.load import PointLoad, DistributedLoad
from fem.analysis import BeamAnalysis
from post_processing.forces import StructureResults

def fit_residual(x_vals, y_vals, deg):
    """Fit a polynomial of degree `deg` and return the sum of squared residuals."""
    coeffs = np.polyfit(x_vals, y_vals, deg)
    y_fit = np.polyval(coeffs, x_vals)
    return float(np.sum((y_vals - y_fit) ** 2))

def run_verification():
    print("=" * 80)
    print("NORMAL STRESS DISTRIBUTION VERIFICATION REPORT")
    print("=" * 80)
    
    # Beam parameters
    L = 2.0
    B = 0.05
    H = 0.1
    E = 210e9
    NU = 0.3
    
    element_types = [
        "euler_bernoulli_2node",
        "euler_bernoulli_3node",
        "timoshenko_2node",
        "timoshenko_3node",
        "reddy_bickford_2node"
    ]
    
    results_summary = []
    
    for etype in element_types:
        # 1. Setup mesh, material, section
        mesh = Mesh()
        mat = Material(1, E, NU)
        sec = RectangularBar(1, B, H)
        
        # 1 element mesh (which generates appropriate number of nodes)
        nodes = mesh.generate_1d_mesh(0, 0, L, 0, 1, mat, sec, etype)
        
        # 2. Add cantilever boundary conditions at left end
        left_node = nodes[0]
        mesh.constraints.add(Constraint(left_node, 0, 0.0))  # u
        mesh.constraints.add(Constraint(left_node, 1, 0.0))  # v
        mesh.constraints.add(Constraint(left_node, 2, 0.0))  # theta
        if "reddy" in etype:
            mesh.constraints.add(Constraint(left_node, 3, 0.0))  # dv/dx
            
        # 3. Add point loads at the tip (right-most node) to produce combined tension & bending
        # Note: Tip node is nodes[-1]
        tip_node = nodes[-1]
        # Axial force Fx = 1.0e5 N
        load_x = PointLoad(1.0e5, 0)
        load_x.node = tip_node
        mesh.point_loads.append(load_x)
        
        # Add distributed load in y direction (linear from -1.0e4 to 0.0)
        for el in mesh.elements:
            dist_load = DistributedLoad(magnitude_start=-1.0e4, magnitude_end=0.0, direction='y')
            dist_load.element = el
            mesh.distributed_loads.append(dist_load)
        
        # 4. Solve
        analysis = BeamAnalysis(mesh)
        analysis.assemble()
        displacements = analysis.solve()
        
        # 5. Post-process forces
        struct_res = StructureResults(mesh, displacements)
        er = struct_res.element_results[0]
        
        # ---------------------------------------------------------------------
        # Verification 1: Cross-sectional distribution along height y (at x = L/2)
        # ---------------------------------------------------------------------
        x_mid = L / 2.0
        ys = np.linspace(-H/2.0, H/2.0, 11)
        sigma_ys = np.array([er.kinematic_normal_stress(x_mid, y) for y in ys])
        
        res_y_deg1 = fit_residual(ys, sigma_ys, 1)
        res_y_deg3 = fit_residual(ys, sigma_ys, 3)
        
        is_y_linear = res_y_deg1 < 1e-10
        is_y_cubic = res_y_deg3 < 1e-10 and not is_y_linear
        
        y_behavior = "LINEAR" if is_y_linear else ("CUBIC" if is_y_cubic else "NONLINEAR")
        
        # ---------------------------------------------------------------------
        # Verification 2: Spanwise distribution along length x (at y = H/2)
        # ---------------------------------------------------------------------
        y_top = H / 2.0
        xs = np.linspace(0.0, L, 11)
        sigma_xs = np.array([er.kinematic_normal_stress(x, y_top) for x in xs])
        
        res_x_deg1 = fit_residual(xs, sigma_xs, 1)
        res_x_deg3 = fit_residual(xs, sigma_xs, 3)
        
        is_x_linear = res_x_deg1 < 1e-10
        is_x_cubic = res_x_deg3 < 1e-10 and not is_x_linear
        
        x_behavior = "LINEAR" if is_x_linear else ("CUBIC" if is_x_cubic else "NONLINEAR")
        
        results_summary.append({
            "Element": etype,
            "y_behavior": y_behavior,
            "y_res_deg1": res_y_deg1,
            "y_res_deg3": res_y_deg3,
            "x_behavior": x_behavior,
            "x_res_deg1": res_x_deg1,
            "x_res_deg3": res_x_deg3,
        })
        
    print(f"{'Element Type':<25} | {'Stress vs y (height)':<25} | {'Stress vs x (span)':<25}")
    print("-" * 80)
    for r in results_summary:
        y_desc = f"{r['y_behavior']} (res={r['y_res_deg1']:.1e})" if r['y_behavior'] == "LINEAR" else f"{r['y_behavior']} (res3={r['y_res_deg3']:.1e})"
        x_desc = f"{r['x_behavior']} (res={r['x_res_deg1']:.1e})" if r['x_behavior'] == "LINEAR" else f"{r['x_behavior']} (res3={r['x_res_deg3']:.1e})"
        print(f"{r['Element']:<25} | {y_desc:<25} | {x_desc:<25}")
    print("=" * 80)
    
    # Check predictions
    assert results_summary[0]["y_behavior"] == "LINEAR"  # EB 2-node
    assert results_summary[1]["y_behavior"] == "LINEAR"  # EB 3-node
    assert results_summary[2]["y_behavior"] == "LINEAR"  # Timoshenko 2-node
    assert results_summary[3]["y_behavior"] == "LINEAR"  # Timoshenko 3-node
    assert results_summary[4]["y_behavior"] == "CUBIC"   # Reddy-Bickford 2-node
    
    assert results_summary[0]["x_behavior"] == "LINEAR"  # EB 2-node (M is linear in x)
    assert results_summary[1]["x_behavior"] == "CUBIC"   # EB 3-node (bending shapes are quintic)
    assert results_summary[2]["x_behavior"] == "LINEAR"  # Timoshenko 2-node
    assert results_summary[3]["x_behavior"] == "LINEAR"  # Timoshenko 3-node
    assert results_summary[4]["x_behavior"] == "LINEAR"  # Reddy-Bickford 2-node (M is linear in x)
    
    print("All theoretical predictions verified successfully!")

if __name__ == "__main__":
    run_verification()
