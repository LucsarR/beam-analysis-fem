"""
Comprehensive convergence tests for complex structures with multiple loads, constraints, and angles.

This test suite addresses the TODO in fem/mesh.py by verifying that:
1. Mesh correctly handles structures with elements at various angles
2. Mesh correctly handles multiple point loads and distributed loads simultaneously
3. Mesh correctly handles complex boundary conditions with multiple constraints
4. Both Euler-Bernoulli and Timoshenko elements converge correctly in complex scenarios
5. Force and displacement solutions converge as mesh is refined

Test cases include:
- L-shaped frame with angled elements
- Frame structure with multiple loads
- Beam with multiple point and distributed loads
- Structure with various constraint configurations
"""

import numpy as np
from fem.mesh import Mesh
from fem.material import Material
from fem.section import RectangularBar
from fem.constraint import Constraint
from fem.load import PointLoad, DistributedLoad
from fem.analysis import EulerBernoulliAnalysis


def test_angled_cantilever_convergence():
    """
    Test convergence for a cantilever beam at 45 degrees with point load at tip.
    Verifies that mesh handles angled elements correctly.
    """
    print("\n" + "="*70)
    print("Test: Angled Cantilever Beam (45°) Convergence")
    print("="*70)
    
    # Beam properties
    L = 2.0  # Length (m)
    angle = 45.0  # degrees
    angle_rad = np.radians(angle)
    P = -1000.0  # Point load at tip (N, downward)
    E = 210e9  # Young's modulus (Pa)
    b = 0.05  # Width (m)
    h = 0.1  # Height (m)
    
    print(f"\nBeam properties: L={L}m, angle={angle}°, b={b}m, h={h}m")
    print(f"Load: P={P}N at tip (vertical)")
    
    # Test with different mesh refinements
    mesh_sizes = [2, 4, 8, 16, 32]
    tip_deflections_y = []
    tip_deflections_x = []
    
    print(f"\n{'Mesh':<8} {'Tip Deflection Y (mm)':<25} {'Tip Deflection X (mm)':<25}")
    print("-" * 70)
    
    for n_elements in mesh_sizes:
        # Create mesh with angled beam
        mesh = Mesh()
        mat = Material(1, E, 0.3)
        sec = RectangularBar(1, b, h)
        
        # End coordinates for angled beam
        x_end = L * np.cos(angle_rad)
        y_end = L * np.sin(angle_rad)
        
        # Generate 1D mesh at angle
        nodes = mesh.generate_1d_mesh(0, 0, x_end, y_end, n_elements, mat, sec, 'euler_bernoulli_2node')
        
        # Apply boundary conditions (cantilever: fixed at left end)
        mesh.constraints.add(Constraint(nodes[0], 0, 0.0))  # Fix x
        mesh.constraints.add(Constraint(nodes[0], 1, 0.0))  # Fix y
        mesh.constraints.add(Constraint(nodes[0], 2, 0.0))  # Fix rotation
        
        # Apply point load at tip (vertical, downward)
        load = PointLoad(P, 1)  # Direction 1 = y-direction (vertical)
        load.node = nodes[-1]
        mesh.point_loads.append(load)
        
        # Analyze
        analysis = EulerBernoulliAnalysis(mesh)
        analysis.assemble()
        displacements = analysis.solve()
        
        # Extract tip deflection
        tip_node_id = nodes[-1].id
        tip_deflection_x = displacements[3*(tip_node_id-1)]      # x displacement
        tip_deflection_y = displacements[3*(tip_node_id-1) + 1]  # y displacement
        tip_deflections_x.append(tip_deflection_x)
        tip_deflections_y.append(tip_deflection_y)
        
        print(f"{n_elements:<8} {tip_deflection_y*1000:<25.6f} {tip_deflection_x*1000:<25.6f}")
    
    # Verify convergence
    print("\nConvergence verification:")
    
    # Check that solution converges (differences decrease with refinement)
    converging = True
    for i in range(1, len(tip_deflections_y)-1):
        diff_current = abs(tip_deflections_y[i+1] - tip_deflections_y[i])
        diff_previous = abs(tip_deflections_y[i] - tip_deflections_y[i-1])
        # Difference should decrease or stay similar
        if diff_current > diff_previous * 1.1:  # Allow 10% tolerance
            converging = False
            print(f"  ⚠ Warning: Convergence slowed between {mesh_sizes[i]} and {mesh_sizes[i+1]} elements")
    
    if converging:
        print("  ✓ Displacement solution converges with mesh refinement")
    else:
        print("  ✗ Displacement solution does not converge properly")
    
    assert converging, "Angled beam solution should converge with mesh refinement"
    
    print("\n✅ Angled cantilever convergence test passed")


def test_l_shaped_frame_convergence():
    """
    Test convergence for an L-shaped frame with elements at 90° to each other.
    Horizontal beam (2m) connected to vertical beam (1m) with tip load.
    """
    print("\n" + "="*70)
    print("Test: L-Shaped Frame Convergence")
    print("="*70)
    
    # Frame properties
    L_horizontal = 2.0  # Horizontal beam length (m)
    L_vertical = 1.0    # Vertical beam length (m)
    P = -500.0  # Point load at top (N, downward)
    E = 210e9   # Young's modulus (Pa)
    b = 0.05    # Width (m)
    h = 0.1     # Height (m)
    
    print(f"\nFrame properties: L_horiz={L_horizontal}m, L_vert={L_vertical}m")
    print(f"Load: P={P}N at top of vertical member")
    
    # Test with different mesh refinements
    mesh_sizes = [2, 4, 8, 16]
    tip_deflections = []
    
    print(f"\n{'Mesh/Element':<15} {'Tip Deflection Y (mm)':<25}")
    print("-" * 45)
    
    for n_elements_per_member in mesh_sizes:
        # Create mesh
        mesh = Mesh()
        mat = Material(1, E, 0.3)
        sec = RectangularBar(1, b, h)
        
        # Create horizontal beam (fixed at left, free at right where vertical connects)
        nodes_horiz = mesh.generate_1d_mesh(0, 0, L_horizontal, 0, 
                                           n_elements_per_member, mat, sec, 'euler_bernoulli_2node')
        
        # Create vertical beam connecting to the end node of horizontal beam
        # Use the last node of horizontal beam as the starting point
        nodes_vert = [nodes_horiz[-1]]  # Start with the connection node
        for i in range(1, n_elements_per_member + 1):
            y = L_vertical * i / n_elements_per_member
            node = mesh.add_node(L_horizontal, y)
            nodes_vert.append(node)
        
        # Add vertical beam elements
        for i in range(n_elements_per_member):
            mesh.add_element(nodes_vert[i], nodes_vert[i+1], mat, sec, 'euler_bernoulli_2node')
        
        # Apply boundary conditions (fixed at left end of horizontal beam)
        mesh.constraints.add(Constraint(nodes_horiz[0], 0, 0.0))  # Fix x
        mesh.constraints.add(Constraint(nodes_horiz[0], 1, 0.0))  # Fix y
        mesh.constraints.add(Constraint(nodes_horiz[0], 2, 0.0))  # Fix rotation
        
        # Apply point load at top of vertical beam
        load = PointLoad(P, 1)  # Direction 1 = y-direction
        load.node = nodes_vert[-1]
        mesh.point_loads.append(load)
        
        # Analyze
        analysis = EulerBernoulliAnalysis(mesh)
        analysis.assemble()
        displacements = analysis.solve()
        
        # Extract tip deflection at top of vertical beam
        tip_node_id = nodes_vert[-1].id
        tip_deflection = displacements[3*(tip_node_id-1) + 1]  # y displacement
        tip_deflections.append(tip_deflection)
        
        print(f"{n_elements_per_member:<15} {tip_deflection*1000:<25.6f}")
    
    # Verify convergence
    print("\nConvergence verification:")
    
    converging = True
    for i in range(1, len(tip_deflections)-1):
        diff_current = abs(tip_deflections[i+1] - tip_deflections[i])
        diff_previous = abs(tip_deflections[i] - tip_deflections[i-1])
        if diff_current > diff_previous * 1.2:  # Allow 20% tolerance for complex structure
            converging = False
            print(f"  ⚠ Warning: Convergence slowed between {mesh_sizes[i]} and {mesh_sizes[i+1]} elements")
    
    if converging:
        print("  ✓ L-shaped frame solution converges with mesh refinement")
    else:
        print("  ✗ L-shaped frame solution does not converge properly")
    
    assert converging, "L-shaped frame solution should converge with mesh refinement"
    
    print("\n✅ L-shaped frame convergence test passed")


def test_multiple_loads_convergence():
    """
    Test convergence for a simply supported beam with multiple point loads and distributed loads.
    """
    print("\n" + "="*70)
    print("Test: Beam with Multiple Loads Convergence")
    print("="*70)
    
    # Beam properties
    L = 4.0     # Length (m)
    P1 = -1000.0  # Point load at L/4 (N)
    P2 = -1500.0  # Point load at 3L/4 (N)
    w = -500.0    # Distributed load on middle half (N/m)
    E = 210e9   # Young's modulus (Pa)
    b = 0.1     # Width (m)
    h = 0.2     # Height (m)
    
    print(f"\nBeam properties: L={L}m, b={b}m, h={h}m")
    print(f"Point load 1: P1={P1}N at x=L/4")
    print(f"Point load 2: P2={P2}N at x=3L/4")
    print(f"Distributed load: w={w}N/m on middle half of beam")
    
    # Test with different mesh refinements (must be multiples of 4 for node placement)
    mesh_sizes = [4, 8, 16, 32]
    midspan_deflections = []
    
    print(f"\n{'Mesh':<8} {'Midspan Deflection (mm)':<25}")
    print("-" * 40)
    
    for n_elements in mesh_sizes:
        # Create mesh
        mesh = Mesh()
        mat = Material(1, E, 0.3)
        sec = RectangularBar(1, b, h)
        
        # Generate 1D mesh
        nodes = mesh.generate_1d_mesh(0, 0, L, 0, n_elements, mat, sec, 'euler_bernoulli_2node')
        
        # Apply boundary conditions (simply supported)
        mesh.constraints.add(Constraint(nodes[0], 0, 0.0))   # Fix x at left
        mesh.constraints.add(Constraint(nodes[0], 1, 0.0))   # Fix y at left
        mesh.constraints.add(Constraint(nodes[-1], 1, 0.0))  # Fix y at right
        
        # Apply first point load at L/4
        load1 = PointLoad(P1, 1)
        load1.node = nodes[n_elements // 4]
        mesh.point_loads.append(load1)
        
        # Apply second point load at 3L/4
        load2 = PointLoad(P2, 1)
        load2.node = nodes[3 * n_elements // 4]
        mesh.point_loads.append(load2)
        
        # Apply distributed load on middle half (from L/4 to 3L/4)
        for i in range(n_elements // 4, 3 * n_elements // 4):
            element = mesh.elements[i]
            dist_load = DistributedLoad(magnitude_start=w, direction='y')
            dist_load.element = element
            mesh.distributed_loads.append(dist_load)
        
        # Analyze
        analysis = EulerBernoulliAnalysis(mesh)
        analysis.assemble()
        displacements = analysis.solve()
        
        # Extract midspan deflection
        midspan_node_id = nodes[n_elements // 2].id
        midspan_deflection = displacements[3*(midspan_node_id-1) + 1]
        midspan_deflections.append(midspan_deflection)
        
        print(f"{n_elements:<8} {midspan_deflection*1000:<25.6f}")
    
    # Verify convergence
    print("\nConvergence verification:")
    
    converging = True
    for i in range(1, len(midspan_deflections)-1):
        diff_current = abs(midspan_deflections[i+1] - midspan_deflections[i])
        diff_previous = abs(midspan_deflections[i] - midspan_deflections[i-1])
        if diff_current > diff_previous * 1.2:  # Allow 20% tolerance
            converging = False
            print(f"  ⚠ Warning: Convergence slowed between {mesh_sizes[i]} and {mesh_sizes[i+1]} elements")
    
    if converging:
        print("  ✓ Multiple loads solution converges with mesh refinement")
    else:
        print("  ✗ Multiple loads solution does not converge properly")
    
    assert converging, "Multiple loads solution should converge with mesh refinement"
    
    print("\n✅ Multiple loads convergence test passed")


def test_timoshenko_angled_beam_convergence():
    """
    Test convergence for Timoshenko elements on an angled beam.
    Verifies that Timoshenko elements handle angles correctly.
    """
    print("\n" + "="*70)
    print("Test: Timoshenko Angled Beam (30°) Convergence")
    print("="*70)
    
    # Beam properties
    L = 2.0  # Length (m)
    angle = 30.0  # degrees
    angle_rad = np.radians(angle)
    P = -1000.0  # Point load at tip (N, downward)
    E = 210e9  # Young's modulus (Pa)
    b = 0.1  # Width (m)
    h = 0.2  # Height (m) - moderately thick for shear effects
    
    print(f"\nBeam properties: L={L}m, angle={angle}°, b={b}m, h={h}m")
    print(f"Load: P={P}N at tip (vertical)")
    
    # Test with different mesh refinements
    mesh_sizes = [2, 4, 8, 16, 32]
    tip_deflections_y = []
    
    print(f"\n{'Mesh':<8} {'Tip Deflection Y (mm)':<25}")
    print("-" * 40)
    
    for n_elements in mesh_sizes:
        # Create mesh with angled beam using Timoshenko elements
        mesh = Mesh()
        mat = Material(1, E, 0.3)
        sec = RectangularBar(1, b, h)
        
        # End coordinates for angled beam
        x_end = L * np.cos(angle_rad)
        y_end = L * np.sin(angle_rad)
        
        # Generate 1D mesh at angle with Timoshenko elements
        nodes = mesh.generate_1d_mesh(0, 0, x_end, y_end, n_elements, mat, sec, 'timoshenko_2node')
        
        # Apply boundary conditions (cantilever: fixed at left end)
        mesh.constraints.add(Constraint(nodes[0], 0, 0.0))  # Fix x
        mesh.constraints.add(Constraint(nodes[0], 1, 0.0))  # Fix y
        mesh.constraints.add(Constraint(nodes[0], 2, 0.0))  # Fix rotation
        
        # Apply point load at tip (vertical, downward)
        load = PointLoad(P, 1)
        load.node = nodes[-1]
        mesh.point_loads.append(load)
        
        # Analyze
        # Note: EulerBernoulliAnalysis is a generic analysis class that works with any element type
        # via polymorphism (calls element.stiffness_matrix() which each element implements)
        analysis = EulerBernoulliAnalysis(mesh)
        analysis.assemble()
        displacements = analysis.solve()
        
        # Extract tip deflection
        tip_node_id = nodes[-1].id
        tip_deflection_y = displacements[3*(tip_node_id-1) + 1]  # y displacement
        tip_deflections_y.append(tip_deflection_y)
        
        print(f"{n_elements:<8} {tip_deflection_y*1000:<25.6f}")
    
    # Verify convergence
    print("\nConvergence verification:")
    
    converging = True
    for i in range(1, len(tip_deflections_y)-1):
        diff_current = abs(tip_deflections_y[i+1] - tip_deflections_y[i])
        diff_previous = abs(tip_deflections_y[i] - tip_deflections_y[i-1])
        if diff_current > diff_previous * 1.1:  # Allow 10% tolerance
            converging = False
            print(f"  ⚠ Warning: Convergence slowed between {mesh_sizes[i]} and {mesh_sizes[i+1]} elements")
    
    if converging:
        print("  ✓ Timoshenko angled beam solution converges with mesh refinement")
    else:
        print("  ✗ Timoshenko angled beam solution does not converge properly")
    
    assert converging, "Timoshenko angled beam solution should converge with mesh refinement"
    
    print("\n✅ Timoshenko angled beam convergence test passed")


def test_complex_structure_forces_convergence():
    """
    Test convergence of force calculations (moment, shear) in a complex structure.
    Simply supported beam with multiple loads - verify forces converge.
    """
    print("\n" + "="*70)
    print("Test: Forces Convergence in Complex Structure")
    print("="*70)
    
    # Beam properties
    L = 3.0     # Length (m)
    P = -2000.0   # Point load at midspan (N)
    w = -1000.0   # Distributed load (N/m)
    E = 210e9   # Young's modulus (Pa)
    b = 0.1     # Width (m)
    h = 0.2     # Height (m)
    
    print(f"\nBeam properties: L={L}m, b={b}m, h={h}m")
    print(f"Point load: P={P}N at midspan")
    print(f"Distributed load: w={w}N/m on entire beam")
    
    # Test with different mesh refinements
    mesh_sizes = [4, 8, 16, 32]
    midspan_moments = []
    quarter_shears = []
    
    print(f"\n{'Mesh':<8} {'Midspan Moment (N·m)':<25} {'Quarter-span Shear (N)':<25}")
    print("-" * 65)
    
    for n_elements in mesh_sizes:
        # Create mesh (must be even for midspan node)
        mesh = Mesh()
        mat = Material(1, E, 0.3)
        sec = RectangularBar(1, b, h)
        
        # Generate 1D mesh
        nodes = mesh.generate_1d_mesh(0, 0, L, 0, n_elements, mat, sec, 'euler_bernoulli_2node')
        
        # Apply boundary conditions (simply supported)
        mesh.constraints.add(Constraint(nodes[0], 0, 0.0))   # Fix x at left
        mesh.constraints.add(Constraint(nodes[0], 1, 0.0))   # Fix y at left
        mesh.constraints.add(Constraint(nodes[-1], 1, 0.0))  # Fix y at right
        
        # Apply point load at midspan
        load = PointLoad(P, 1)
        load.node = nodes[n_elements // 2]
        mesh.point_loads.append(load)
        
        # Apply distributed load to all elements
        for element in mesh.elements:
            dist_load = DistributedLoad(magnitude_start=w, direction='y')
            dist_load.element = element
            mesh.distributed_loads.append(dist_load)
        
        # Analyze
        analysis = EulerBernoulliAnalysis(mesh)
        analysis.assemble()
        displacements = analysis.solve()
        
        # Calculate moment at midspan element
        # For even number of elements, midspan node is at n_elements // 2
        # The element ending at midspan is at index n_elements // 2 - 1
        midspan_element_idx = n_elements // 2 - 1
        midspan_element = mesh.elements[midspan_element_idx]
        
        node_start_id = midspan_element.node_start.id
        node_end_id = midspan_element.node_end.id
        u_global = np.array([
            displacements[3*(node_start_id-1)],
            displacements[3*(node_start_id-1) + 1],
            displacements[3*(node_start_id-1) + 2],
            displacements[3*(node_end_id-1)],
            displacements[3*(node_end_id-1) + 1],
            displacements[3*(node_end_id-1) + 2]
        ])
        u_local = midspan_element.R.T @ u_global
        # Get moment at the END of this element (which is at the midspan node)
        moment = midspan_element.bending_moment(midspan_element.length, u_local)
        midspan_moments.append(moment)
        
        # Calculate shear at quarter-span element
        quarter_element_idx = n_elements // 4
        quarter_element = mesh.elements[quarter_element_idx]
        
        node_start_id = quarter_element.node_start.id
        node_end_id = quarter_element.node_end.id
        u_global = np.array([
            displacements[3*(node_start_id-1)],
            displacements[3*(node_start_id-1) + 1],
            displacements[3*(node_start_id-1) + 2],
            displacements[3*(node_end_id-1)],
            displacements[3*(node_end_id-1) + 1],
            displacements[3*(node_end_id-1) + 2]
        ])
        u_local = quarter_element.R.T @ u_global
        shear = quarter_element.shear_force(quarter_element.length / 2, u_local)
        quarter_shears.append(shear)
        
        print(f"{n_elements:<8} {moment:<25.2f} {shear:<25.2f}")
    
    # Verify convergence
    print("\nConvergence verification:")
    
    # Check moment convergence
    moment_converging = True
    for i in range(1, len(midspan_moments)-1):
        diff_current = abs(midspan_moments[i+1] - midspan_moments[i])
        diff_previous = abs(midspan_moments[i] - midspan_moments[i-1])
        if diff_current > diff_previous * 1.2:
            moment_converging = False
    
    # Check shear convergence
    shear_converging = True
    for i in range(1, len(quarter_shears)-1):
        diff_current = abs(quarter_shears[i+1] - quarter_shears[i])
        diff_previous = abs(quarter_shears[i] - quarter_shears[i-1])
        if diff_current > diff_previous * 1.2:
            shear_converging = False
    
    if moment_converging:
        print("  ✓ Bending moment converges with mesh refinement")
    else:
        print("  ✗ Bending moment does not converge properly")
    
    if shear_converging:
        print("  ✓ Shear force converges with mesh refinement")
    else:
        print("  ✗ Shear force does not converge properly")
    
    assert moment_converging and shear_converging, "Forces should converge with mesh refinement"
    
    print("\n✅ Forces convergence test passed")


def run_all_tests():
    """Run all complex structure convergence tests."""
    print("\n" + "="*70)
    print("COMPLEX STRUCTURES CONVERGENCE VERIFICATION TESTS")
    print("="*70)
    print("\nThese tests verify that the mesh correctly handles:")
    print("• Elements at various angles (not just horizontal)")
    print("• Multiple loads (point and distributed) simultaneously")
    print("• Complex boundary conditions")
    print("• Both Euler-Bernoulli and Timoshenko elements in complex scenarios")
    print("• Convergence of both displacements and forces")
    
    test_angled_cantilever_convergence()
    test_l_shaped_frame_convergence()
    test_multiple_loads_convergence()
    test_timoshenko_angled_beam_convergence()
    test_complex_structure_forces_convergence()
    
    print("\n" + "="*70)
    print("✅ ALL COMPLEX STRUCTURES TESTS PASSED!")
    print("="*70)
    print("\nConclusion:")
    print("• Mesh correctly handles elements at various angles")
    print("• Mesh correctly handles multiple loads simultaneously")
    print("• Mesh correctly handles complex boundary conditions")
    print("• Both Euler-Bernoulli and Timoshenko elements work correctly in complex scenarios")
    print("• Displacement and force solutions converge with mesh refinement")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_tests()
