import json
import numpy as np
from fem.mesh import Mesh
from fem.material import Material
from fem.section import RectangularBar
from fem.constraint import Constraint
from fem.load import PointLoad, DistributedLoad
from fem.analysis import BeamAnalysis

# Load from projects/reddy_test_distributed.json
with open("projects/reddy_test_distributed.json", "r") as f:
    project_data = json.load(f)

nodes = project_data["nodes"]
elements = project_data["elements"]
properties = project_data["properties"]
constraints = project_data["constraints"]
point_loads = project_data["point_loads"]
distributed_loads = project_data["distributed_loads"]

mesh = Mesh()
node_objs = []

# Add nodes
for x, y in nodes:
    node_objs.append(mesh.add_node(x, y))

# Add elements with selected property
original_to_mesh_elements = {}
for orig_idx, (n1, n2, etype, prop_name, n_subdiv) in enumerate(elements, start=1):
    prop = next(p for p in properties if p["name"] == prop_name)
    node_start = mesh.get_node_by_id(n1)
    node_end = mesh.get_node_by_id(n2)
    
    # Material
    mat_data = prop["material"]
    mat = Material(1, mat_data["E"], mat_data["nu"])
    
    # Section
    sec_data = prop["section"]
    sec = RectangularBar(1, sec_data["kwargs"]["width"], sec_data["kwargs"]["height"])
    
    if n_subdiv == 1:
        el = mesh.add_element(
            node_start,
            node_end,
            mat,
            sec,
            element_type=etype,
            stiffness_integration="analytical",
        )
        original_to_mesh_elements[orig_idx] = [el.id]
    else:
        # Subdivide
        x_start, y_start = node_start.x, node_start.y
        x_end, y_end = node_end.x, node_end.y
        subdiv_nodes = [node_start]
        for i in range(1, n_subdiv):
            x = x_start + (x_end - x_start) * i / n_subdiv
            y = y_start + (y_end - y_start) * i / n_subdiv
            existing = next((n for n in mesh.nodes if np.isclose(n.x, x) and np.isclose(n.y, y)), None)
            if existing:
                subdiv_nodes.append(existing)
            else:
                subdiv_nodes.append(mesh.add_node(x, y))
        subdiv_nodes.append(node_end)
        
        subdiv_ids = []
        for i in range(n_subdiv):
            el = mesh.add_element(
                subdiv_nodes[i],
                subdiv_nodes[i+1],
                mat,
                sec,
                element_type=etype,
                stiffness_integration="analytical",
            )
            subdiv_ids.append(el.id)
        original_to_mesh_elements[orig_idx] = subdiv_ids

# Add constraints
for node_id, direction, value in constraints:
    mesh.constraints.add(Constraint(mesh.get_node_by_id(node_id), direction, value))

# Add point loads
for node_id, direction, magnitude in point_loads:
    mesh.point_loads.append(PointLoad(magnitude, direction))
    mesh.point_loads[-1].node = mesh.get_node_by_id(node_id)

# Add distributed loads
for element_id, magnitude_start, magnitude_end, direction, func_str, _ in distributed_loads:
    mesh_el_ids = original_to_mesh_elements.get(element_id)
    orig_elem = elements[element_id - 1]
    orig_n1, orig_n2 = orig_elem[0], orig_elem[1]
    node_orig_start = mesh.get_node_by_id(orig_n1)
    node_orig_end = mesh.get_node_by_id(orig_n2)
    L_orig = np.hypot(node_orig_end.x - node_orig_start.x, node_orig_end.y - node_orig_start.y)
    
    for mesh_el_id in mesh_el_ids:
        el = mesh.get_element_by_id(mesh_el_id)
        if magnitude_start is not None:
            load = DistributedLoad(float(magnitude_start), float(magnitude_start), direction, func=None)
        else:
            load = DistributedLoad(None, None, direction, func=None)
        load.element = el
        mesh.distributed_loads.append(load)

# Run analysis
analysis = BeamAnalysis(mesh, structural_behavior="frame")
analysis.assemble()
displacements = analysis.solve()

print("Solved Node Displacements:")
for i, node in enumerate(mesh.nodes):
    disp = displacements[i*4 : (i+1)*4]
    print(f"Node {node.id} at ({node.x:.3f}, {node.y:.3f}): u={disp[0]:.6f}, v={disp[1]:.6f}, theta={disp[2]:.6f}, dv/dx={disp[3]:.6f}")
