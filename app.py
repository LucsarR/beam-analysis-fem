import streamlit as st
import numpy as np
import pandas as pd

from fem.mesh import Mesh
from fem.material import Material
from fem.section import create_section
from fem.constraint import Constraint
from fem.load import PointLoad, DistributedLoad, MomentLoad
from fem.analysis import EulerBernoulliAnalysis

st.title("FEM Beam Analysis Tool")

# --- Input: Nodes ---
st.header("Nodes")
nodes = []
n_nodes = st.number_input("Number of nodes", min_value=2, max_value=20, value=2)
for i in range(n_nodes):
    col1, col2 = st.columns(2)
    x = col1.number_input(f"Node {i+1} x", value=float(i))
    y = col2.number_input(f"Node {i+1} y", value=0.0)
    nodes.append((x, y))

# --- Input: Material ---
st.header("Material")
E = st.number_input("Young's modulus E", value=70e3)
nu = st.number_input("Poisson's ratio ν", min_value=0.0, max_value=0.5, value=0.3)
material = Material(1, E, nu)

# --- Input: Section ---
st.header("Section")
section_types = [
    "rectangular_bar", "rectangular_tube", "circular_bar", "circular_tube",
    "trapezoidal_bar", "trapezoidal_tube", "hexagonal_bar", "hexagonal_tube",
    "ibeam", "c_section", "l_section", "t_section", "z_section", "hat_section", "general"
]
section_type = st.selectbox("Section type", section_types, index=0)

section_kwargs = {}
if section_type == "rectangular_bar":
    width = st.number_input("Width", value=0.05)
    height = st.number_input("Height", value=0.10)
    section_kwargs = dict(width=width, height=height)
elif section_type == "rectangular_tube":
    width = st.number_input("Width", value=0.05)
    height = st.number_input("Height", value=0.10)
    thickness = st.number_input("Thickness", value=0.005)
    section_kwargs = dict(width=width, height=height, thickness=thickness)
elif section_type == "circular_bar":
    diameter = st.number_input("Diameter", value=0.05)
    section_kwargs = dict(diameter=diameter)
elif section_type == "circular_tube":
    outer_diameter = st.number_input("Outer diameter", value=0.05)
    thickness = st.number_input("Thickness", value=0.005)
    section_kwargs = dict(outer_diameter=outer_diameter, thickness=thickness)
elif section_type == "trapezoidal_bar":
    base1 = st.number_input("Base 1", value=0.05)
    base2 = st.number_input("Base 2", value=0.10)
    height = st.number_input("Height", value=0.10)
    section_kwargs = dict(base1=base1, base2=base2, height=height)
elif section_type == "trapezoidal_tube":
    base1 = st.number_input("Base 1", value=0.05)
    base2 = st.number_input("Base 2", value=0.10)
    height = st.number_input("Height", value=0.10)
    thickness = st.number_input("Thickness", value=0.005)
    section_kwargs = dict(base1=base1, base2=base2, height=height, thickness=thickness)
elif section_type == "hexagonal_bar":
    side = st.number_input("Side", value=0.05)
    section_kwargs = dict(side=side)
elif section_type == "hexagonal_tube":
    outer_side = st.number_input("Outer side", value=0.05)
    thickness = st.number_input("Thickness", value=0.005)
    section_kwargs = dict(outer_side=outer_side, thickness=thickness)
elif section_type == "ibeam":
    h = st.number_input("Height h", value=0.10)
    b = st.number_input("Flange width b", value=0.05)
    tw = st.number_input("Web thickness tw", value=0.005)
    tf = st.number_input("Flange thickness tf", value=0.005)
    section_kwargs = dict(h=h, b=b, tw=tw, tf=tf)
elif section_type == "c_section":
    h = st.number_input("Height h", value=0.10)
    b = st.number_input("Flange width b", value=0.05)
    tw = st.number_input("Web thickness tw", value=0.005)
    tf = st.number_input("Flange thickness tf", value=0.005)
    section_kwargs = dict(h=h, b=b, tw=tw, tf=tf)
elif section_type == "l_section":
    b = st.number_input("Width b", value=0.05)
    h = st.number_input("Height h", value=0.10)
    t = st.number_input("Thickness t", value=0.005)
    section_kwargs = dict(b=b, h=h, t=t)
elif section_type == "t_section":
    b = st.number_input("Flange width b", value=0.05)
    h = st.number_input("Height h", value=0.10)
    tw = st.number_input("Web thickness tw", value=0.005)
    tf = st.number_input("Flange thickness tf", value=0.005)
    section_kwargs = dict(b=b, h=h, tw=tw, tf=tf)
elif section_type == "z_section":
    h = st.number_input("Height h", value=0.10)
    b = st.number_input("Flange width b", value=0.05)
    tw = st.number_input("Web thickness tw", value=0.005)
    tf = st.number_input("Flange thickness tf", value=0.005)
    section_kwargs = dict(h=h, b=b, tw=tw, tf=tf)
elif section_type == "hat_section":
    h = st.number_input("Height h", value=0.10)
    b = st.number_input("Flange width b", value=0.05)
    tw = st.number_input("Web thickness tw", value=0.005)
    tf = st.number_input("Flange thickness tf", value=0.005)
    section_kwargs = dict(h=h, b=b, tw=tw, tf=tf)
elif section_type == "general":
    area = st.number_input("Area", value=0.001)
    inertia = st.number_input("Inertia", value=1e-6)
    section_kwargs = dict(area=area, inertia=inertia)

section = create_section(section_type, 1, **section_kwargs)

# --- Input: Elements ---
st.header("Elements")
elements = []
element_types = {
    "Euler-Bernoulli 2-node": "euler_bernoulli_2node",
    "Euler-Bernoulli 3-node": "euler_bernoulli_3node",
    "Timoshenko 2-node": "timoshenko_2node"
}
n_elements = st.number_input("Number of elements", min_value=1, max_value=n_nodes-1, value=n_nodes-1)
for i in range(n_elements):
    col1, col2, col3 = st.columns(3)
    n1 = int(col1.number_input(f"Element {i+1} start node", min_value=1, max_value=n_nodes, value=i+1))
    n2 = int(col2.number_input(f"Element {i+1} end node", min_value=1, max_value=n_nodes, value=i+2))
    el_type = col3.selectbox(f"Element {i+1} type", list(element_types.keys()), index=0, key=f"etype_{i}")
    elements.append((n1, n2, element_types[el_type]))

# --- Input: Constraints ---
st.header("Constraints")
constraints = []
n_constraints = st.number_input("Number of constraints", min_value=0, max_value=n_nodes*3, value=0)
for i in range(n_constraints):
    col1, col2, col3 = st.columns(3)
    node_id = int(col1.number_input(f"Constraint {i+1} node", min_value=1, max_value=n_nodes, value=1))
    direction = int(col2.selectbox(f"Constraint {i+1} direction", options=[0,1,2], format_func=lambda x: ["x","y","rotation"][x], key=f"cdir_{i}"))
    value = col3.number_input(f"Constraint {i+1} value", value=0.0)
    constraints.append((node_id, direction, value))

# --- Input: Point Loads ---
st.header("Point Loads")
point_loads = []
n_loads = st.number_input("Number of point loads", min_value=0, max_value=n_nodes*3, value=0)
for i in range(n_loads):
    col1, col2, col3 = st.columns(3)
    node_id = int(col1.number_input(f"Load {i+1} node", min_value=1, max_value=n_nodes, value=1))
    direction = int(col2.selectbox(f"Load {i+1} direction", options=[0,1,2], format_func=lambda x: ["x","y","moment"][x], key=f"ldir_{i}"))
    magnitude = col3.number_input(f"Load {i+1} magnitude", value=0.0)
    point_loads.append((node_id, direction, magnitude))

# --- Input: Distributed Loads ---
st.header("Distributed Loads")
distributed_loads = []
n_dist_loads = st.number_input("Number of distributed loads", min_value=0, max_value=n_elements, value=0)
for i in range(n_dist_loads):
    col1, col2, col3, col4 = st.columns(4)
    element_id = int(col1.number_input(f"Distributed Load {i+1} element", min_value=1, max_value=n_elements, value=1))
    magnitude_start = col2.number_input(f"q_ini {i+1}", value=0.0)
    magnitude_end = col3.number_input(f"q_fim {i+1}", value=0.0)
    direction = col4.selectbox(f"Direction {i+1}", options=['x', 'y', 'l', 't'], key=f"ddir_{i}")
    distributed_loads.append((element_id, magnitude_start, magnitude_end, direction))

# --- Input: Moment Loads ---
st.header("Moment Loads")
moment_loads = []
n_moment_loads = st.number_input("Number of moment loads", min_value=0, max_value=n_nodes, value=0)
for i in range(n_moment_loads):
    col1, col2 = st.columns(2)
    node_id = int(col1.number_input(f"Moment Load {i+1} node", min_value=1, max_value=n_nodes, value=1))
    magnitude = col2.number_input(f"Moment {i+1} magnitude", value=0.0)
    moment_loads.append((node_id, magnitude))

# --- Run Analysis ---
if st.button("Run Analysis"):
    mesh = Mesh()
    node_objs = []
    for x, y in nodes:
        node_objs.append(mesh.add_node(x, y))
    for n1, n2, etype in elements:
        mesh.add_element(mesh.get_node_by_id(n1), mesh.get_node_by_id(n2), material, section, element_type=etype)
    # Add constraints
    for node_id, direction, value in constraints:
        mesh.constraints.add(Constraint(mesh.get_node_by_id(node_id), direction, value))
    # Add point loads
    for node_id, direction, magnitude in point_loads:
        mesh.point_loads.append(PointLoad(magnitude, direction))
        mesh.point_loads[-1].node = mesh.get_node_by_id(node_id)
    # Add distributed loads
    for element_id, magnitude_start, magnitude_end, direction in distributed_loads:
        el = mesh.get_element_by_id(element_id)
        load = DistributedLoad(magnitude_start, magnitude_end, direction)
        load.element = el
        mesh.distributed_loads.append(load)
    # Add moment loads
    for node_id, magnitude in moment_loads:
        load = MomentLoad(magnitude)
        load.node = mesh.get_node_by_id(node_id)
        mesh.point_loads.append(load)
    # Run analysis (only Euler-Bernoulli supported for now)
    analysis = EulerBernoulliAnalysis(mesh)
    analysis.assemble()
    displacements = analysis.solve()
    # Prepare results
    disp_data = []
    for i, node in enumerate(mesh.nodes):
        u = displacements[3*i]
        v = displacements[3*i+1]
        theta = displacements[3*i+2]
        disp_data.append({"Node": node.id, "x": node.x, "y": node.y, "u": u, "v": v, "theta": theta})
    df_disp = pd.DataFrame(disp_data)
    st.subheader("Nodal Displacements")
    st.dataframe(df_disp)
    # Export CSV
    csv = df_disp.to_csv(index=False).encode('utf-8')
    st.download_button("Download Displacements CSV", csv, "displacements.csv", "text/csv")