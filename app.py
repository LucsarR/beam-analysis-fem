import streamlit as st
import numpy as np
import pandas as pd

from fem.mesh import Mesh
from fem.material import Material
from fem.section import create_section
from fem.constraint import Constraint
from fem.load import PointLoad, DistributedLoad
from fem.analysis import EulerBernoulliAnalysis
from config import DEFAULT_E, DEFAULT_NU, SECTION_TYPES, ELEMENT_TYPES

# --- Post-processing and Plotting ---
from post_processing.forces import StructureResults
from post_processing.plotter import plot_structure_diagram, plot_normal_stress_distribution

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

# --- Input: Properties (Material + Section) ---
st.header("Properties (Material + Section)")
properties = []
n_properties = st.number_input("Number of properties", min_value=1, max_value=10, value=1)
for i in range(n_properties):
    st.subheader(f"Property {i+1}")
    prop_name = st.text_input(f"Property {i+1} name", value=f"Property_{i+1}", key=f"propname_{i}")
    # Material
    E = st.number_input(f"Young's modulus E [{prop_name}]", value=DEFAULT_E, key=f"E_{i}")
    nu = st.number_input(f"Poisson's ratio ν [{prop_name}]", min_value=0.0, max_value=0.5, value=DEFAULT_NU, key=f"nu_{i}")
    material = Material(i+1, E, nu)
    # Section
    section_type = st.selectbox(f"Section type [{prop_name}]", SECTION_TYPES, index=0, key=f"sectype_{i}")
    section_kwargs = {}
    if section_type == "rectangular_bar":
        width = st.number_input(f"Width [{prop_name}]", value=0.05, key=f"width_{i}")
        height = st.number_input(f"Height [{prop_name}]", value=0.10, key=f"height_{i}")
        section_kwargs = dict(width=width, height=height)
    elif section_type == "rectangular_tube":
        width = st.number_input(f"Width [{prop_name}]", value=0.05, key=f"width_{i}")
        height = st.number_input(f"Height [{prop_name}]", value=0.10, key=f"height_{i}")
        thickness = st.number_input(f"Thickness [{prop_name}]", value=0.005, key=f"thick_{i}")
        section_kwargs = dict(width=width, height=height, thickness=thickness)
    elif section_type == "circular_bar":
        diameter = st.number_input(f"Diameter [{prop_name}]", value=0.05, key=f"diam_{i}")
        section_kwargs = dict(diameter=diameter)
    elif section_type == "circular_tube":
        outer_diameter = st.number_input(f"Outer diameter [{prop_name}]", value=0.05, key=f"odiam_{i}")
        thickness = st.number_input(f"Thickness [{prop_name}]", value=0.005, key=f"thick_{i}")
        section_kwargs = dict(outer_diameter=outer_diameter, thickness=thickness)
    elif section_type == "trapezoidal_bar":
        base1 = st.number_input(f"Base 1 [{prop_name}]", value=0.05, key=f"base1_{i}")
        base2 = st.number_input(f"Base 2 [{prop_name}]", value=0.10, key=f"base2_{i}")
        height = st.number_input(f"Height [{prop_name}]", value=0.10, key=f"height_{i}")
        section_kwargs = dict(base1=base1, base2=base2, height=height)
    elif section_type == "trapezoidal_tube":
        base1 = st.number_input(f"Base 1 [{prop_name}]", value=0.05, key=f"base1_{i}")
        base2 = st.number_input(f"Base 2 [{prop_name}]", value=0.10, key=f"base2_{i}")
        height = st.number_input(f"Height [{prop_name}]", value=0.10, key=f"height_{i}")
        thickness = st.number_input(f"Thickness [{prop_name}]", value=0.005, key=f"thick_{i}")
        section_kwargs = dict(base1=base1, base2=base2, height=height, thickness=thickness)
    elif section_type == "hexagonal_bar":
        side = st.number_input(f"Side [{prop_name}]", value=0.05, key=f"side_{i}")
        section_kwargs = dict(side=side)
    elif section_type == "hexagonal_tube":
        outer_side = st.number_input(f"Outer side [{prop_name}]", value=0.05, key=f"oside_{i}")
        thickness = st.number_input(f"Thickness [{prop_name}]", value=0.005, key=f"thick_{i}")
        section_kwargs = dict(outer_side=outer_side, thickness=thickness)
    elif section_type == "ibeam":
        h = st.number_input(f"Height h [{prop_name}]", value=0.10, key=f"h_{i}")
        b = st.number_input(f"Flange width b [{prop_name}]", value=0.05, key=f"b_{i}")
        tw = st.number_input(f"Web thickness tw [{prop_name}]", value=0.005, key=f"tw_{i}")
        tf = st.number_input(f"Flange thickness tf [{prop_name}]", value=0.005, key=f"tf_{i}")
        section_kwargs = dict(h=h, b=b, tw=tw, tf=tf)
    elif section_type == "c_section":
        h = st.number_input(f"Height h [{prop_name}]", value=0.10, key=f"h_{i}")
        b = st.number_input(f"Flange width b [{prop_name}]", value=0.05, key=f"b_{i}")
        tw = st.number_input(f"Web thickness tw [{prop_name}]", value=0.005, key=f"tw_{i}")
        tf = st.number_input(f"Flange thickness tf [{prop_name}]", value=0.005, key=f"tf_{i}")
        section_kwargs = dict(h=h, b=b, tw=tw, tf=tf)
    elif section_type == "l_section":
        b = st.number_input(f"Width b [{prop_name}]", value=0.05, key=f"b_{i}")
        h = st.number_input(f"Height h [{prop_name}]", value=0.10, key=f"h_{i}")
        t = st.number_input(f"Thickness t [{prop_name}]", value=0.005, key=f"t_{i}")
        section_kwargs = dict(b=b, h=h, t=t)
    elif section_type == "t_section":
        b = st.number_input(f"Flange width b [{prop_name}]", value=0.05, key=f"b_{i}")
        h = st.number_input(f"Height h [{prop_name}]", value=0.10, key=f"h_{i}")
        tw = st.number_input(f"Web thickness tw [{prop_name}]", value=0.005, key=f"tw_{i}")
        tf = st.number_input(f"Flange thickness tf [{prop_name}]", value=0.005, key=f"tf_{i}")
        section_kwargs = dict(b=b, h=h, tw=tw, tf=tf)
    elif section_type == "z_section":
        h = st.number_input(f"Height h [{prop_name}]", value=0.10, key=f"h_{i}")
        b = st.number_input(f"Flange width b [{prop_name}]", value=0.05, key=f"b_{i}")
        tw = st.number_input(f"Web thickness tw [{prop_name}]", value=0.005, key=f"tw_{i}")
        tf = st.number_input(f"Flange thickness tf [{prop_name}]", value=0.005, key=f"tf_{i}")
        section_kwargs = dict(h=h, b=b, tw=tw, tf=tf)
    elif section_type == "hat_section":
        h = st.number_input(f"Height h [{prop_name}]", value=0.10, key=f"h_{i}")
        b = st.number_input(f"Flange width b [{prop_name}]", value=0.05, key=f"b_{i}")
        tw = st.number_input(f"Web thickness tw [{prop_name}]", value=0.005, key=f"tw_{i}")
        tf = st.number_input(f"Flange thickness tf [{prop_name}]", value=0.005, key=f"tf_{i}")
        section_kwargs = dict(h=h, b=b, tw=tw, tf=tf)
    elif section_type == "general":
        area = st.number_input(f"Area [{prop_name}]", value=0.001, key=f"area_{i}")
        inertia = st.number_input(f"Inertia [{prop_name}]", value=1e-6, key=f"inertia_{i}")
        section_kwargs = dict(area=area, inertia=inertia)
    section = create_section(section_type, i+1, **section_kwargs)
    properties.append({"name": prop_name, "material": material, "section": section})

# --- Input: Elements ---
st.header("Elements")
elements = []
element_types = ELEMENT_TYPES
property_names = [prop["name"] for prop in properties]
n_elements = st.number_input("Number of elements", min_value=1, max_value=n_nodes-1, value=n_nodes-1)
for i in range(n_elements):
    col1, col2 = st.columns(2)
    n1 = int(col1.number_input(f"Start node (Elem {i+1})", min_value=1, max_value=n_nodes, value=i+1, key=f"en1_{i}"))
    n2 = int(col2.number_input(f"End node (Elem {i+1})", min_value=1, max_value=n_nodes, value=i+2, key=f"en2_{i}"))
    el_type = st.selectbox(f"Element type (Elem {i+1})", list(element_types.keys()), index=0, key=f"etype_{i}")
    prop_idx = st.selectbox(f"Property (Elem {i+1})", property_names, index=0, key=f"propidx_{i}")
    n_subdiv = st.number_input(f"Subdivisions (Elem {i+1})", min_value=1, max_value=20, value=1, key=f"subdiv_{i}")
    elements.append((n1, n2, element_types[el_type], prop_idx, n_subdiv))

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
n_dist_loads = st.number_input("Number of distributed loads", min_value=0, value=0)
for i in range(n_dist_loads):
    col1, col2, col3, col4, col5 = st.columns(5)
    element_id = int(col1.number_input(f"Distributed Load {i+1} element", min_value=1, value=1))
    load_type = col2.selectbox(f"Type {i+1}", options=["constant", "linear", "custom"], key=f"dltype_{i}")
    direction = col3.selectbox(f"Direction {i+1}", options=['x', 'y', 'l', 't'], key=f"ddir_{i}")
    if load_type == "constant":
        magnitude = col4.number_input(f"Value {i+1}", value=0.0, key=f"dlval_{i}")
        distributed_loads.append((element_id, magnitude, None, direction, None))
    elif load_type == "linear":
        magnitude_start = col4.number_input(f"Start {i+1}", value=0.0, key=f"dlstart_{i}")
        magnitude_end = col5.number_input(f"End {i+1}", value=0.0, key=f"dlend_{i}")
        distributed_loads.append((element_id, magnitude_start, magnitude_end, direction, None))
    elif load_type == "custom":
        func_str = col4.text_input(f"Function f(x) {i+1}", value="", key=f"dlfunc_{i}")
        # Validate function string
        error_msg = ""
        if func_str:
            try:
                import numpy as np
                x = 0.0
                L = 1.0
                test_val = eval(func_str, {"np": np, "x": x, "L": L})
            except Exception as e:
                error_msg = f"Invalid function: {e}"
                st.error(error_msg)
        distributed_loads.append((element_id, None, None, direction, func_str if not error_msg else None))

# --- Run Analysis ---
if st.button("Run Analysis"):
    mesh = Mesh()
    node_objs = []
    for x, y in nodes:
        node_objs.append(mesh.add_node(x, y))
    # Add elements with selected property
    for n1, n2, etype, prop_name, n_subdiv in elements:
        prop = next(p for p in properties if p["name"] == prop_name)
        node_start = mesh.get_node_by_id(n1)
        node_end = mesh.get_node_by_id(n2)
        if n_subdiv == 1:
            mesh.add_element(
                node_start,
                node_end,
                prop["material"],
                prop["section"],
                element_type=etype
            )
        else:
            # Subdivide element, reusing start/end nodes
            x_start, y_start = node_start.x, node_start.y
            x_end, y_end = node_end.x, node_end.y
            subdiv_nodes = [node_start]
            for i in range(1, n_subdiv):
                x = x_start + (x_end - x_start) * i / n_subdiv
                y = y_start + (y_end - y_start) * i / n_subdiv
                # Check if node already exists at (x, y)
                existing = next((n for n in mesh.nodes if np.isclose(n.x, x) and np.isclose(n.y, y)), None)
                if existing:
                    subdiv_nodes.append(existing)
                else:
                    subdiv_nodes.append(mesh.add_node(x, y))
            subdiv_nodes.append(node_end)
            for i in range(n_subdiv):
                mesh.add_element(
                    subdiv_nodes[i],
                    subdiv_nodes[i+1],
                    prop["material"],
                    prop["section"],
                    element_type=etype
                )
    # Add constraints
    for node_id, direction, value in constraints:
        mesh.constraints.add(Constraint(mesh.get_node_by_id(node_id), direction, value))
    # Add point loads
    for node_id, direction, magnitude in point_loads:
        mesh.point_loads.append(PointLoad(magnitude, direction))
        mesh.point_loads[-1].node = mesh.get_node_by_id(node_id)
    # Add distributed loads
    for element_id, magnitude_start, magnitude_end, direction, func_str in distributed_loads:
        el = mesh.get_element_by_id(element_id)
        load = DistributedLoad(magnitude_start, magnitude_end, direction, func=func_str)
        load.element = el
        mesh.distributed_loads.append(load)
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

    # Store results in session state
    st.session_state["mesh"] = mesh
    st.session_state["displacements"] = displacements

    # --- Post-processing and Plotting ---
    structure_results = StructureResults(mesh, displacements)
    st.session_state["structure_results"] = structure_results

# Use session state for post-processing and plotting
if "structure_results" in st.session_state:
    structure_results = st.session_state["structure_results"]

    st.subheader("Diagrams")
    diagram_type = st.selectbox(
        "Select diagram to show",
        options=["Moment", "Shear", "Normal Force"],
        index=0
    )
    diagram_map = {
        "Moment": "moment",
        "Shear": "shear",
        "Normal Force": "normal"
    }
    fill_diagram = st.checkbox("Show force diagram as filled area above element", value=False)
    if fill_diagram:
        diagram_scale = st.slider("Force diagram scale", min_value=0.05, max_value=0.5, value=0.2)
    else:
        diagram_scale = 0.2  # Default value, not used unless fill_diagram is True

    if st.button("Show Diagram"):
        fig = plot_structure_diagram(
            structure_results,
            force_type=diagram_map[diagram_type],
            fill_diagram=fill_diagram,
            scale=diagram_scale
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Normal Stress Distribution in Cross Section")
    element_ids = [el.id for el in st.session_state["mesh"].elements]
    if element_ids:
        selected_element_id = st.selectbox("Select element", element_ids)
        selected_element_result = next(er for er in structure_results.element_results if er.element.id == selected_element_id)
        x_pos = st.slider(
            "Position ALONG element",
            min_value=0.0,
            max_value=float(selected_element_result.length),
            value=0.0,
            step=0.01,
        )
        if st.button("Show Normal Stress Distribution"):
            try:
                fig = plot_normal_stress_distribution(selected_element_result, x_pos)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")


# TODO: Create a more user friendly app interface with better layout and validation, create a way to save/load projects.