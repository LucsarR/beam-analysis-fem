import streamlit as st
import numpy as np
import pandas as pd
import json
import traceback
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sympy as _sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

from fem.mesh import Mesh
from fem.material import Material
from fem.section import create_section
from fem.constraint import Constraint
from fem.load import PointLoad, DistributedLoad
from fem.analysis import EulerBernoulliAnalysis
from config import DEFAULT_E, DEFAULT_NU, DEFAULT_G, SECTION_TYPES, ELEMENT_TYPES

# --- Post-processing and Plotting ---
from post_processing.forces import StructureResults
from post_processing.plotter import (
    plot_structure_diagram,
    plot_normal_stress_distribution,
    plot_normal_stress_side_view,
    plot_structure_preview,
    find_position_on_structure,
)


def _func_str_to_latex(func_str: str) -> str:
    """Convert a Python math expression string (using numpy syntax) to a LaTeX string.

    Uses sympy for accurate conversion.
    """
    if not func_str:
        return ""

    try:
        # Replace numpy prefixed names with bare names that sympy understands
        clean = func_str.replace("np.", "")
        x, L = _sp.symbols("x L", positive=True)
        transformations = standard_transformations + (implicit_multiplication_application,)
        expr = parse_expr(clean, local_dict={"x": x, "L": L}, transformations=transformations)
        return _sp.latex(expr)
    except Exception:
        return func_str


# --- Page Configuration ---
st.set_page_config(
    page_title="FEM Beam Analysis Tool",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔧 FEM Beam Analysis Tool")
st.markdown("_A finite element analysis tool for beam structures_")

# --- Hide number-input step buttons for float/coordinate fields where arbitrary values
#     are entered, but keep +/- controls for integer fields (step=1). ---
st.markdown("""
<style>
div[data-testid="stNumberInput"] button {
    display: none !important;
}
div[data-testid="stNumberInput"]:has(input[step="1"]) button {
    display: flex !important;
}
</style>
""", unsafe_allow_html=True)

# --- Helper Functions for Project Management ---
def save_project_to_dict():
    """Save current project state to dictionary."""
    project_data = {
        "metadata": {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "description": st.session_state.get("project_description", "")
        },
        "nodes": st.session_state.get("nodes", []),
        "properties": [],
        "elements": st.session_state.get("elements", []),
        "constraints": st.session_state.get("constraints", []),
        "point_loads": st.session_state.get("point_loads", []),
        "distributed_loads": st.session_state.get("distributed_loads", [])
    }
    
    # Convert properties to serializable format
    for prop in st.session_state.get("properties", []):
        prop_dict = {
            "name": prop.get("name", ""),
            "material": {
                "E": prop["material"].E if "material" in prop else DEFAULT_E,
                "G": prop["material"].G if "material" in prop else DEFAULT_G,
                "nu": prop["material"].nu if "material" in prop else DEFAULT_NU,
                "input_mode": prop.get("mat_input_mode", "Calculate G (from E and ν)")
            },
            "section": {
                "type": prop.get("section_type", "general"),
                "kwargs": prop.get("section_kwargs", {})
            }
        }
        project_data["properties"].append(prop_dict)
    
    return project_data

def load_project_from_dict(project_data):
    """Load project state from dictionary."""
    try:
        # Clear widget state keys so widgets re-initialize from the loaded values.
        # When a widget has an explicit key, Streamlit stores its value in
        # st.session_state[key] and ignores the 'value' parameter on subsequent
        # reruns.  Removing those stale entries forces Streamlit to respect the
        # 'value' argument computed from the freshly loaded project data.
        persistent_keys = {
            "nodes", "properties", "elements", "constraints", "point_loads",
            "distributed_loads", "project_description", "file_loader_key",
        }
        for k in list(st.session_state.keys()):
            if k not in persistent_keys:
                del st.session_state[k]

        st.session_state["nodes"] = project_data.get("nodes", [])
        st.session_state["elements"] = project_data.get("elements", [])
        st.session_state["constraints"] = project_data.get("constraints", [])
        st.session_state["point_loads"] = project_data.get("point_loads", [])
        st.session_state["distributed_loads"] = project_data.get("distributed_loads", [])
        st.session_state["project_description"] = project_data.get("metadata", {}).get("description", "")
        
        # Reconstruct properties
        properties = []
        for prop_data in project_data.get("properties", []):
            mat_data = prop_data["material"]
            mat_mode = mat_data.get("input_mode", "Calculate G (from E and ν)")
            if mat_mode == "Calculate ν (from E and G)":
                material = Material(
                    len(properties) + 1,
                    E=mat_data["E"],
                    G=mat_data.get("G", DEFAULT_G),
                )
            elif mat_mode == "Calculate E (from G and ν)":
                material = Material(
                    len(properties) + 1,
                    G=mat_data.get("G", DEFAULT_G),
                    nu=mat_data["nu"],
                )
            else:
                # Default: Calculate G from E and ν (also covers old files without input_mode)
                material = Material(
                    len(properties) + 1,
                    E=mat_data["E"],
                    nu=mat_data["nu"],
                )
            section_data = prop_data["section"]
            section = create_section(
                section_data["type"],
                len(properties) + 1,
                **section_data["kwargs"]
            )
            properties.append({
                "name": prop_data["name"],
                "material": material,
                "mat_input_mode": mat_mode,
                "section": section,
                "section_type": section_data["type"],
                "section_kwargs": section_data["kwargs"]
            })
        st.session_state["properties"] = properties
        
        return True
    except Exception as e:
        st.error(f"Error loading project: {e}")
        return False

def validate_number(value, min_val=None, max_val=None, field_name="Value"):
    """Validate numeric input."""
    if value is None:
        return False, f"{field_name} is required."
    if min_val is not None and value < min_val:
        return False, f"{field_name} must be at least {min_val}."
    if max_val is not None and value > max_val:
        return False, f"{field_name} must be at most {max_val}."
    return True, ""

def create_section_preview(section_type, **kwargs):
    """
    Create a matplotlib figure with a schematic diagram of the section type,
    showing the dimensions with labels to help users understand the measurements.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect('equal')
    ax.axis('off')

    def add_dimension(x1, y1, x2, y2, label, scale, shape_ref=None):
        """
        Add a scale-proportional dimension line with double-headed arrows,
        extension lines from the shape edge, and a label.

        For horizontal lines (|dx| > |dy|): shape_ref is the y-coordinate of the
        nearest shape edge; for vertical lines: shape_ref is the x-coordinate.
        """
        text_gap = scale * 0.10
        ext_overshoot = scale * 0.04  # extension line protrusion past the dim line

        dx = x2 - x1
        dy = y2 - y1

        # Dimension line with double-headed arrows
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='<->', color='dimgray',
                                   lw=1, mutation_scale=8))

        if abs(dx) >= abs(dy):  # Horizontal dimension
            if shape_ref is not None:
                sign = 1 if y1 > shape_ref else -1
                ax.plot([x1, x1], [shape_ref, y1 + sign * ext_overshoot],
                        color='dimgray', linewidth=0.8)
                ax.plot([x2, x2], [shape_ref, y1 + sign * ext_overshoot],
                        color='dimgray', linewidth=0.8)
            # Label below (or above) the dimension line
            label_sign = -1 if shape_ref is None or y1 <= shape_ref else 1
            va = 'top' if label_sign < 0 else 'bottom'
            ax.text((x1 + x2) / 2, y1 + label_sign * text_gap, label,
                    ha='center', va=va, fontsize=10, color='royalblue')
        else:  # Vertical dimension
            if shape_ref is not None:
                sign = 1 if x1 > shape_ref else -1
                ax.plot([shape_ref, x1 + sign * ext_overshoot], [y1, y1],
                        color='dimgray', linewidth=0.8)
                ax.plot([shape_ref, x1 + sign * ext_overshoot], [y2, y2],
                        color='dimgray', linewidth=0.8)
            # Label to the left (or right) of the dimension line
            label_sign = -1 if shape_ref is None or x1 <= shape_ref else 1
            ha = 'right' if label_sign < 0 else 'left'
            ax.text(x1 + label_sign * text_gap, (y1 + y2) / 2, label,
                    ha=ha, va='center', fontsize=10, color='royalblue')

    if section_type == "rectangular_bar":
        width = kwargs.get("width", 0.05)
        height = kwargs.get("height", 0.10)
        scale = max(width, height)
        dim_off = scale * 0.30

        rect = mpatches.Rectangle((-width/2, -height/2), width, height,
                                   linewidth=2, edgecolor='black', facecolor='lightgray')
        ax.add_patch(rect)

        add_dimension(-width/2, -height/2 - dim_off, width/2, -height/2 - dim_off,
                      'width', scale, shape_ref=-height/2)
        add_dimension(-width/2 - dim_off, -height/2, -width/2 - dim_off, height/2,
                      'height', scale, shape_ref=-width/2)

        margin = scale * 0.60
        ax.set_xlim(-width/2 - margin, width/2 + margin)
        ax.set_ylim(-height/2 - margin, height/2 + margin)

    elif section_type == "rectangular_tube":
        width = kwargs.get("width", 0.05)
        height = kwargs.get("height", 0.10)
        thickness = kwargs.get("thickness", 0.005)
        scale = max(width, height)
        dim_off = scale * 0.30

        outer_rect = mpatches.Rectangle((-width/2, -height/2), width, height,
                                         linewidth=2, edgecolor='black', facecolor='lightgray')
        ax.add_patch(outer_rect)
        inner_rect = mpatches.Rectangle((-width/2 + thickness, -height/2 + thickness),
                                         width - 2*thickness, height - 2*thickness,
                                         linewidth=1, edgecolor='black', facecolor='white')
        ax.add_patch(inner_rect)

        add_dimension(-width/2, -height/2 - dim_off, width/2, -height/2 - dim_off,
                      'width', scale, shape_ref=-height/2)
        add_dimension(-width/2 - dim_off, -height/2, -width/2 - dim_off, height/2,
                      'height', scale, shape_ref=-width/2)
        # Thickness indicator on top edge
        t_off = scale * 0.15
        add_dimension(-width/2, height/2 + t_off, -width/2 + thickness, height/2 + t_off,
                      't', scale, shape_ref=height/2)

        margin = scale * 0.60
        ax.set_xlim(-width/2 - margin, width/2 + margin)
        ax.set_ylim(-height/2 - margin, height/2 + margin)

    elif section_type == "circular_bar":
        diameter = kwargs.get("diameter", 0.05)
        radius = diameter / 2
        scale = diameter
        dim_off = scale * 0.30

        circle = mpatches.Circle((0, 0), radius, linewidth=2,
                                  edgecolor='black', facecolor='lightgray')
        ax.add_patch(circle)
        ax.plot([-radius, radius], [0, 0], 'b--', linewidth=1)

        add_dimension(-radius, -radius - dim_off, radius, -radius - dim_off,
                      'diameter', scale, shape_ref=-radius)

        margin = scale * 0.60
        ax.set_xlim(-radius - margin, radius + margin)
        ax.set_ylim(-radius - margin, radius + margin)

    elif section_type == "circular_tube":
        outer_diameter = kwargs.get("outer_diameter", 0.05)
        thickness = kwargs.get("thickness", 0.005)
        outer_radius = outer_diameter / 2
        inner_radius = outer_radius - thickness
        scale = outer_diameter
        dim_off = scale * 0.30

        outer_circle = mpatches.Circle((0, 0), outer_radius, linewidth=2,
                                        edgecolor='black', facecolor='lightgray')
        ax.add_patch(outer_circle)
        inner_circle = mpatches.Circle((0, 0), inner_radius, linewidth=1,
                                        edgecolor='black', facecolor='white')
        ax.add_patch(inner_circle)
        ax.plot([-outer_radius, outer_radius], [0, 0], 'b--', linewidth=1)

        add_dimension(-outer_radius, -outer_radius - dim_off,
                      outer_radius, -outer_radius - dim_off,
                      'outer_diameter', scale, shape_ref=-outer_radius)
        # Thickness: horizontal from inner edge to outer edge on the right, at y = small offset
        t_off = scale * 0.10
        add_dimension(inner_radius, t_off, outer_radius, t_off,
                      't', scale, shape_ref=0)

        margin = scale * 0.60
        ax.set_xlim(-outer_radius - margin, outer_radius + margin)
        ax.set_ylim(-outer_radius - margin, outer_radius + margin)

    elif section_type == "trapezoidal_bar":
        base1 = kwargs.get("base1", 0.05)
        base2 = kwargs.get("base2", 0.10)
        height = kwargs.get("height", 0.10)
        max_base = max(base1, base2)
        scale = max(max_base, height)
        dim_off = scale * 0.30

        points = [[-base1/2, -height/2], [base1/2, -height/2],
                  [base2/2, height/2], [-base2/2, height/2]]
        trapezoid = mpatches.Polygon(points, closed=True, linewidth=2,
                                      edgecolor='black', facecolor='lightgray')
        ax.add_patch(trapezoid)

        add_dimension(-base1/2, -height/2 - dim_off, base1/2, -height/2 - dim_off,
                      'base1', scale, shape_ref=-height/2)
        add_dimension(-base2/2, height/2 + dim_off, base2/2, height/2 + dim_off,
                      'base2', scale, shape_ref=height/2)
        add_dimension(-max_base/2 - dim_off, -height/2, -max_base/2 - dim_off, height/2,
                      'height', scale, shape_ref=-max_base/2)

        margin = scale * 0.60
        ax.set_xlim(-max_base/2 - margin, max_base/2 + margin)
        ax.set_ylim(-height/2 - margin, height/2 + margin)

    elif section_type == "trapezoidal_tube":
        base1 = kwargs.get("base1", 0.05)
        base2 = kwargs.get("base2", 0.10)
        height = kwargs.get("height", 0.10)
        thickness = kwargs.get("thickness", 0.005)
        max_base = max(base1, base2)
        scale = max(max_base, height)
        dim_off = scale * 0.30

        outer_points = [[-base1/2, -height/2], [base1/2, -height/2],
                        [base2/2, height/2], [-base2/2, height/2]]
        outer_trapezoid = mpatches.Polygon(outer_points, closed=True, linewidth=2,
                                            edgecolor='black', facecolor='lightgray')
        ax.add_patch(outer_trapezoid)
        inner_base1 = max(base1 - 2*thickness, 0.001)
        inner_base2 = max(base2 - 2*thickness, 0.001)
        inner_height = max(height - 2*thickness, 0.001)
        inner_points = [[-inner_base1/2, -inner_height/2], [inner_base1/2, -inner_height/2],
                        [inner_base2/2, inner_height/2], [-inner_base2/2, inner_height/2]]
        inner_trapezoid = mpatches.Polygon(inner_points, closed=True, linewidth=1,
                                            edgecolor='black', facecolor='white')
        ax.add_patch(inner_trapezoid)

        add_dimension(-base1/2, -height/2 - dim_off, base1/2, -height/2 - dim_off,
                      'base1', scale, shape_ref=-height/2)
        add_dimension(-base2/2, height/2 + dim_off, base2/2, height/2 + dim_off,
                      'base2', scale, shape_ref=height/2)
        add_dimension(-max_base/2 - dim_off, -height/2, -max_base/2 - dim_off, height/2,
                      'height', scale, shape_ref=-max_base/2)

        margin = scale * 0.60
        ax.set_xlim(-max_base/2 - margin, max_base/2 + margin)
        ax.set_ylim(-height/2 - margin, height/2 + margin)

    elif section_type == "ibeam":
        h = kwargs.get("h", 0.10)
        b = kwargs.get("b", 0.05)
        tw = kwargs.get("tw", 0.005)
        tf = kwargs.get("tf", 0.005)
        scale = max(h, b)
        dim_off = scale * 0.30

        bottom_flange = mpatches.Rectangle((-b/2, -h/2), b, tf,
                                            linewidth=1, edgecolor='black', facecolor='lightgray')
        ax.add_patch(bottom_flange)
        web = mpatches.Rectangle((-tw/2, -h/2 + tf), tw, h - 2*tf,
                                  linewidth=1, edgecolor='black', facecolor='lightgray')
        ax.add_patch(web)
        top_flange = mpatches.Rectangle((-b/2, h/2 - tf), b, tf,
                                         linewidth=1, edgecolor='black', facecolor='lightgray')
        ax.add_patch(top_flange)

        add_dimension(-b/2, -h/2 - dim_off, b/2, -h/2 - dim_off,
                      'b', scale, shape_ref=-h/2)
        add_dimension(-b/2 - dim_off, -h/2, -b/2 - dim_off, h/2,
                      'h', scale, shape_ref=-b/2)
        # Web thickness above top flange
        tw_off = scale * 0.15
        add_dimension(-tw/2, h/2 + tw_off, tw/2, h/2 + tw_off,
                      'tw', scale, shape_ref=h/2)
        # Flange thickness on right side
        add_dimension(b/2 + tw_off, h/2 - tf, b/2 + tw_off, h/2,
                      'tf', scale, shape_ref=b/2)

        margin = scale * 0.60
        ax.set_xlim(-b/2 - margin, b/2 + margin)
        ax.set_ylim(-h/2 - margin, h/2 + margin)

    elif section_type in ["c_section", "l_section", "t_section", "z_section", "hat_section", "hexagonal_bar", "hexagonal_tube"]:
        # For complex sections, show a simplified diagram
        ax.text(0, 0, f"{section_type.replace('_', ' ').title()}\n\nSee parameters\nfor dimensions",
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)

    elif section_type == "general":
        # For general section, just show a placeholder
        ax.text(0, 0, "General Section\n\nArea and Inertia\ndefined manually",
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)

    plt.tight_layout()
    return fig

# --- Sidebar for Project Management ---
with st.sidebar:
    st.header("📁 Project Management")
    
    # Project Description
    project_desc = st.text_area(
        "Project Description",
        value=st.session_state.get("project_description", ""),
        key="project_description_input",
        help="Add a description for this project"
    )
    if project_desc:
        st.session_state["project_description"] = project_desc
    
    st.divider()
    
    # Save Project
    st.subheader("💾 Save Project")
    project_name = st.text_input("Project Name", value="my_project", key="save_project_name")
    if st.button("📥 Save to File", use_container_width=True):
        if project_name:
            project_data = save_project_to_dict()
            json_str = json.dumps(project_data, indent=2)
            st.download_button(
                label="⬇️ Download Project File",
                data=json_str,
                file_name=f"{project_name}.json",
                mime="application/json",
                use_container_width=True
            )
            st.success(f"Project ready for download!")
        else:
            st.error("Please enter a project name.")
    
    st.divider()
    
    # Load Project
    st.subheader("📂 Load Project")
    # Use a dynamic key so the file uploader resets after a successful load
    # (prevents requiring the user to click "x" to fully apply the loaded project)
    file_loader_key = f"project_loader_{st.session_state.get('file_loader_key', 0)}"
    uploaded_file = st.file_uploader(
        "Choose a project file",
        type=["json"],
        help="Upload a previously saved project JSON file",
        key=file_loader_key
    )
    if uploaded_file is not None:
        try:
            project_data = json.loads(uploaded_file.read())
            if load_project_from_dict(project_data):
                st.success("✅ Project loaded successfully!")
                # Increment key so the file uploader resets on the next run
                st.session_state["file_loader_key"] = st.session_state.get("file_loader_key", 0) + 1
                st.rerun()
            else:
                st.error("❌ Failed to load project.")
        except json.JSONDecodeError:
            st.error("❌ Invalid JSON file.")
    
    st.divider()
    
    # New Project
    if st.button("🆕 New Project", use_container_width=True):
        # Clear all session state so widget values reset to their defaults
        st.session_state.clear()
        st.success("✅ New project created!")
        st.rerun()
    
    st.divider()
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("**Version:** 1.0")
    st.markdown("**Author:** Lucas Sarmento")

# --- Initialize session state ---
if "nodes" not in st.session_state:
    st.session_state["nodes"] = []
if "properties" not in st.session_state:
    st.session_state["properties"] = []
if "elements" not in st.session_state:
    st.session_state["elements"] = []
if "constraints" not in st.session_state:
    st.session_state["constraints"] = []
if "point_loads" not in st.session_state:
    st.session_state["point_loads"] = []
if "distributed_loads" not in st.session_state:
    st.session_state["distributed_loads"] = []

# --- Main Content: Tabs for better organization ---
tab1, tab2, tab3, tab4 = st.tabs(["📐 Structure Definition", "⚙️ Analysis", "📊 Results", "ℹ️ Help"])

with tab1:
    # --- Input: Nodes ---
    with st.expander("🔵 Nodes", expanded=True):
        st.markdown("Define the nodal points of your structure.")
        
        n_nodes = st.number_input(
            "Number of nodes",
            min_value=2,
            max_value=20,
            value=len(st.session_state.get("nodes", [])) if len(st.session_state.get("nodes", [])) >= 2 else 2,
            help="Number of nodes in the structure (minimum 2)"
        )
        
        nodes = []
        validation_errors = []
        
        for i in range(n_nodes):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            # Get existing values if available
            existing_node = st.session_state.get("nodes", [])[i] if i < len(st.session_state.get("nodes", [])) else None
            default_x = existing_node[0] if existing_node else float(i)
            default_y = existing_node[1] if existing_node else 0.0
            
            x = col1.number_input(
                f"Node {i+1} - X coordinate",
                value=default_x,
                format="%.4f",
                key=f"node_x_{i}"
            )
            y = col2.number_input(
                f"Node {i+1} - Y coordinate",
                value=default_y,
                format="%.4f",
                key=f"node_y_{i}"
            )
            
            # Visual indicator
            col3.markdown(f"**Node {i+1}**")
            
            nodes.append((x, y))
        
        # Check for duplicate nodes
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], start=i+1):
                if np.isclose(node1[0], node2[0]) and np.isclose(node1[1], node2[1]):
                    validation_errors.append(f"⚠️ Nodes {i+1} and {j+1} have the same coordinates!")
        
        if validation_errors:
            for error in validation_errors:
                st.warning(error)
        else:
            st.success(f"✅ {n_nodes} nodes defined successfully.")
        
        st.session_state["nodes"] = nodes

    # --- Input: Properties (Material + Section) ---
    with st.expander("🔧 Properties (Material + Section)", expanded=True):
        st.markdown("Define material and section properties for your elements.")
        
        n_properties = st.number_input(
            "Number of properties",
            min_value=1,
            max_value=10,
            value=len(st.session_state.get("properties", [])) if len(st.session_state.get("properties", [])) >= 1 else 1,
            help="Different property sets for different element types"
        )
        
        properties = []
        
        for i in range(n_properties):
            with st.container(border=True):
                st.subheader(f"Property Set {i+1}")
                
                # Get existing values if available
                existing_prop = st.session_state.get("properties", [])[i] if i < len(st.session_state.get("properties", [])) else None
                
                col1, col2 = st.columns([2, 3])
                
                with col1:
                    prop_name = st.text_input(
                        "Property Name",
                        value=existing_prop.get("name", f"Property_{i+1}") if existing_prop else f"Property_{i+1}",
                        key=f"propname_{i}",
                        help="Unique name for this property set"
                    )
                    
                    st.markdown("**Material Properties**")

                    _MAT_MODES = [
                        "Calculate G (from E and ν)",
                        "Calculate ν (from E and G)",
                        "Calculate E (from G and ν)",
                    ]
                    _existing_mode = existing_prop.get("mat_input_mode", _MAT_MODES[0]) if existing_prop else _MAT_MODES[0]
                    _mode_index = _MAT_MODES.index(_existing_mode) if _existing_mode in _MAT_MODES else 0

                    mat_input_mode = st.radio(
                        "Calculate:",
                        _MAT_MODES,
                        index=_mode_index,
                        key=f"mat_mode_{i}",
                        help=(
                            "Choose which elastic constant to derive. "
                            "Enter the other two — only 2 of E, G, ν can be "
                            "specified at the same time; the third is computed "
                            "from G = E / (2·(1 + ν))."
                        ),
                        horizontal=True,
                    )

                    if mat_input_mode == "Calculate G (from E and ν)":
                        E = st.number_input(
                            "Young's Modulus E",
                            value=existing_prop["material"].E if existing_prop and "material" in existing_prop else DEFAULT_E,
                            format="%.2e",
                            key=f"E_{i}",
                            help="Elastic modulus of the material",
                        )
                        valid_E, msg_E = validate_number(E, min_val=0.0, field_name="Young's modulus")
                        if not valid_E:
                            st.error(msg_E)

                        nu = st.number_input(
                            "Poisson's Ratio ν",
                            min_value=0.0,
                            max_value=0.5,
                            value=existing_prop["material"].nu if existing_prop and "material" in existing_prop else DEFAULT_NU,
                            format="%.3f",
                            key=f"nu_{i}",
                            help="Poisson's ratio (0.0 to 0.5)",
                        )

                        _G_computed = E / (2.0 * (1.0 + nu))
                        st.number_input(
                            "Shear Modulus G (computed)",
                            value=_G_computed,
                            format="%.2e",
                            key=f"G_{i}",
                            disabled=True,
                            help="Computed automatically from G = E / (2·(1 + ν))",
                        )
                        mat_E, mat_nu, mat_G = E, nu, None

                    elif mat_input_mode == "Calculate ν (from E and G)":
                        E = st.number_input(
                            "Young's Modulus E",
                            value=existing_prop["material"].E if existing_prop and "material" in existing_prop else DEFAULT_E,
                            format="%.2e",
                            key=f"E_{i}",
                            help="Elastic modulus of the material",
                        )
                        valid_E, msg_E = validate_number(E, min_val=0.0, field_name="Young's modulus")
                        if not valid_E:
                            st.error(msg_E)

                        G_in = st.number_input(
                            "Shear Modulus G",
                            value=existing_prop["material"].G if existing_prop and "material" in existing_prop else DEFAULT_G,
                            format="%.2e",
                            key=f"G_{i}",
                            help="Shear modulus of the material",
                        )
                        valid_G, msg_G = validate_number(G_in, min_val=0.0, field_name="Shear modulus")
                        if not valid_G:
                            st.error(msg_G)

                        _nu_computed = E / (2.0 * G_in) - 1.0 if G_in > 0 else 0.0
                        st.number_input(
                            "Poisson's Ratio ν (computed)",
                            value=_nu_computed,
                            format="%.4f",
                            key=f"nu_{i}",
                            disabled=True,
                            help="Computed automatically from ν = E / (2·G) − 1",
                        )
                        if G_in == 0:
                            st.error("⚠️ Shear modulus G must be greater than zero.")
                        elif not (0.0 <= _nu_computed <= 0.5):
                            st.warning(
                                f"⚠️ Computed Poisson's ratio ν = {_nu_computed:.4f} is outside the "
                                "physically valid range [0, 0.5]. Check your E and G values."
                            )
                        mat_E, mat_nu, mat_G = E, None, G_in

                    else:  # "Calculate E (from G and ν)"
                        G_in = st.number_input(
                            "Shear Modulus G",
                            value=existing_prop["material"].G if existing_prop and "material" in existing_prop else DEFAULT_G,
                            format="%.2e",
                            key=f"G_{i}",
                            help="Shear modulus of the material",
                        )
                        valid_G, msg_G = validate_number(G_in, min_val=0.0, field_name="Shear modulus")
                        if not valid_G:
                            st.error(msg_G)

                        nu = st.number_input(
                            "Poisson's Ratio ν",
                            min_value=0.0,
                            max_value=0.5,
                            value=existing_prop["material"].nu if existing_prop and "material" in existing_prop else DEFAULT_NU,
                            format="%.3f",
                            key=f"nu_{i}",
                            help="Poisson's ratio (0.0 to 0.5)",
                        )

                        _E_computed = 2.0 * G_in * (1.0 + nu)
                        st.number_input(
                            "Young's Modulus E (computed)",
                            value=_E_computed,
                            format="%.2e",
                            key=f"E_{i}",
                            disabled=True,
                            help="Computed automatically from E = 2·G·(1 + ν)",
                        )
                        mat_E, mat_nu, mat_G = None, nu, G_in

                
                with col2:
                    st.markdown("**Section Properties**")
                    section_type = st.selectbox(
                        "Section Type",
                        SECTION_TYPES,
                        index=SECTION_TYPES.index(existing_prop.get("section_type", "rectangular_bar")) if existing_prop and "section_type" in existing_prop else 0,
                        key=f"sectype_{i}",
                        help="Cross-sectional shape"
                    )
                    
                    section_kwargs = {}
                    existing_kwargs = existing_prop.get("section_kwargs", {}) if existing_prop else {}
                    
                    # Section-specific inputs with validation
                    if section_type == "rectangular_bar":
                        width = st.number_input(
                            "Width",
                            value=existing_kwargs.get("width", 0.05),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"width_{i}"
                        )
                        height = st.number_input(
                            "Height",
                            value=existing_kwargs.get("height", 0.10),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"height_{i}"
                        )
                        section_kwargs = dict(width=width, height=height)
                    
                    elif section_type == "rectangular_tube":
                        width = st.number_input(
                            "Width",
                            value=existing_kwargs.get("width", 0.05),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"width_{i}"
                        )
                        height = st.number_input(
                            "Height",
                            value=existing_kwargs.get("height", 0.10),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"height_{i}"
                        )
                        thickness = st.number_input(
                            "Wall Thickness",
                            value=existing_kwargs.get("thickness", 0.005),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"thick_{i}"
                        )
                        section_kwargs = dict(width=width, height=height, thickness=thickness)
                        
                        # Validate thickness
                        if thickness >= min(width, height) / 2:
                            st.error("⚠️ Wall thickness must be less than half the minimum dimension!")
                    
                    elif section_type == "circular_bar":
                        diameter = st.number_input(
                            "Diameter",
                            value=existing_kwargs.get("diameter", 0.05),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"diam_{i}"
                        )
                        section_kwargs = dict(diameter=diameter)
                    
                    elif section_type == "circular_tube":
                        outer_diameter = st.number_input(
                            "Outer Diameter",
                            value=existing_kwargs.get("outer_diameter", 0.05),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"odiam_{i}"
                        )
                        thickness = st.number_input(
                            "Wall Thickness",
                            value=existing_kwargs.get("thickness", 0.005),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"thick_{i}"
                        )
                        section_kwargs = dict(outer_diameter=outer_diameter, thickness=thickness)
                        
                        # Validate thickness
                        if thickness >= outer_diameter / 2:
                            st.error("⚠️ Wall thickness must be less than half the outer diameter!")
                    
                    elif section_type == "trapezoidal_bar":
                        base1 = st.number_input(
                            "Base 1",
                            value=existing_kwargs.get("base1", 0.05),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"base1_{i}"
                        )
                        base2 = st.number_input(
                            "Base 2",
                            value=existing_kwargs.get("base2", 0.10),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"base2_{i}"
                        )
                        height = st.number_input(
                            "Height",
                            value=existing_kwargs.get("height", 0.10),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"height_{i}"
                        )
                        section_kwargs = dict(base1=base1, base2=base2, height=height)
                    
                    elif section_type == "trapezoidal_tube":
                        base1 = st.number_input(
                            "Base 1",
                            value=existing_kwargs.get("base1", 0.05),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"base1_{i}"
                        )
                        base2 = st.number_input(
                            "Base 2",
                            value=existing_kwargs.get("base2", 0.10),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"base2_{i}"
                        )
                        height = st.number_input(
                            "Height",
                            value=existing_kwargs.get("height", 0.10),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"height_{i}"
                        )
                        thickness = st.number_input(
                            "Wall Thickness",
                            value=existing_kwargs.get("thickness", 0.005),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"thick_{i}"
                        )
                        section_kwargs = dict(base1=base1, base2=base2, height=height, thickness=thickness)
                    
                    elif section_type == "hexagonal_bar":
                        side = st.number_input(
                            "Side Length",
                            value=existing_kwargs.get("side", 0.05),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"side_{i}"
                        )
                        section_kwargs = dict(side=side)
                    
                    elif section_type == "hexagonal_tube":
                        outer_side = st.number_input(
                            "Outer Side Length",
                            value=existing_kwargs.get("outer_side", 0.05),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"oside_{i}"
                        )
                        thickness = st.number_input(
                            "Wall Thickness",
                            value=existing_kwargs.get("thickness", 0.005),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"thick_{i}"
                        )
                        section_kwargs = dict(outer_side=outer_side, thickness=thickness)
                    
                    elif section_type == "ibeam":
                        h = st.number_input(
                            "Height h",
                            value=existing_kwargs.get("h", 0.10),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"h_{i}"
                        )
                        b = st.number_input(
                            "Flange Width b",
                            value=existing_kwargs.get("b", 0.05),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"b_{i}"
                        )
                        tw = st.number_input(
                            "Web Thickness tw",
                            value=existing_kwargs.get("tw", 0.005),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"tw_{i}"
                        )
                        tf = st.number_input(
                            "Flange Thickness tf",
                            value=existing_kwargs.get("tf", 0.005),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"tf_{i}"
                        )
                        section_kwargs = dict(h=h, b=b, tw=tw, tf=tf)
                    
                    elif section_type == "c_section":
                        h = st.number_input(
                            "Height h",
                            value=existing_kwargs.get("h", 0.10),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"h_{i}"
                        )
                        b = st.number_input(
                            "Flange Width b",
                            value=existing_kwargs.get("b", 0.05),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"b_{i}"
                        )
                        tw = st.number_input(
                            "Web Thickness tw",
                            value=existing_kwargs.get("tw", 0.005),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"tw_{i}"
                        )
                        tf = st.number_input(
                            "Flange Thickness tf",
                            value=existing_kwargs.get("tf", 0.005),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"tf_{i}"
                        )
                        section_kwargs = dict(h=h, b=b, tw=tw, tf=tf)
                    
                    elif section_type == "l_section":
                        b = st.number_input(
                            "Width b",
                            value=existing_kwargs.get("b", 0.05),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"b_{i}"
                        )
                        h = st.number_input(
                            "Height h",
                            value=existing_kwargs.get("h", 0.10),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"h_{i}"
                        )
                        t = st.number_input(
                            "Thickness t",
                            value=existing_kwargs.get("t", 0.005),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"t_{i}"
                        )
                        section_kwargs = dict(b=b, h=h, t=t)
                    
                    elif section_type == "t_section":
                        b = st.number_input(
                            "Flange Width b",
                            value=existing_kwargs.get("b", 0.05),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"b_{i}"
                        )
                        h = st.number_input(
                            "Height h",
                            value=existing_kwargs.get("h", 0.10),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"h_{i}"
                        )
                        tw = st.number_input(
                            "Web Thickness tw",
                            value=existing_kwargs.get("tw", 0.005),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"tw_{i}"
                        )
                        tf = st.number_input(
                            "Flange Thickness tf",
                            value=existing_kwargs.get("tf", 0.005),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"tf_{i}"
                        )
                        section_kwargs = dict(b=b, h=h, tw=tw, tf=tf)
                    
                    elif section_type == "z_section":
                        h = st.number_input(
                            "Height h",
                            value=existing_kwargs.get("h", 0.10),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"h_{i}"
                        )
                        b = st.number_input(
                            "Flange Width b",
                            value=existing_kwargs.get("b", 0.05),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"b_{i}"
                        )
                        tw = st.number_input(
                            "Web Thickness tw",
                            value=existing_kwargs.get("tw", 0.005),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"tw_{i}"
                        )
                        tf = st.number_input(
                            "Flange Thickness tf",
                            value=existing_kwargs.get("tf", 0.005),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"tf_{i}"
                        )
                        section_kwargs = dict(h=h, b=b, tw=tw, tf=tf)
                    
                    elif section_type == "hat_section":
                        h = st.number_input(
                            "Height h",
                            value=existing_kwargs.get("h", 0.10),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"h_{i}"
                        )
                        b = st.number_input(
                            "Flange Width b",
                            value=existing_kwargs.get("b", 0.05),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"b_{i}"
                        )
                        tw = st.number_input(
                            "Web Thickness tw",
                            value=existing_kwargs.get("tw", 0.005),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"tw_{i}"
                        )
                        tf = st.number_input(
                            "Flange Thickness tf",
                            value=existing_kwargs.get("tf", 0.005),
                            format="%.4f",
                            min_value=0.0001,
                            key=f"tf_{i}"
                        )
                        section_kwargs = dict(h=h, b=b, tw=tw, tf=tf)
                    
                    elif section_type == "general":
                        area = st.number_input(
                            "Cross-sectional Area (m²)",
                            value=existing_kwargs.get("area", 0.001),
                            format="%.6f",
                            min_value=0.000001,
                            key=f"area_{i}"
                        )
                        inertia = st.number_input(
                            "Moment of Inertia (m⁴)",
                            value=existing_kwargs.get("inertia", 1e-6),
                            format="%.6e",
                            min_value=1e-12,
                            key=f"inertia_{i}"
                        )
                        section_kwargs = dict(area=area, inertia=inertia)
                    
                    # Display section preview diagram
                    st.markdown("**Section Preview**")
                    try:
                        preview_fig = create_section_preview(section_type, **section_kwargs)
                        st.pyplot(preview_fig, use_container_width=False)
                        plt.close(preview_fig)  # Clean up to avoid memory issues
                    except Exception as e:
                        st.info("Preview not available for this section type.")
                
                material = Material(i+1, E=mat_E, nu=mat_nu, G=mat_G)
                section = create_section(section_type, i+1, **section_kwargs)
                properties.append({
                    "name": prop_name,
                    "material": material,
                    "mat_input_mode": mat_input_mode,
                    "section": section,
                    "section_type": section_type,
                    "section_kwargs": section_kwargs
                })
        
        st.session_state["properties"] = properties
        st.success(f"✅ {n_properties} property set(s) defined successfully.")

    # --- Input: Elements ---
    with st.expander("🔗 Elements", expanded=True):
        st.markdown("Define beam elements connecting nodes.")
        
        if not properties:
            st.warning("⚠️ Please define at least one property set first.")
        else:
            property_names = [prop["name"] for prop in properties]
            element_types = ELEMENT_TYPES
            
            n_elements = st.number_input(
                "Number of elements",
                min_value=1,
                max_value=n_nodes-1,
                value=len(st.session_state.get("elements", [])) if len(st.session_state.get("elements", [])) >= 1 else min(n_nodes-1, 1),
                help="Number of beam elements"
            )
            
            elements = []
            
            for i in range(n_elements):
                with st.container(border=True):
                    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
                    
                    existing_elem = st.session_state.get("elements", [])[i] if i < len(st.session_state.get("elements", [])) else None
                    
                    n1 = int(col1.number_input(
                        f"Start Node",
                        min_value=1,
                        max_value=n_nodes,
                        value=existing_elem[0] if existing_elem else min(i+1, n_nodes),
                        key=f"en1_{i}",
                        help=f"Element {i+1} start node"
                    ))
                    
                    n2 = int(col2.number_input(
                        f"End Node",
                        min_value=1,
                        max_value=n_nodes,
                        value=existing_elem[1] if existing_elem else min(i+2, n_nodes),
                        key=f"en2_{i}",
                        help=f"Element {i+1} end node"
                    ))
                    
                    if n1 == n2:
                        st.error(f"⚠️ Element {i+1}: Start and end nodes must be different!")
                    
                    el_type = col3.selectbox(
                        f"Type",
                        list(element_types.keys()),
                        index=list(element_types.keys()).index(next((k for k, v in element_types.items() if v == existing_elem[2]), list(element_types.keys())[0])) if existing_elem and len(existing_elem) > 2 else 0,
                        key=f"etype_{i}",
                        help=f"Element {i+1} formulation"
                    )
                    
                    prop_idx = col4.selectbox(
                        f"Property",
                        property_names,
                        index=property_names.index(existing_elem[3]) if existing_elem and len(existing_elem) > 3 and existing_elem[3] in property_names else 0,
                        key=f"propidx_{i}",
                        help=f"Element {i+1} property"
                    )
                    
                    n_subdiv = col5.number_input(
                        f"Subdivisions",
                        min_value=1,
                        max_value=256,
                        value=existing_elem[4] if existing_elem and len(existing_elem) > 4 else 1,
                        key=f"subdiv_{i}",
                        help=f"Element {i+1} mesh refinement"
                    )
                    
                    elements.append((n1, n2, element_types[el_type], prop_idx, n_subdiv))
            
            st.session_state["elements"] = elements
            st.success(f"✅ {n_elements} element(s) defined successfully.")

    # --- Input: Constraints ---
    with st.expander("🔒 Constraints (Boundary Conditions)", expanded=True):
        st.markdown("Define fixed or prescribed displacements and rotations.")

        # Detect whether any Reddy-Bickford element is in use (needs 4 DOFs/node)
        _has_reddy = any(
            e[2] == "reddy_bickford_2node"
            for e in st.session_state.get("elements", [])
        )
        _dof_options_c = [0, 1, 2, 3] if _has_reddy else [0, 1, 2]
        _dof_labels_c = ["X displacement", "Y displacement", "Rotation", "Slope (dv/dx)"]
        
        n_constraints = st.number_input(
            "Number of constraints",
            min_value=0,
            max_value=n_nodes * (4 if _has_reddy else 3),
            value=len(st.session_state.get("constraints", [])),
            help="Define boundary conditions (e.g., fixed supports)"
        )
        
        constraints = []
        
        if n_constraints > 0:
            for i in range(n_constraints):
                with st.container(border=True):
                    col1, col2, col3 = st.columns([2, 2, 2])
                    
                    existing_const = st.session_state.get("constraints", [])[i] if i < len(st.session_state.get("constraints", [])) else None
                    
                    node_id = int(col1.number_input(
                        f"Node",
                        min_value=1,
                        max_value=n_nodes,
                        value=existing_const[0] if existing_const else 1,
                        key=f"cnode_{i}",
                        help=f"Constraint {i+1} at node"
                    ))

                    _existing_dir = existing_const[1] if existing_const and len(existing_const) > 1 else 0
                    _dir_index = _dof_options_c.index(_existing_dir) if _existing_dir in _dof_options_c else 0
                    direction = int(col2.selectbox(
                        f"DOF",
                        options=_dof_options_c,
                        format_func=lambda x: _dof_labels_c[x],
                        index=_dir_index,
                        key=f"cdir_{i}",
                        help=f"Constraint {i+1} direction"
                    ))
                    
                    value = col3.number_input(
                        f"Value",
                        value=existing_const[2] if existing_const and len(existing_const) > 2 else 0.0,
                        format="%.6f",
                        key=f"cval_{i}",
                        help=f"Constraint {i+1} prescribed value"
                    )
                    
                    constraints.append((node_id, direction, value))
            
            st.session_state["constraints"] = constraints
            st.success(f"✅ {n_constraints} constraint(s) defined successfully.")
        else:
            st.info("ℹ️ No constraints defined. The structure may be unstable without proper boundary conditions.")

    # --- Input: Point Loads ---
    with st.expander("⬇️ Point Loads", expanded=True):
        st.markdown("Define concentrated forces and moments at nodes.")

        # Detect Reddy elements (computed above for constraints, re-used here)
        _dof_options_l = [0, 1, 2, 3] if _has_reddy else [0, 1, 2]
        _dof_labels_l = ["X force", "Y force", "Moment", "Applied slope (dv/dx)"]
        
        n_loads = st.number_input(
            "Number of point loads",
            min_value=0,
            max_value=n_nodes * (4 if _has_reddy else 3),
            value=len(st.session_state.get("point_loads", [])),
            help="Define point loads applied at nodes"
        )
        
        point_loads = []
        
        if n_loads > 0:
            for i in range(n_loads):
                with st.container(border=True):
                    col1, col2, col3 = st.columns([2, 2, 2])
                    
                    existing_load = st.session_state.get("point_loads", [])[i] if i < len(st.session_state.get("point_loads", [])) else None
                    
                    node_id = int(col1.number_input(
                        f"Node",
                        min_value=1,
                        max_value=n_nodes,
                        value=existing_load[0] if existing_load else 1,
                        key=f"lnode_{i}",
                        help=f"Load {i+1} at node"
                    ))

                    _existing_ldir = existing_load[1] if existing_load and len(existing_load) > 1 else 1
                    # default to direction 1 (Y force); clamp to valid range
                    _ldir_default = _existing_ldir if _existing_ldir in _dof_options_l else 1
                    _ldir_index = _dof_options_l.index(_ldir_default)
                    direction = int(col2.selectbox(
                        f"Direction",
                        options=_dof_options_l,
                        format_func=lambda x: _dof_labels_l[x],
                        index=_ldir_index,
                        key=f"ldir_{i}",
                        help=f"Load {i+1} direction"
                    ))
                    
                    magnitude = col3.number_input(
                        f"Magnitude",
                        value=existing_load[2] if existing_load and len(existing_load) > 2 else 0.0,
                        format="%.4f",
                        key=f"lmag_{i}",
                        help=f"Load {i+1} magnitude"
                    )
                    
                    point_loads.append((node_id, direction, magnitude))
            
            st.session_state["point_loads"] = point_loads
            st.success(f"✅ {n_loads} point load(s) defined successfully.")
        else:
            st.info("ℹ️ No point loads defined.")

    # --- Input: Distributed Loads ---
    with st.expander("📏 Distributed Loads", expanded=True):
        st.markdown("Define distributed loads along elements.")
        
        n_dist_loads = st.number_input(
            "Number of distributed loads",
            min_value=0,
            value=len(st.session_state.get("distributed_loads", [])),
            help="Define distributed loads on elements"
        )
        
        distributed_loads = []
        
        if n_dist_loads > 0:
            for i in range(n_dist_loads):
                with st.container(border=True):
                    col1, col2, col3 = st.columns([2, 2, 3])
                    
                    existing_dload = st.session_state.get("distributed_loads", [])[i] if i < len(st.session_state.get("distributed_loads", [])) else None
                    
                    element_id = int(col1.number_input(
                        f"Element",
                        min_value=1,
                        value=existing_dload[0] if existing_dload else 1,
                        key=f"dlelem_{i}",
                        help=f"Distributed load {i+1} on element"
                    ))
                    
                    load_type = col2.selectbox(
                        f"Type",
                        options=["constant", "linear", "custom"],
                        index=["constant", "linear", "custom"].index(existing_dload[5] if existing_dload and len(existing_dload) > 5 else "constant"),
                        key=f"dltype_{i}",
                        help=f"Load {i+1} distribution type"
                    )
                    
                    direction = col3.selectbox(
                        f"Direction",
                        options=['x', 'y', 'l', 't'],
                        format_func=lambda x: {"x": "Global X", "y": "Global Y", "l": "Local axial", "t": "Local transverse"}[x],
                        index=['x', 'y', 'l', 't'].index(existing_dload[3] if existing_dload and len(existing_dload) > 3 else 'y'),
                        key=f"ddir_{i}",
                        help=f"Load {i+1} direction"
                    )
                    
                    col4, col5 = st.columns(2)
                    
                    if load_type == "constant":
                        magnitude = col4.number_input(
                            f"Value",
                            value=existing_dload[1] if existing_dload and existing_dload[1] is not None else 0.0,
                            format="%.4f",
                            key=f"dlval_{i}",
                            help=f"Load {i+1} constant value"
                        )
                        distributed_loads.append((element_id, magnitude, None, direction, None, "constant"))
                    
                    elif load_type == "linear":
                        magnitude_start = col4.number_input(
                            f"Start",
                            value=existing_dload[1] if existing_dload and existing_dload[1] is not None else 0.0,
                            format="%.4f",
                            key=f"dlstart_{i}",
                            help=f"Load {i+1} start value"
                        )
                        magnitude_end = col5.number_input(
                            f"End",
                            value=existing_dload[2] if existing_dload and existing_dload[2] is not None else 0.0,
                            format="%.4f",
                            key=f"dlend_{i}",
                            help=f"Load {i+1} end value"
                        )
                        distributed_loads.append((element_id, magnitude_start, magnitude_end, direction, None, "linear"))
                    
                    elif load_type == "custom":
                        func_str = col4.text_input(
                            f"Function f(x)",
                            value=existing_dload[4] if existing_dload and existing_dload[4] is not None else "",
                            key=f"dlfunc_{i}",
                            help="Enter function in terms of x and L (e.g., 'x**2' or 'np.sin(x/L)')"
                        )
                        
                        # Validate function string
                        error_msg = ""
                        if func_str:
                            try:
                                x = 0.0
                                L = 1.0
                                test_val = eval(func_str, {"np": np, "x": x, "L": L})
                                col5.success("✅ Valid function")
                            except Exception as e:
                                error_msg = f"Invalid function: {e}"
                                col5.error(error_msg)
                            
                            # Show LaTeX representation for user verification
                            latex_str = _func_str_to_latex(func_str)
                            if latex_str and not error_msg:
                                st.markdown("**Load function** $q(x)$:")
                                st.latex(rf"q(x) = {latex_str}")
                        
                        distributed_loads.append((element_id, None, None, direction, func_str if not error_msg else None, "custom"))
            
            st.session_state["distributed_loads"] = distributed_loads
            st.success(f"✅ {n_dist_loads} distributed load(s) defined successfully.")
        else:
            st.info("ℹ️ No distributed loads defined.")


with tab2:
    st.header("⚙️ Run Analysis")
    st.markdown("Execute finite element analysis on the defined structure.")
    
    # Summary of model
    with st.expander("📋 Model Summary", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Nodes", len(nodes))
        col2.metric("Elements", len(st.session_state.get("elements", [])))
        col3.metric("Properties", len(st.session_state.get("properties", [])))
        
        col4, col5, col6 = st.columns(3)
        col4.metric("Constraints", len(st.session_state.get("constraints", [])))
        col5.metric("Point Loads", len(st.session_state.get("point_loads", [])))
        col6.metric("Distributed Loads", len(st.session_state.get("distributed_loads", [])))

        # Show total number of mesh nodal points (including subdivision nodes)
        mesh_obj = st.session_state.get("mesh", None)
        if mesh_obj is not None:
            mesh_node_count = len(mesh_obj.nodes)
        else:
            # Estimate: user-defined nodes + interior subdivision nodes per element
            # Element tuple structure: (n1, n2, etype, prop_name, n_subdiv)
            mesh_node_count = len(nodes)
            for elem in st.session_state.get("elements", []):
                n1, n2, etype, prop_name, n_subdiv = elem if len(elem) == 5 else (*elem, 1)
                if n_subdiv > 1:
                    mesh_node_count += n_subdiv - 1
        mesh_label = "Mesh Nodal Points" if mesh_obj is not None else "Mesh Nodal Points (est.)"
        st.metric(mesh_label, mesh_node_count)

    # Validation before analysis
    can_analyze = True
    validation_messages = []
    
    if len(nodes) < 2:
        can_analyze = False
        validation_messages.append("❌ At least 2 nodes are required.")
    
    if len(st.session_state.get("elements", [])) < 1:
        can_analyze = False
        validation_messages.append("❌ At least 1 element is required.")
    
    if len(st.session_state.get("properties", [])) < 1:
        can_analyze = False
        validation_messages.append("❌ At least 1 property set is required.")
    
    if len(st.session_state.get("constraints", [])) < 1:
        validation_messages.append("⚠️ Warning: No constraints defined. The structure may be unstable.")
    
    if len(st.session_state.get("point_loads", [])) < 1 and len(st.session_state.get("distributed_loads", [])) < 1:
        validation_messages.append("⚠️ Warning: No loads defined.")
    
    if validation_messages:
        for msg in validation_messages:
            if "❌" in msg:
                st.error(msg)
            else:
                st.warning(msg)
    
    if can_analyze:
        # Preview button - shows structure with loads before analysis
        if st.button("👁️ Preview Structure with Loads", use_container_width=True):
            with st.spinner("Generating structure preview..."):
                try:
                    preview_fig = plot_structure_preview(
                        nodes=nodes,
                        elements=st.session_state.get("elements", []),
                        properties=st.session_state.get("properties", []),
                        constraints=st.session_state.get("constraints", []),
                        point_loads=st.session_state.get("point_loads", []),
                        distributed_loads=st.session_state.get("distributed_loads", [])
                    )
                    st.plotly_chart(preview_fig, use_container_width=True)
                    st.success("✅ Structure preview generated successfully! Review your setup before running the analysis.")
                except Exception as e:
                    st.error(f"Error generating preview: {str(e)}")
                    st.code(traceback.format_exc())
        
        st.divider()
        
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            with st.spinner("Running finite element analysis..."):
                try:
                    mesh = Mesh()
                    node_objs = []
                    
                    # Add nodes
                    for x, y in nodes:
                        node_objs.append(mesh.add_node(x, y))
                    
                    # Add elements with selected property.
                    # Track the mapping from original element index (1-based) to all
                    # mesh element IDs created for it (may be multiple when n_subdiv > 1).
                    original_to_mesh_elements = {}  # {orig_idx (1-based): [mesh_element_id, ...]}
                    for orig_idx, (n1, n2, etype, prop_name, n_subdiv) in enumerate(
                            st.session_state["elements"], start=1):
                        prop = next(p for p in st.session_state["properties"] if p["name"] == prop_name)
                        node_start = mesh.get_node_by_id(n1)
                        node_end = mesh.get_node_by_id(n2)
                        
                        if n_subdiv == 1:
                            el = mesh.add_element(
                                node_start,
                                node_end,
                                prop["material"],
                                prop["section"],
                                element_type=etype
                            )
                            original_to_mesh_elements[orig_idx] = [el.id]
                        else:
                            # Subdivide element
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
                                    prop["material"],
                                    prop["section"],
                                    element_type=etype
                                )
                                subdiv_ids.append(el.id)
                            original_to_mesh_elements[orig_idx] = subdiv_ids
                    
                    # Add constraints
                    for node_id, direction, value in st.session_state["constraints"]:
                        mesh.constraints.add(Constraint(mesh.get_node_by_id(node_id), direction, value))
                    
                    # Add point loads
                    for node_id, direction, magnitude in st.session_state["point_loads"]:
                        mesh.point_loads.append(PointLoad(magnitude, direction))
                        mesh.point_loads[-1].node = mesh.get_node_by_id(node_id)
                    
                    # Add distributed loads.
                    # element_id in the distributed_loads list is the 1-based index of the
                    # original element as defined by the user (before subdivision).  Apply
                    # the load to every mesh element that was created for that original element
                    # so that subdivided elements all carry the correct distributed load.
                    for element_id, magnitude_start, magnitude_end, direction, func_str, _ in st.session_state["distributed_loads"]:
                        mesh_el_ids = original_to_mesh_elements.get(element_id)
                        if mesh_el_ids is None:
                            raise ValueError(
                                f"Distributed load references element {element_id}, "
                                f"which was not found in the mesh. "
                                f"Valid element indices are: {list(original_to_mesh_elements.keys())}"
                            )
                        for mesh_el_id in mesh_el_ids:
                            el = mesh.get_element_by_id(mesh_el_id)
                            load = DistributedLoad(magnitude_start, magnitude_end, direction, func=func_str)
                            load.element = el
                            mesh.distributed_loads.append(load)
                    
                    # Run analysis
                    # Note: EulerBernoulliAnalysis (or its alias BeamAnalysis) is a generic
                    # beam analysis class that works with all element types (Euler-Bernoulli,
                    # Timoshenko, and mixed) through polymorphism. Each element implements its
                    # own stiffness_matrix() method according to its respective beam theory.
                    # For new code, consider using BeamAnalysis for clarity.
                    analysis = EulerBernoulliAnalysis(mesh)
                    analysis.assemble()
                    displacements = analysis.solve()

                    # Get dpn (degrees of freedom per node) from analysis
                    dpn = getattr(analysis, 'dpn', 3)

                    # Get reactions from analysis
                    reactions = analysis.get_reactions()

                    # Store results
                    st.session_state["mesh"] = mesh
                    st.session_state["displacements"] = displacements
                    st.session_state["reactions"] = reactions
                    st.session_state["n_original_nodes"] = len(nodes)
                    st.session_state["dpn"] = dpn

                    # Prepare post-processing
                    structure_results = StructureResults(mesh, displacements, reactions, dpn)
                    st.session_state["structure_results"] = structure_results
                    # Reset diagram visibility flags so new results are shown cleanly
                    st.session_state["force_diagram_generated"] = False
                    st.session_state["stress_dist_generated"] = False

                    st.success("✅ Analysis completed successfully!")

                    # Display results preview
                    st.subheader("📊 Nodal Displacements")
                    disp_data = []
                    for i, node in enumerate(mesh.nodes):
                        u = displacements[dpn*i]
                        v = displacements[dpn*i+1]
                        theta = displacements[dpn*i+2]
                        row = {
                            "Node": node.id,
                            "X": f"{node.x:.4f}",
                            "Y": f"{node.y:.4f}",
                            "U": f"{u:.6e}",
                            "V": f"{v:.6e}",
                            "θ": f"{theta:.6e}"
                        }
                        if dpn == 4 and dpn*i+3 < len(displacements):
                            row["dv/dx"] = f"{displacements[dpn*i+3]:.6e}"
                        disp_data.append(row)
                    
                    df_disp = pd.DataFrame(disp_data)
                    st.dataframe(df_disp, use_container_width=True)
                    
                    # Export CSV
                    csv = df_disp.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Displacements CSV",
                        csv,
                        "displacements.csv",
                        "text/csv",
                        use_container_width=True
                    )
                    
                    # Display reactions at constraints
                    if reactions:
                        st.subheader("⚙️ Reaction Forces at Constraints")
                        
                        # Map direction to labels
                        direction_labels = {0: "X", 1: "Y", 2: "Rotation", 3: "Slope (dv/dx)"}
                        
                        # Create node lookup dictionary for O(1) access
                        node_lookup = {n.id: n for n in mesh.nodes}
                        
                        reaction_data = []
                        for (node_id, direction), force in reactions.items():
                            node = node_lookup[node_id]
                            reaction_data.append({
                                "Node": node_id,
                                "X": f"{node.x:.4f}",
                                "Y": f"{node.y:.4f}",
                                "Direction": direction_labels.get(direction, f"DOF {direction}"),
                                "Reaction": f"{force:.6e}"
                            })
                        
                        df_reactions = pd.DataFrame(reaction_data)
                        st.dataframe(df_reactions, use_container_width=True)
                        
                        # Export reactions CSV
                        csv_reactions = df_reactions.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Reactions CSV",
                            csv_reactions,
                            "reactions.csv",
                            "text/csv",
                            use_container_width=True
                        )
                        
                        st.info("ℹ️ Positive reactions indicate forces in positive coordinate directions.")
                    
                    st.info("ℹ️ Go to the 'Results' tab to view diagrams and stress distributions.")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    with st.expander("View Error Details"):
                        st.code(traceback.format_exc())
    else:
        st.error("❌ Cannot run analysis. Please fix the validation errors above.")

with tab3:
    st.header("📊 Results Visualization")
    
    # Use session state for post-processing and plotting
    if "structure_results" in st.session_state:
        structure_results = st.session_state["structure_results"]
        
        # Diagrams
        with st.expander("📈 Force Diagrams", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                diagram_type = st.selectbox(
                    "Select diagram type",
                    options=["Moment", "Shear", "Normal Force"],
                    index=0,
                    help="Choose which force diagram to display"
                )
            
            with col2:
                fill_diagram = st.checkbox(
                    "Fill diagram",
                    value=False,
                    help="Show diagram as filled area"
                )
            
            fill_color = "green"
            fill_opacity = 0.2
            if fill_diagram:
                fill_color_options = {
                    "🟢 Green": "green",
                    "🔵 Blue": "blue",
                    "🔴 Red": "red",
                    "🟠 Orange": "orange",
                    "🟣 Purple": "purple",
                    "🩵 Teal": "teal",
                }
                col_color, col_opacity = st.columns(2)
                with col_color:
                    fill_color_label = st.selectbox(
                        "Fill color",
                        options=list(fill_color_options.keys()),
                        index=0,
                        help="Choose the fill color for the diagram area"
                    )
                    fill_color = fill_color_options[fill_color_label]
                with col_opacity:
                    fill_opacity = st.slider(
                        "Fill opacity",
                        min_value=0.05,
                        max_value=1.0,
                        value=0.2,
                        step=0.05,
                        help="Adjust the transparency of the fill area"
                    )
            
            # --- Performance & display options ----------------------------
            col_pts, col_hide = st.columns(2)
            with col_pts:
                diag_n_points = st.slider(
                    "Diagram resolution (points per element)",
                    min_value=10,
                    max_value=100,
                    value=30,
                    step=5,
                    help=(
                        "Number of sample points per element. "
                        "Lower values render faster; higher values give a smoother curve."
                    ),
                    key="diag_n_points",
                )
            with col_hide:
                hide_subdiv_nodes = st.checkbox(
                    "Hide subdivision nodes",
                    value=False,
                    help=(
                        "When elements are subdivided, many intermediate nodes are added. "
                        "Enable this to show only the original user-defined nodes."
                    ),
                    key="hide_subdiv_nodes",
                )
            
            # --- Extract force value at an exact global position ----------
            st.markdown("##### 📍 Extract value at exact position")
            use_query = st.checkbox(
                "Query a specific (x, y) position",
                value=False,
                help="Mark a specific global coordinate on the diagram and read the force value there.",
                key="diag_use_query",
            )
            query_xy = None
            if use_query:
                col_qx, col_qy = st.columns(2)
                with col_qx:
                    q_x = st.number_input(
                        "x coordinate",
                        value=0.0,
                        format="%.4f",
                        help="Global x of the point to query",
                        key="diag_query_x",
                    )
                with col_qy:
                    q_y = st.number_input(
                        "y coordinate",
                        value=0.0,
                        format="%.4f",
                        help="Global y of the point to query",
                        key="diag_query_y",
                    )
                query_xy = (q_x, q_y)
            
            diagram_map = {
                "Moment": "moment",
                "Shear": "shear",
                "Normal Force": "normal"
            }
            
            if st.button("📊 Generate Diagram", use_container_width=True):
                st.session_state["force_diagram_generated"] = True
            
            if st.session_state.get("force_diagram_generated", False):
                try:
                    n_orig = st.session_state.get("n_original_nodes", None)
                    
                    # Pre-flight check for query position
                    if query_xy is not None:
                        hit = find_position_on_structure(structure_results, query_xy[0], query_xy[1])
                        if hit is None or not hit["is_on_structure"]:
                            st.warning(
                                f"⚠️ The queried position ({query_xy[0]:.4f}, {query_xy[1]:.4f}) "
                                "is outside the structure (or too far from any element). "
                                "The nearest point on the structure will be used."
                            )
                        elif hit is not None:
                            ft = diagram_map[diagram_type]
                            if ft == "moment":
                                force_val_preview = hit["element_result"].bending_moment(hit["local_x"])
                            elif ft == "shear":
                                force_val_preview = hit["element_result"].shear_force(hit["local_x"])
                            else:
                                force_val_preview = hit["element_result"].normal_force(hit["local_x"])
                            st.info(
                                f"📍 Nearest point on structure: "
                                f"({hit['proj_x']:.4f}, {hit['proj_y']:.4f})  |  "
                                f"{diagram_type} = **{force_val_preview:.4f}**  |  "
                                f"Distance from query: {hit['distance']:.4f}"
                            )
                    
                    fig = plot_structure_diagram(
                        structure_results,
                        force_type=diagram_map[diagram_type],
                        n_points=diag_n_points,
                        fill_diagram=fill_diagram,
                        fill_color=fill_color,
                        fill_opacity=fill_opacity,
                        show_subdivision_nodes=not hide_subdiv_nodes,
                        n_original_nodes=n_orig,
                        query_xy=query_xy,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating diagram: {e}")
        
        # Normal Stress Distribution - Cross Section and Side View
        with st.expander("🔍 Normal Stress Distribution - Cross Section & Side View", expanded=False):
            element_ids = [el.id for el in st.session_state["mesh"].elements]
            
            if element_ids:
                # ---- Position selection mode --------------------------------
                use_global_pos = st.checkbox(
                    "Use global (x, y) position to select element automatically",
                    value=False,
                    help=(
                        "Enter a global coordinate and the tool will find the nearest "
                        "element and its local position automatically."
                    ),
                    key="section_use_global_pos",
                )
                
                if use_global_pos:
                    col_gx, col_gy = st.columns(2)
                    with col_gx:
                        sec_gx = st.number_input(
                            "x coordinate",
                            value=0.0,
                            format="%.4f",
                            help="Global x of the cut position",
                            key="section_global_x",
                        )
                    with col_gy:
                        sec_gy = st.number_input(
                            "y coordinate",
                            value=0.0,
                            format="%.4f",
                            help="Global y of the cut position",
                            key="section_global_y",
                        )
                    
                    hit = find_position_on_structure(structure_results, sec_gx, sec_gy)
                    if hit is None:
                        st.warning("⚠️ No elements found in the mesh.")
                        selected_element_result = None
                        x_pos = 0.0
                    elif not hit["is_on_structure"]:
                        st.warning(
                            f"⚠️ The position ({sec_gx:.4f}, {sec_gy:.4f}) is outside the "
                            "structure. The nearest point on the structure will be used."
                        )
                        selected_element_result = hit["element_result"]
                        x_pos = hit["local_x"]
                        st.info(
                            f"Snapped to element {selected_element_result.element.id} "
                            f"at local x = {x_pos:.4f}  "
                            f"(global: {hit['proj_x']:.4f}, {hit['proj_y']:.4f})  |  "
                            f"Distance: {hit['distance']:.4f}"
                        )
                    else:
                        selected_element_result = hit["element_result"]
                        x_pos = hit["local_x"]
                        st.success(
                            f"📍 Element {selected_element_result.element.id}  |  "
                            f"Local x = {x_pos:.4f}  |  "
                            f"Projected point: ({hit['proj_x']:.4f}, {hit['proj_y']:.4f})"
                        )
                else:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        selected_element_id = st.selectbox(
                            "Select element",
                            element_ids,
                            help="Choose element to analyze",
                            key="stress_element_selector"
                        )
                    
                    selected_element_result = next(
                        er for er in structure_results.element_results
                        if er.element.id == selected_element_id
                    )
                    
                    with col2:
                        x_pos = st.slider(
                            "Position along element",
                            min_value=0.0,
                            max_value=float(selected_element_result.length),
                            value=0.0,
                            step=0.01,
                            help="Select position along element to view stress",
                            key="stress_position_slider"
                        )
                
                # --- Section-point query ------------------------------------------------
                st.markdown("##### 📍 Extract stress at specific section position")
                use_section_query = st.checkbox(
                    "Query a specific y-position in the cross-section",
                    value=False,
                    help=(
                        "Enter a y-coordinate within the section (measured from the centroid) "
                        "to read the normal stress σ = N/A − M·y/I at that exact point."
                    ),
                    key="section_query_enabled",
                )
                section_query_y = None
                if use_section_query:
                    section_query_y = st.number_input(
                        "Section y-coordinate (from centroid)",
                        value=0.0,
                        format="%.4f",
                        help="y position within the cross-section (positive = above centroid)",
                        key="section_query_y_value",
                    )
                
                if selected_element_result is not None and st.button(
                    "🔍 Show Stress Distribution (Both Views)", use_container_width=True
                ):
                    st.session_state["stress_dist_generated"] = True
                
                if st.session_state.get("stress_dist_generated", False) and selected_element_result is not None:
                    try:
                        # If a section-y query is active, compute and show the stress value
                        if use_section_query and section_query_y is not None:
                            N_val = selected_element_result.normal_force(x_pos)
                            M_val = selected_element_result.bending_moment(x_pos)
                            sigma_val = selected_element_result.element.section.normal_stress(N_val, M_val, section_query_y)
                            st.info(
                                f"📍 At section y = {section_query_y:.4f}  |  "
                                f"N = {N_val:.4f}  |  M = {M_val:.4f}  |  "
                                f"σ = **{sigma_val:.4f}**"
                            )
                        
                        # Display cross-section view (front view)
                        st.subheader("📐 Cross-Section View (Front)")
                        fig_cross = plot_normal_stress_distribution(
                            selected_element_result,
                            x_pos,
                            query_y=section_query_y if use_section_query else None,
                        )
                        st.plotly_chart(fig_cross, use_container_width=True)
                        
                        # Display side view
                        st.subheader("↔️ Side View")
                        fig_side = plot_normal_stress_side_view(selected_element_result, x_pos)
                        st.plotly_chart(fig_side, use_container_width=True)
                        
                        # Add information box
                        st.info(
                            "💡 **How to interpret the views:**\n"
                            "- **Cross-Section View (Front)**: Shows stress distribution across the section at the cut position\n"
                            "- **Side View**: Shows stress profile along the beam height with arrows indicating magnitude and direction"
                        )
                    except Exception as e:
                        st.error(f"Error: {e}")
                        with st.expander("View Error Details"):
                            st.code(traceback.format_exc())
            else:
                st.warning("No elements available for stress analysis.")
    else:
        st.info("ℹ️ Please run the analysis first in the 'Analysis' tab.")

with tab4:
    st.header("ℹ️ Help & Instructions")
    
    st.markdown("""
    ### 🚀 Quick Start Guide
    
    #### 1. Define Structure (Structure Definition Tab)
    - **Nodes**: Define the nodal points of your beam structure (x, y coordinates)
    - **Properties**: Set material properties (Young's modulus E, Poisson's ratio ν, Shear modulus G)
      and cross-section geometry (rectangular, circular, I-beam, C-section, and more)
    - **Elements**: Connect nodes with beam elements, select element formulation, assign a property
      set, and choose the mesh refinement (number of sub-divisions per element)
    - **Constraints**: Apply boundary conditions — fix translations and/or rotations at nodes,
      or prescribe non-zero displacements
    - **Point Loads**: Apply concentrated forces or moments at nodes
    - **Distributed Loads**: Apply loads distributed along elements (constant, linear, or custom function)
    
    #### 2. Run Analysis (Analysis Tab)
    - Review the model summary (node/element counts, mesh node count)
    - Click **👁️ Preview Structure with Loads** to inspect the setup before solving
    - Click **🚀 Run Analysis** to perform the FEM calculation
    - View nodal displacements and reaction forces at constrained DOFs
    - Download displacements and reactions as CSV files
    
    #### 3. View Results (Results Tab)
    - **Force Diagrams**: Plot bending moment, shear force, or normal force diagrams
      with optional fill, color, and transparency controls
    - **Point Query**: Read the force/moment value at any position along the structure
    - **Normal Stress Distribution**: Examine the stress field across the cross-section
      at any cut position (cross-sectional view) and along the beam length (side view)
    
    #### 4. Save/Load Projects (Sidebar)
    - **Save**: Download the full project as a JSON file
    - **Load**: Upload a previously saved project to restore all inputs
    - **New**: Start a fresh project (clears all current data)
    
    ### 📝 Tips
    - Use the expanders to organize your input; each section can be collapsed
    - Validation messages will guide you through errors before running the analysis
    - The **Section Preview** diagram shows dimension labels for the selected cross-section type
    - The app is **unit-agnostic** — choose any consistent unit system (see table below)
    - For Reddy-Bickford elements, a fourth DOF per node (curvature) is automatically activated
    
    ### 🔧 Supported Features
    - **Element Formulations**: Euler-Bernoulli (2-node, 3-node), Timoshenko (2-node, 3-node),
      Reddy-Bickford (2-node)
    - **Section Types**: Rectangular bar/tube, Circular bar/tube, Trapezoidal bar/tube,
      Hexagonal bar/tube, I-beam, C-section, L-section, T-section, Z-section, Hat section, General
    - **Load Types**: Point loads and moments, distributed loads (constant, linear, custom functions
      of position x and element length L)
    - **Mesh Refinement**: Each element can be subdivided into multiple sub-elements for
      improved accuracy without manually adding intermediate nodes
    - **Analysis**: Linear static FEM with automatic assembly, boundary condition enforcement,
      and reaction force recovery
    - **Visualization**: Force diagrams, stress distributions, deformed/undeformed structure
      preview, and section geometry diagrams

    ---

    ### 📐 Unit Reference — Common Consistent Systems in Structural Engineering

    The tool is **unit-agnostic**: you may use any coherent set of units.
    The table below lists the most common choices; pick one column and use it throughout.
    All quantities in a single analysis must use the **same** unit system.

    | Quantity | SI (N, m) | SI (kN, m) | SI (N, mm) | SI (kN, mm) | US (kip, in) | US (kip, ft) |
    |---|---|---|---|---|---|---|
    | Length | m | m | mm | mm | in | ft |
    | Force | N | kN | N | kN | kip | kip |
    | Moment | N·m | kN·m | N·mm | kN·mm | kip·in | kip·ft |
    | Distributed load | N/m | kN/m | N/mm | kN/mm | kip/in | kip/ft |
    | Pressure / Stress | Pa (N/m²) | kPa | MPa (N/mm²) | GPa (kN/mm²) | ksi (kip/in²) | ksf (kip/ft²) |
    | Young's modulus E — Steel | 200 × 10⁹ Pa | 200 × 10⁶ kPa | 200 000 MPa | 200 GPa | 29 000 ksi | 4 176 000 ksf |
    | Young's modulus E — Aluminum | 70 × 10⁹ Pa | 70 × 10⁶ kPa | 70 000 MPa | 70 GPa | 10 000 ksi | 1 440 000 ksf |
    | Young's modulus E — Concrete | 30 × 10⁹ Pa | 30 × 10⁶ kPa | 30 000 MPa | 30 GPa | 4 350 ksi | 626 400 ksf |

    > **Note on other common combinations:** Some practitioners use **MPa with metres**
    > (i.e. stresses in MPa but lengths in m) or **kN with mm**. These are *not* coherent
    > because 1 MPa = 1 N/mm² ≠ 1 N/m², so a mixed setup will give wrong results.
    > Always verify that your force unit divided by your length unit squared equals your
    > stress/modulus unit before entering values.
    """)
