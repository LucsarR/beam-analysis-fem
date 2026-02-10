import numpy as np
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Constants
STRESS_TOLERANCE = 1e-6  # Tolerance for detecting near-zero stress values

def plot_structure_preview(nodes, elements, properties, constraints, point_loads, distributed_loads):
    """
    Interactive Plotly plot: structure preview showing nodes, elements, loads, and constraints
    BEFORE analysis is run. This allows users to verify their setup before running the analysis.
    
    Args:
        nodes: List of (x, y) tuples representing node coordinates
        elements: List of (n1, n2, etype, prop_name, n_subdiv) tuples where n1, n2 are 1-based node IDs
        properties: List of property dicts with 'name', 'material', 'section', etc.
        constraints: List of (node_id, direction, value) tuples where direction is 0=x, 1=y, 2=rotation
        point_loads: List of (node_id, direction, magnitude) tuples where direction is 0=x, 1=y, 2=moment
        distributed_loads: List of (element_id, magnitude_start, magnitude_end, direction, func_str, load_type) tuples
    
    Returns:
        Plotly figure object
    
    Note:
        - Empty lists are handled gracefully (will show only available elements)
        - Node IDs in elements, constraints, and loads are 1-based indices
        - If an element references an invalid node ID, that element will be skipped
        - Scaling automatically adjusts for structures with zero range (e.g., vertical/horizontal beams)
    """
    fig = go.Figure()
    
    # Plot nodes
    node_xs = [node[0] for node in nodes]
    node_ys = [node[1] for node in nodes]
    node_ids = [str(i+1) for i in range(len(nodes))]
    
    fig.add_trace(go.Scatter(
        x=node_xs,
        y=node_ys,
        mode='markers+text',
        marker=dict(color='blue', size=12, symbol='circle'),
        text=node_ids,
        textposition='top center',
        name='Nodes',
        hovertemplate='Node %{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>'
    ))
    
    # Calculate structure bounds for scaling
    x_range = max(node_xs) - min(node_xs)
    y_range = max(node_ys) - min(node_ys)
    # Use a minimum scale of 1.0 to handle structures with zero range (e.g., vertical/horizontal beams)
    scale = max(x_range, y_range, 1.0) * 0.1  # Scale for arrows and symbols
    
    # Plot elements
    for i, (n1, n2, etype, prop_name, n_subdiv) in enumerate(elements):
        # Skip elements with invalid node IDs
        if n1 < 1 or n1 > len(nodes) or n2 < 1 or n2 > len(nodes):
            continue
        
        x1, y1 = nodes[n1-1]
        x2, y2 = nodes[n2-1]
        
        fig.add_trace(go.Scatter(
            x=[x1, x2],
            y=[y1, y2],
            mode='lines',
            line=dict(color='black', width=3),
            name=f'Element {i+1}',
            hovertemplate=f'Element {i+1}<br>Nodes: {n1} → {n2}<br>Type: {etype}<br>Property: {prop_name}<extra></extra>',
            showlegend=False
        ))
    
    # Plot constraints (boundary conditions)
    constraint_symbols = {0: 'triangle-right', 1: 'triangle-up', 2: 'circle'}  # x, y, rotation
    constraint_colors = {0: 'red', 1: 'green', 2: 'purple'}
    constraint_labels = {0: 'X-fixed', 1: 'Y-fixed', 2: 'Rotation-fixed'}
    
    for node_id, direction, value in constraints:
        # Skip constraints with invalid node IDs
        if node_id < 1 or node_id > len(nodes):
            continue
        
        x, y = nodes[node_id-1]
        
        # Offset constraint symbols slightly from node
        offset_x = scale * 0.3 * (1 if direction == 0 else 0)
        offset_y = scale * 0.3 * (1 if direction == 1 else 0)
        
        fig.add_trace(go.Scatter(
            x=[x - offset_x],
            y=[y - offset_y],
            mode='markers',
            marker=dict(
                color=constraint_colors.get(direction, 'gray'),
                size=15,
                symbol=constraint_symbols.get(direction, 'square'),
                line=dict(color='black', width=1)
            ),
            name=f'{constraint_labels.get(direction, "Unknown")}',
            hovertemplate=f'Constraint<br>Node: {node_id}<br>DOF: {constraint_labels.get(direction, "Unknown")}<br>Value: {value:.3f}<extra></extra>',
            showlegend=False
        ))
    
    # Plot point loads as arrows
    for node_id, direction, magnitude in point_loads:
        # Skip loads with invalid node IDs
        if node_id < 1 or node_id > len(nodes):
            continue
        
        x, y = nodes[node_id-1]
        
        # Determine arrow direction
        if direction == 0:  # X direction
            dx = scale * np.sign(magnitude)
            dy = 0
        elif direction == 1:  # Y direction
            dx = 0
            dy = scale * np.sign(magnitude)
        else:  # Moment (direction == 2)
            # Draw a small arc for moment
            theta = np.linspace(0, 1.5*np.pi, 20)
            arc_r = scale * 0.4
            arc_x = x + arc_r * np.cos(theta)
            arc_y = y + arc_r * np.sin(theta)
            
            fig.add_trace(go.Scatter(
                x=arc_x,
                y=arc_y,
                mode='lines',
                line=dict(color='orange', width=3),
                name=f'Moment Load',
                hovertemplate=f'Moment Load<br>Node: {node_id}<br>Magnitude: {magnitude:.3f}<extra></extra>',
                showlegend=False
            ))
            
            # Add arrowhead for moment
            arrow_x = arc_x[-1]
            arrow_y = arc_y[-1]
            arrow_dx = (arc_x[-1] - arc_x[-2]) * 2
            arrow_dy = (arc_y[-1] - arc_y[-2]) * 2
            
            fig.add_annotation(
                x=arrow_x, y=arrow_y,
                ax=arrow_x - arrow_dx, ay=arrow_y - arrow_dy,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=3,
                arrowcolor='orange'
            )
            continue
        
        # For X and Y loads, draw arrow
        fig.add_annotation(
            x=x, y=y,
            ax=x - dx, ay=y - dy,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=4,
            arrowcolor='orange',
            text=f'{magnitude:.1f}N',
            font=dict(size=10, color='orange'),
            bgcolor='rgba(255,255,255,0.7)'
        )
    
    # Plot distributed loads
    for element_id, magnitude_start, magnitude_end, direction, func_str, load_type in distributed_loads:
        # Find the element
        if element_id > len(elements):
            continue
        
        n1, n2, _, _, _ = elements[element_id-1]
        x1, y1 = nodes[n1-1]
        x2, y2 = nodes[n2-1]
        
        # Calculate element properties
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if L < 1e-10:
            continue
        
        # Element direction
        dx_elem = (x2 - x1) / L
        dy_elem = (y2 - y1) / L
        
        # Perpendicular direction (for transverse loads)
        perp_x = -dy_elem
        perp_y = dx_elem
        
        # Determine load direction vector
        if direction == 'x':
            load_dir_x, load_dir_y = 1.0, 0.0
        elif direction == 'y':
            load_dir_x, load_dir_y = 0.0, 1.0
        elif direction == 'l':  # local axial
            load_dir_x, load_dir_y = dx_elem, dy_elem
        elif direction == 't':  # local transverse
            load_dir_x, load_dir_y = perp_x, perp_y
        else:
            load_dir_x, load_dir_y = 0.0, 1.0
        
        # Draw multiple arrows along the element to represent distributed load
        n_arrows = 5
        for i in range(n_arrows):
            t = (i + 0.5) / n_arrows
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)
            
            # Determine magnitude at this position
            if load_type == "constant":
                mag = magnitude_start if magnitude_start is not None else 0.0
            elif load_type == "linear":
                # Linear interpolation between start and end magnitude
                if magnitude_start is not None and magnitude_end is not None:
                    mag = magnitude_start + t * (magnitude_end - magnitude_start)
                else:
                    mag = 0.0
            else:  # custom function
                # For preview, just show arrows - actual magnitude calculation requires eval
                mag = 1.0  # placeholder
            
            # Arrow length proportional to magnitude
            arrow_scale = scale * 0.5 * np.sign(mag) if mag != 0 else scale * 0.5
            arrow_dx = load_dir_x * arrow_scale
            arrow_dy = load_dir_y * arrow_scale
            
            fig.add_annotation(
                x=px, y=py,
                ax=px - arrow_dx, ay=py - arrow_dy,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='darkorange'
            )
    
    # Update layout
    fig.update_layout(
        title="Structure Preview - Nodes, Elements, Loads, and Constraints",
        xaxis_title="x (m)",
        yaxis_title="y (m)",
        showlegend=True,
        width=900,
        height=600,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # Equal aspect ratio
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    return fig

def plot_structure_diagram(structure_results, force_type="moment", n_points=50, scale=0.2, fill_diagram=False):
    """
    Interactive Plotly plot: structure in 2D with force diagram (moment, shear, normal) along each element.
    The diagram is projected along the element's physical path, colored by force value using a true gradient line.
    """
    force_labels = {
        "moment": "Bending Moment",
        "shear": "Shear Force",
        "normal": "Normal Force"
    }
    label = force_labels.get(force_type, force_type)
    colorscale = "rainbow"

    fig = go.Figure()

    # Plot nodes
    for node in structure_results.mesh.nodes:
        fig.add_trace(go.Scatter(
            x=[node.x], y=[node.y],
            mode='markers+text',
            marker=dict(color='black', size=8),
            text=[str(node.id)],
            textposition='top right',
            name=f'Node {node.id}',
            hoverinfo='text'
        ))

    # Gather all force values for normalization
    all_vals = []
    for el_result in structure_results.element_results:
        L = el_result.length
        xs = np.linspace(0, L, n_points)
        if force_type == "moment":
            vals = np.array([el_result.bending_moment(x) for x in xs])
        elif force_type == "shear":
            vals = np.array([el_result.shear_force(x) for x in xs])
        elif force_type == "normal":
            vals = np.array([el_result.normal_force(x) for x in xs])
        all_vals.extend(vals)
    all_vals = np.array(all_vals)
    vmax_abs = np.max(np.abs(all_vals)) if np.max(np.abs(all_vals)) > 0 else 1.0

    # Plot elements and force diagrams
    for el_result in structure_results.element_results:
        n1 = el_result.element.node_start
        n2 = el_result.element.node_end
        x1, y1 = n1.x, n1.y
        x2, y2 = n2.x, n2.y
        fig.add_trace(go.Scatter(
            x=[x1, x2], y=[y1, y2],
            mode='lines',
            line=dict(color='black', width=2),
            name=f'Element {el_result.element.id}',
            hoverinfo='skip'
        ))

        # Force diagram along element
        L = el_result.length
        xs = np.linspace(0, L, n_points)
        if force_type == "moment":
            vals = np.array([el_result.bending_moment(x) for x in xs])
        elif force_type == "shear":
            vals = np.array([el_result.shear_force(x) for x in xs])
        elif force_type == "normal":
            vals = np.array([el_result.normal_force(x) for x in xs])
        vals_normalized = vals / vmax_abs
        diagram_scale = scale * L

        dx = x2 - x1
        dy = y2 - y1
        perp = np.array([-dy, dx])
        norm_perp = np.linalg.norm(perp)
        if norm_perp > 0:
            perp = perp / norm_perp
        else:
            perp = np.array([0.0, 0.0])

        # 1. Gradient line ON the element (no offset)
        pxs = []
        pys = []
        for i in range(n_points):
            t = xs[i] / L
            px = x1 + t * dx
            py = y1 + t * dy
            pxs.append(px)
            pys.append(py)

        vmin = np.min(all_vals)
        vmax = np.max(all_vals)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap(colorscale)
        for i in range(n_points - 1):
            color_rgba = cmap(norm(vals[i]))
            color_hex = mcolors.to_hex(color_rgba)
            fig.add_trace(go.Scatter(
                x=[pxs[i], pxs[i+1]],
                y=[pys[i], pys[i+1]],
                mode='lines',
                line=dict(color=color_hex, width=4),
                hoverinfo='text',
                text=[
                    f"x={pxs[i]:.3f}, y={pys[i]:.3f}<br>{label}={vals[i]:.3f}",
                    f"x={pxs[i+1]:.3f}, y={pys[i+1]:.3f}<br>{label}={vals[i+1]:.3f}"
                ],
                showlegend=False
            ))

        # 2. If fill_diagram, plot the area above the element (offset)
        if fill_diagram:
            pxs_off = []
            pys_off = []
            for i in range(n_points):
                t = xs[i] / L
                px = x1 + t * dx
                py = y1 + t * dy
                px_off = px + vals_normalized[i] * perp[0] * diagram_scale
                py_off = py + vals_normalized[i] * perp[1] * diagram_scale
                pxs_off.append(px_off)
                pys_off.append(py_off)
            # Polygon: offset curve + element (back)
            x_poly = pxs_off + [x2, x1]
            y_poly = pys_off + [y2, y1]
            fig.add_trace(go.Scatter(
                x=x_poly,
                y=y_poly,
                fill='toself',
                fillcolor='rgba(0,0,255,0.2)' if force_type == "moment" else 'rgba(255,140,0,0.2)' if force_type == "shear" else 'rgba(0,128,0,0.2)',
                line=dict(color='rgba(0,0,0,0)', width=0),
                hoverinfo='skip',
                showlegend=False,
                name=f'{label} Area'
            ))

    # Add a colorbar using a dummy invisible scatter
    colorbar_vals = np.linspace(np.min(all_vals), np.max(all_vals), 100)
    fig.add_trace(go.Scatter(
        x=[None]*100, y=[None]*100,
        mode='markers',
        marker=dict(
            size=0.1,
            color=colorbar_vals,
            colorscale=colorscale,
            colorbar=dict(title=label),
            showscale=True
        ),
        hoverinfo='none',
        showlegend=False
    ))

    fig.update_layout(
        title=f"Structure with {label} Diagram",
        xaxis_title="x",
        yaxis_title="y",
        showlegend=False,
        width=900,
        height=600
    )
    return fig

def plot_normal_stress_distribution(element_result, x, n_points=200):
    """
    Interactive Plotly plot: 2D contour of normal stress over the section shape at position x along the element.
    """
    section = element_result.element.section
    if not hasattr(section, "xy_grid"):
        raise ValueError("Section type does not support 2D stress contour plotting.")

    X, Y, mask = section.xy_grid(n_points)
    N = element_result.normal_force(x)
    M = element_result.bending_moment(x)
    SIGMA = np.full_like(X, np.nan)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if mask[i, j]:
                SIGMA[i, j] = section.normal_stress(N, M, Y[i, j])

    # Mask out-of-section points
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    SIGMA_flat = SIGMA.flatten()
    mask_flat = mask.flatten()
    X_plot = X_flat[mask_flat]
    Y_plot = Y_flat[mask_flat]
    SIGMA_plot = SIGMA_flat[mask_flat]

    fig = go.Figure(data=go.Scatter(
        x=X_plot,
        y=Y_plot,
        mode='markers',
        marker=dict(
            size=6,
            color=SIGMA_plot,
            colorscale="Rainbow",  # Use rainbow for cross-section
            colorbar=dict(title="Normal Stress"),
            showscale=True
        ),
        text=[f"x={xv:.3f}<br>y={yv:.3f}<br>σ={sv:.3f}" for xv, yv, sv in zip(X_plot, Y_plot, SIGMA_plot)],
        hoverinfo='text'
    ))

    # Calculate neutral axis position
    if abs(M) > 1e-12:  # Avoid division by zero
        y_neutral = N * section.inertia / (M * section.area)
        x_min, x_max = np.min(X_plot), np.max(X_plot)
        x_margin = 0.1 * (x_max - x_min)
        # Dashed line
        fig.add_trace(go.Scatter(
            x=[x_min - x_margin, x_max + x_margin],
            y=[y_neutral, y_neutral],
            mode='lines',
            line=dict(color='black', dash='dash', width=3),
            hoverinfo='skip',
            showlegend=False
        ))
        # Text annotation at the right end
        fig.add_trace(go.Scatter(
            x=[x_max + x_margin],
            y=[y_neutral],
            mode='text',
            text=["Neutral Axis"],
            textposition="middle right",
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        title=f"Normal Stress Contour at x={x:.2f}",
        xaxis_title="Section x",
        yaxis_title="Section y",
        width=600,
        height=500,
        showlegend=False
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

def plot_normal_stress_side_view(element_result, n_points=50):
    """
    Interactive Plotly plot: Side view of normal stress distribution along the beam element.
    Shows stress at top fiber, bottom fiber, and neutral axis along the element length.
    
    Args:
        element_result: ElementResult object containing the element and stress functions
        n_points: Number of points along the element length
    
    Returns:
        Plotly figure object
    """
    section = element_result.element.section
    L = element_result.length
    
    # Get section bounds
    y_min, y_max = section.get_y_bounds()
    
    # Sample positions along element
    xs = np.linspace(0, L, n_points)
    
    # Calculate stresses at each position
    stress_top = []
    stress_bottom = []
    stress_neutral = []
    
    for x in xs:
        N = element_result.normal_force(x)
        M = element_result.bending_moment(x)
        
        # Stress at top fiber (y_max)
        sigma_top = section.normal_stress(N, M, y_max)
        stress_top.append(sigma_top)
        
        # Stress at bottom fiber (y_min)
        sigma_bottom = section.normal_stress(N, M, y_min)
        stress_bottom.append(sigma_bottom)
        
        # Stress at neutral axis (y=0 for symmetric sections, or centroid)
        sigma_neutral = section.normal_stress(N, M, 0.0)
        stress_neutral.append(sigma_neutral)
    
    # Convert to numpy arrays
    stress_top = np.array(stress_top)
    stress_bottom = np.array(stress_bottom)
    stress_neutral = np.array(stress_neutral)
    
    # Create figure
    fig = go.Figure()
    
    # Add beam element baseline (x-axis represents element, y-axis represents stress)
    fig.add_trace(go.Scatter(
        x=xs,
        y=np.zeros_like(xs),
        mode='lines',
        line=dict(color='black', width=2),
        name='Element axis',
        hoverinfo='skip'
    ))
    
    # Add stress at top fiber
    fig.add_trace(go.Scatter(
        x=xs,
        y=stress_top,
        mode='lines+markers',
        line=dict(color='red', width=2),
        marker=dict(size=4),
        name=f'Top fiber (y={y_max:.3f})',
        hovertemplate='x=%{x:.3f} m<br>σ=%{y:.3f} MPa<extra></extra>'
    ))
    
    # Add stress at bottom fiber
    fig.add_trace(go.Scatter(
        x=xs,
        y=stress_bottom,
        mode='lines+markers',
        line=dict(color='blue', width=2),
        marker=dict(size=4),
        name=f'Bottom fiber (y={y_min:.3f})',
        hovertemplate='x=%{x:.3f} m<br>σ=%{y:.3f} MPa<extra></extra>'
    ))
    
    # Add stress at neutral axis (if it varies)
    if not np.allclose(stress_neutral, 0.0, atol=STRESS_TOLERANCE):
        fig.add_trace(go.Scatter(
            x=xs,
            y=stress_neutral,
            mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name='Centroid (y=0)',
            hovertemplate='x=%{x:.3f} m<br>σ=%{y:.3f} MPa<extra></extra>'
        ))
    
    # Fill area between top and bottom stress
    fig.add_trace(go.Scatter(
        x=np.concatenate([xs, xs[::-1]]),
        y=np.concatenate([stress_top, stress_bottom[::-1]]),
        fill='toself',
        fillcolor='rgba(128, 128, 128, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        name='Stress envelope',
        showlegend=False
    ))
    
    fig.update_layout(
        title="Normal Stress Distribution - Side View",
        xaxis_title="Position along element (m)",
        yaxis_title="Normal Stress σ (MPa)",
        width=900,
        height=500,
        showlegend=True,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig