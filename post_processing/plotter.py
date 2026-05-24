import numpy as np
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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
        textfont=dict(size=14, color='black'),
        name='Nodes',
        hovertemplate='Node %{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>'
    ))
    
    # Calculate structure bounds for scaling
    x_range = max(node_xs) - min(node_xs)
    y_range = max(node_ys) - min(node_ys)
    # Use a minimum scale of 1.0 to handle structures with zero range (e.g., vertical/horizontal beams).
    # A factor of 0.15 (15% of the largest dimension) keeps arrows clearly visible across all structure sizes.
    scale = max(x_range, y_range, 1.0) * 0.15  # Scale for arrows and symbols
    
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
            dx = scale * 2 * np.sign(magnitude)
            dy = 0
        elif direction == 1:  # Y direction
            dx = 0
            dy = scale * 2 * np.sign(magnitude)
        else:  # Moment (direction == 2)
            # Draw a small arc for moment
            theta = np.linspace(0, 1.5*np.pi, 20)
            arc_r = scale * 0.5
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
            arrowsize=1.0,
            arrowwidth=4,
            arrowcolor='orange',
            text=f'{magnitude:.1f}N',
            font=dict(size=10, color='darkorange'),
            bgcolor='rgba(255,255,255,0.75)',
            bordercolor='rgba(255,140,0,0.6)',
            borderwidth=1,
            borderpad=2,
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

        def _mag_at(t):
            """Return load magnitude (N/m) at fractional position t ∈ [0, 1]."""
            if load_type == "constant":
                return magnitude_start if magnitude_start is not None else 0.0
            elif load_type == "linear":
                if magnitude_start is not None and magnitude_end is not None:
                    return magnitude_start + t * (magnitude_end - magnitude_start)
                return 0.0
            else:  # custom function
                if func_str:
                    try:
                        x_val = t * L
                        return float(eval(func_str, {"np": np, "x": x_val, "L": L}))
                    except Exception:
                        pass
                return 1.0  # fallback placeholder

        # Compute magnitudes along the element for contour and arrows
        n_contour = 40
        contour_ts = np.linspace(0, 1, n_contour)
        contour_mags = np.array([_mag_at(t) for t in contour_ts])

        # Normalise so max |magnitude| maps to scale * 0.7 arrow length
        max_abs_mag = np.max(np.abs(contour_mags))
        if max_abs_mag < 1e-10:
            max_abs_mag = 1.0

        norm_factor = scale * 0.7 / max_abs_mag

        # Draw filled load-distribution contour (envelope of the distributed load).
        # Tips are placed in the load direction from the beam axis,
        # so that arrows point FROM the beam element toward the envelope.
        axis_xs = x1 + contour_ts * (x2 - x1)
        axis_ys = y1 + contour_ts * (y2 - y1)
        tip_xs = axis_xs + load_dir_x * contour_mags * norm_factor
        tip_ys = axis_ys + load_dir_y * contour_mags * norm_factor

        # Filled polygon: axis → tips (forward) → axis (backward)
        poly_xs = list(axis_xs) + list(tip_xs[::-1])
        poly_ys = list(axis_ys) + list(tip_ys[::-1])

        direction_label = {"x": "Global X", "y": "Global Y", "l": "Local axial", "t": "Local transverse"}.get(direction, direction)

        fig.add_trace(go.Scatter(
            x=poly_xs,
            y=poly_ys,
            fill='toself',
            fillcolor='rgba(255,140,0,0.15)',
            line=dict(width=0),
            mode='lines',
            showlegend=False,
            hoverinfo='skip',
        ))

        # Contour outline along the tips
        fig.add_trace(go.Scatter(
            x=list(tip_xs),
            y=list(tip_ys),
            mode='lines',
            line=dict(color='darkorange', width=2),
            showlegend=False,
            hoverinfo='skip',
        ))

        # Invisible hover points along the contour for interactive magnitude display
        hover_ts = np.linspace(0, 1, 10)
        hover_mags = np.array([_mag_at(t) for t in hover_ts])
        hover_xs = x1 + hover_ts * (x2 - x1)
        hover_ys = y1 + hover_ts * (y2 - y1)
        hover_texts = [
            f'Distributed Load<br>Element: {element_id}<br>Type: {load_type}<br>'
            f'Direction: {direction_label}<br>Position: {t*100:.0f}%<br>Magnitude: {m:.3f} N/m'
            for t, m in zip(hover_ts, hover_mags)
        ]
        fig.add_trace(go.Scatter(
            x=hover_xs,
            y=hover_ys,
            mode='markers',
            marker=dict(color='darkorange', size=8, opacity=0),
            hovertemplate='%{text}<extra></extra>',
            text=hover_texts,
            showlegend=False,
            name=f'Dist. Load {element_id}',
        ))

        # Draw multiple arrows along the element to represent distributed load.
        # Each arrow has its head (arrowhead) at the envelope tip and its tail at the beam,
        # so the arrow points FROM the beam element toward the envelope (in the load direction).
        n_arrows = 5
        for i in range(n_arrows):
            t = (i + 0.5) / n_arrows
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)

            mag = _mag_at(t)
            if abs(mag) < 1e-10:
                continue

            # Arrow length proportional to actual magnitude (normalised)
            arrow_len = mag * norm_factor
            arrow_dx = load_dir_x * arrow_len
            arrow_dy = load_dir_y * arrow_len

            # head is at the envelope tip; tail is on the beam axis
            fig.add_annotation(
                x=px + arrow_dx, y=py + arrow_dy,
                ax=px, ay=py,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='darkorange',
                text='',
            )

        # Add magnitude labels at the contour tip (away from the beam) for the
        # start, middle, and end of the element so labels never overlap the beam.
        label_positions = [0.05, 0.5, 0.95] if load_type != "constant" else [0.5]
        for t_label in label_positions:
            px_l = x1 + t_label * (x2 - x1)
            py_l = y1 + t_label * (y2 - y1)
            mag_l = _mag_at(t_label)
            if abs(mag_l) < 1e-10:
                continue
            arrow_len_l = mag_l * norm_factor
            # Position the label at the envelope tip, slightly beyond it
            label_x = px_l + load_dir_x * arrow_len_l
            label_y = py_l + load_dir_y * arrow_len_l
            fig.add_annotation(
                x=label_x, y=label_y,
                showarrow=False,
                text=f'{mag_l:.1f}',
                font=dict(size=10, color='darkorange'),
                bgcolor='rgba(255,255,255,0.75)',
                bordercolor='rgba(255,140,0,0.6)',
                borderwidth=1,
                borderpad=2,
            )
    
    # Update layout
    fig.update_layout(
        title="Structure Preview - Nodes, Elements, Loads, and Constraints",
        xaxis_title="x",
        yaxis_title="y",
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

def find_position_on_structure(structure_results, x_global, y_global):
    """
    Find the closest point on the structure to a given global (x, y) position.

    Projects the query point onto every element segment (clamped to the segment
    endpoints) and returns information about the nearest hit.

    Args:
        structure_results: StructureResults object with mesh and element_results.
        x_global: Global x coordinate of the query point.
        y_global: Global y coordinate of the query point.

    Returns:
        dict with keys:
            element_result  – ElementResults object for the nearest element
            local_x         – position along that element (0 … L)
            proj_x, proj_y  – global coordinates of the projected (nearest) point
            distance        – Euclidean distance from the query point to the structure
            is_on_structure – True when the query point is within 5 % of the
                              overall structure scale from the nearest element
        Returns None if the mesh has no elements.
    """
    if not structure_results.element_results:
        return None

    all_xs = [n.x for n in structure_results.mesh.nodes]
    all_ys = [n.y for n in structure_results.mesh.nodes]
    x_range = max(all_xs) - min(all_xs) if len(all_xs) > 1 else 1.0
    y_range = max(all_ys) - min(all_ys) if len(all_ys) > 1 else 1.0
    structure_scale = max(x_range, y_range, 1.0)
    # 5 % of the overall structure size is a practical snap tolerance:
    # tight enough to reject clearly off-axis queries, loose enough to
    # tolerate small floating-point offsets or clicks near elements.
    tolerance = structure_scale * 0.05

    best = None
    best_dist = float('inf')

    for el_result in structure_results.element_results:
        n1 = el_result.element.node_start
        n2 = el_result.element.node_end
        x1, y1, x2, y2 = n1.x, n1.y, n2.x, n2.y
        L = el_result.length
        dx, dy = x2 - x1, y2 - y1

        if L < 1e-10:
            t = 0.0
        else:
            # Parametric projection: t is the scalar in [0,1] that minimises
            # the distance from the query point to the line defined by n1→n2.
            # Clamping to [0,1] constrains the projection to the segment.
            t = float(np.clip(
                ((x_global - x1) * dx + (y_global - y1) * dy) / (L * L),
                0.0, 1.0
            ))

        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        dist = float(np.sqrt((x_global - proj_x) ** 2 + (y_global - proj_y) ** 2))

        if dist < best_dist:
            best_dist = dist
            best = {
                "element_result": el_result,
                "local_x": t * L,
                "proj_x": proj_x,
                "proj_y": proj_y,
                "distance": dist,
                "is_on_structure": dist <= tolerance,
            }

    return best


def _auto_diagram_points_per_element(n_elements, target_total_points=1000, min_points=10, max_points=100):
    """Auto-scale points-per-element from a global target for the whole structure."""
    safe_elements = max(int(n_elements), 1)
    points = int(round(target_total_points / safe_elements))
    return max(min_points, min(max_points, points))


def plot_structure_diagram(
    structure_results,
    force_type="moment",
    n_points=None,
    scale=0.2,
    fill_diagram=False,
    fill_color="green",
    fill_opacity=0.2,
    show_subdivision_nodes=True,
    n_original_nodes=None,
    query_xy=None,
):
    """
    Interactive Plotly plot: structure in 2D with force diagram (moment, shear, normal) along each element.

    Performance notes
    -----------------
    The gradient along each element is rendered with a *single* Scatter trace that
    uses per-marker colouring (marker.color array).  This is up to ~30× faster than
    the previous approach of one trace per line-segment, making the diagram usable
    even with many subdivided elements.

    Parameters
    ----------
    structure_results : StructureResults
    force_type : {"moment", "shear", "normal"}
    n_points : int or None
        Number of sample points per element for the force diagram.
        When None, it is auto-scaled from a global target across the structure.
    scale : float
        Diagram amplitude as a fraction of the overall structure size.
    fill_diagram : bool
        When True, shade the area between the element axis and the force curve.
    fill_color : str
        Matplotlib colour name for the fill.
    fill_opacity : float
        Opacity (0–1) of the fill area.
    show_subdivision_nodes : bool
        When False, only nodes with id ≤ n_original_nodes are labelled.
    n_original_nodes : int or None
        Number of user-defined (non-subdivision) nodes.  Only used when
        show_subdivision_nodes is False.
    query_xy : tuple(float, float) or None
        If provided, a marker is drawn at the closest point on the structure and
        the force value there is annotated.  Use find_position_on_structure() to
        check validity before calling.
    """
    force_labels = {
        "moment": "Bending Moment",
        "shear": "Shear Force",
        "normal": "Normal Force"
    }
    label = force_labels.get(force_type, force_type)
    colorscale = "rainbow"

    fig = go.Figure()
    if n_points is None:
        n_points = _auto_diagram_points_per_element(len(structure_results.element_results))

    # --- Plot nodes -------------------------------------------------------
    for node in structure_results.mesh.nodes:
        # Optionally skip subdivision (intermediate) nodes
        if not show_subdivision_nodes and n_original_nodes is not None:
            if node.id > n_original_nodes:
                continue
        fig.add_trace(go.Scatter(
            x=[node.x], y=[node.y],
            mode='markers+text',
            marker=dict(color='black', size=8),
            text=[str(node.id)],
            textposition='top right',
            name=f'Node {node.id}',
            hoverinfo='text'
        ))

    # --- Gather all force values for global colour normalisation ----------
    all_vals = []
    for el_result in structure_results.element_results:
        L = el_result.length
        xs = np.linspace(0, L, n_points)
        if force_type == "moment":
            vals = np.array([el_result.bending_moment(xi) for xi in xs])
        elif force_type == "shear":
            vals = np.array([el_result.shear_force(xi) for xi in xs])
        else:
            vals = np.array([el_result.normal_force(xi) for xi in xs])
        all_vals.extend(vals)
    all_vals = np.array(all_vals)
    vmax_abs = np.max(np.abs(all_vals)) if np.max(np.abs(all_vals)) > 0 else 1.0
    vmin = float(np.min(all_vals))
    vmax = float(np.max(all_vals))

    # Overall structure bounds → consistent diagram amplitude
    all_node_xs = [n.x for n in structure_results.mesh.nodes]
    all_node_ys = [n.y for n in structure_results.mesh.nodes]
    x_range = max(all_node_xs) - min(all_node_xs)
    y_range = max(all_node_ys) - min(all_node_ys)
    structure_scale = max(x_range, y_range, 1.0)
    diagram_scale = scale * structure_scale

    # Pre-compute fill rgba once
    if fill_diagram:
        r, g, b = mcolors.to_rgb(fill_color)
        fill_rgba = f'rgba({int(r*255)},{int(g*255)},{int(b*255)},{fill_opacity})'

    # --- Plot elements and force diagrams ---------------------------------
    for el_result in structure_results.element_results:
        n1 = el_result.element.node_start
        n2 = el_result.element.node_end
        x1, y1 = n1.x, n1.y
        x2, y2 = n2.x, n2.y

        # Element axis (black line)
        fig.add_trace(go.Scatter(
            x=[x1, x2], y=[y1, y2],
            mode='lines',
            line=dict(color='black', width=2),
            name=f'Element {el_result.element.id}',
            hoverinfo='skip'
        ))

        L = el_result.length
        xs = np.linspace(0, L, n_points)
        if force_type == "moment":
            vals = np.array([el_result.bending_moment(xi) for xi in xs])
        elif force_type == "shear":
            vals = np.array([el_result.shear_force(xi) for xi in xs])
        else:
            vals = np.array([el_result.normal_force(xi) for xi in xs])
        vals_normalized = vals / vmax_abs

        dx = x2 - x1
        dy = y2 - y1
        perp = np.array([-dy, dx])
        norm_perp = np.linalg.norm(perp)
        perp = perp / norm_perp if norm_perp > 0 else np.array([0.0, 0.0])

        # Points along element axis (vectorised)
        ts = xs / L if L > 1e-10 else np.zeros_like(xs)
        pxs = x1 + ts * dx   # numpy array
        pys = y1 + ts * dy   # numpy array

        # Gradient: ONE trace with per-marker colour (replaces the old per-segment
        # trace loop – reduces trace count by ~n_points, a ≈30× speedup).
        fig.add_trace(go.Scatter(
            x=pxs.tolist(),
            y=pys.tolist(),
            mode='markers',
            marker=dict(
                size=7,
                color=vals,
                colorscale=colorscale,
                cmin=vmin,
                cmax=vmax,
                showscale=False,
            ),
            customdata=np.column_stack([vals]),
            hovertemplate=(
                f'x=%{{x:.3f}}, y=%{{y:.3f}}<br>'
                f'{label}=%{{customdata[0]:.3f}}'
                f'<extra></extra>'
            ),
            showlegend=False,
        ))

        # Fill diagram (already one trace per element – keep as-is)
        if fill_diagram:
            pxs_off = (pxs + vals_normalized * perp[0] * diagram_scale).tolist()
            pys_off = (pys + vals_normalized * perp[1] * diagram_scale).tolist()
            x_poly = pxs_off + [x2, x1]
            y_poly = pys_off + [y2, y1]
            fig.add_trace(go.Scatter(
                x=x_poly, y=y_poly,
                fill='toself',
                fillcolor=fill_rgba,
                line=dict(color='rgba(0,0,0,0)', width=0),
                hoverinfo='skip',
                showlegend=False,
                name=f'{label} Area'
            ))
            # Hover markers on the offset outline
            fig.add_trace(go.Scatter(
                x=pxs_off, y=pys_off,
                mode='markers',
                marker=dict(size=8, opacity=0, color='rgba(0,0,0,0)'),
                customdata=np.column_stack([vals, pxs, pys]),
                hovertemplate=(
                    f'x=%{{customdata[1]:.3f}}, y=%{{customdata[2]:.3f}}<br>'
                    f'{label}=%{{customdata[0]:.3f}}'
                    f'<extra></extra>'
                ),
                showlegend=False,
            ))

    # --- Query point marker -----------------------------------------------
    if query_xy is not None:
        hit = find_position_on_structure(structure_results, query_xy[0], query_xy[1])
        if hit is not None:
            er = hit["element_result"]
            lx = hit["local_x"]
            if force_type == "moment":
                force_val = er.bending_moment(lx)
            elif force_type == "shear":
                force_val = er.shear_force(lx)
            else:
                force_val = er.normal_force(lx)

            fig.add_trace(go.Scatter(
                x=[hit["proj_x"]], y=[hit["proj_y"]],
                mode='markers',
                marker=dict(color='red', size=14, symbol='x-open', line=dict(width=3)),
                name='Queried Point',
                hovertemplate=(
                    f'Queried Point<br>'
                    f'x=%{{x:.4f}}, y=%{{y:.4f}}<br>'
                    f'{label}={force_val:.4f}'
                    f'<extra></extra>'
                ),
                showlegend=True,
            ))
            fig.add_annotation(
                x=hit["proj_x"], y=hit["proj_y"],
                text=f'{label[:3]}={force_val:.3f}',
                showarrow=True,
                arrowhead=2,
                arrowcolor='red',
                font=dict(size=12, color='red'),
                bgcolor='rgba(255,255,255,0.85)',
                bordercolor='red',
                borderwidth=1,
                ax=20, ay=-30,
            )

    # --- Colour bar -------------------------------------------------------
    colorbar_vals = np.linspace(vmin, vmax, 100)
    fig.add_trace(go.Scatter(
        x=[None] * 100, y=[None] * 100,
        mode='markers',
        marker=dict(
            size=0.1,
            color=colorbar_vals,
            colorscale=colorscale,
            colorbar=dict(title=label),
            showscale=True,
        ),
        hoverinfo='none',
        showlegend=False,
    ))

    fig.update_layout(
        title=f"Structure with {label} Diagram",
        xaxis_title="x",
        yaxis_title="y",
        showlegend=False,
        width=900,
        height=600,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig

def plot_normal_stress_distribution(element_result, x, n_points=100, query_y=None):
    """
    Interactive Plotly plot: 2D contour of normal stress over the section shape at position x along the element.

    n_points is reduced from the previous default of 200 to 100 for faster rendering;
    the stress formula is also vectorised to avoid the slow nested Python loop.

    query_y: optional float – if provided, a horizontal marker line is drawn at that
             section-y position to highlight the queried point.
    """
    section = element_result.element.section
    if not hasattr(section, "xy_grid"):
        raise ValueError("Section type does not support 2D stress contour plotting.")

    X, Y, mask = section.xy_grid(n_points)
    N = element_result.normal_force(x)
    M = element_result.bending_moment(x)
    # Vectorised: compute stress everywhere, NaN outside the section
    SIGMA = np.where(mask, N / section.area - M * Y / section.inertia, np.nan)

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

    x_min, x_max = np.min(X_plot), np.max(X_plot)
    x_margin = 0.1 * (x_max - x_min)

    # Calculate neutral axis position
    if abs(M) > 1e-12:  # Avoid division by zero
        y_neutral = N * section.inertia / (M * section.area)
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

    # Draw query-y marker line if provided
    if query_y is not None:
        sigma_at_query = N / section.area - M * query_y / section.inertia
        fig.add_trace(go.Scatter(
            x=[x_min - x_margin, x_max + x_margin],
            y=[query_y, query_y],
            mode='lines',
            line=dict(color='red', dash='dot', width=2),
            hoverinfo='skip',
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[x_max + x_margin],
            y=[query_y],
            mode='text',
            text=[f"σ={sigma_at_query:.4f}"],
            textposition="middle right",
            showlegend=False,
            hoverinfo='skip',
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

def plot_normal_stress_side_view(element_result, x, n_points=30):
    """
    Interactive Plotly plot: Side view of normal stress distribution along the element height.
    Shows:
    - Horizontal line representing the beam element (side view)
    - Vertical line at cut position x
    - Stress profile with arrows showing stress magnitude and direction at different heights
    
    Args:
        element_result: ElementResults object
        x: Position along element where to show stress
        n_points: Number of points to sample along the section height
    
    Returns:
        Plotly figure object
    """
    section = element_result.element.section
    
    # Get normal force and moment at position x
    N = element_result.normal_force(x)
    M = element_result.bending_moment(x)
    
    # Determine section height range
    if hasattr(section, 'height'):
        # Rectangular or tube sections
        h = section.height
        y_min, y_max = -h/2, h/2
    elif hasattr(section, 'diameter'):
        # Circular sections
        d = section.diameter
        y_min, y_max = -d/2, d/2
    else:
        # Generic section - use a reasonable default
        y_min, y_max = -0.1, 0.1
    
    # Sample stress values at different heights
    y_values = np.linspace(y_min, y_max, n_points)
    sigma_values = np.array([section.normal_stress(N, M, y) for y in y_values])
    
    # Determine max stress for scaling arrows
    max_sigma = np.max(np.abs(sigma_values))
    if max_sigma < 1e-12:
        max_sigma = 1.0  # Avoid division by zero
    
    # Create figure
    fig = go.Figure()
    
    # Element length
    L = element_result.length
    
    # Draw beam element as horizontal line (side view)
    fig.add_trace(go.Scatter(
        x=[0, L],
        y=[0, 0],
        mode='lines',
        line=dict(color='black', width=4),
        name='Beam Element',
        hoverinfo='text',
        text=[f'Element start (x=0)', f'Element end (x={L:.3f})'],
        showlegend=False
    ))
    
    # Draw vertical line at cut position
    section_height = y_max - y_min
    fig.add_trace(go.Scatter(
        x=[x, x],
        y=[-section_height*0.6, section_height*0.6],
        mode='lines',
        line=dict(color='red', width=3, dash='dash'),
        name='Cut Position',
        hoverinfo='text',
        text=[f'Cut at x={x:.3f}', ''],
        showlegend=False
    ))
    
    # Colormap for stress values
    vmin = np.min(sigma_values)
    vmax = np.max(sigma_values)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap('rainbow')
    
    # Scale factor for arrows (proportional to element length)
    arrow_scale = L * 0.15
    
    # Draw stress profile with arrows at each height
    for i, (y, sigma) in enumerate(zip(y_values, sigma_values)):
        # Normalize stress for arrow length
        sigma_normalized = sigma / max_sigma if max_sigma > 0 else 0
        arrow_length = sigma_normalized * arrow_scale
        
        # Arrow color based on stress value
        color_rgba = cmap(norm(sigma))
        color_hex = mcolors.to_hex(color_rgba)
        
        # Arrow position: starts at cut line, extends based on stress
        x_start = x
        x_end = x + arrow_length
        
        # Draw arrow line
        fig.add_trace(go.Scatter(
            x=[x_start, x_end],
            y=[y, y],
            mode='lines',
            line=dict(color=color_hex, width=2),
            hoverinfo='text',
            text=[f'Height y={y:.4f}<br>Stress σ={sigma:.3f}'],
            showlegend=False
        ))
        
        # Add arrowhead
        if abs(arrow_length) > 1e-6:
            # Arrowhead direction based on stress sign
            arrow_sign = np.sign(arrow_length)
            arrow_size = min(abs(arrow_length) * 0.15, L * 0.02)
            
            fig.add_annotation(
                x=x_end,
                y=y,
                ax=x_end - arrow_sign * arrow_size * 1.5,
                ay=y,
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor=color_hex
            )
    
    # Add text annotations for compression/tension
    if len(sigma_values) > 0:
        # Top fiber
        top_stress = sigma_values[-1]
        fig.add_annotation(
            x=L * 1.05,
            y=y_max,
            text=f"Top: σ={top_stress:.2f}",
            showarrow=False,
            font=dict(size=13, color='black'),
            bgcolor="rgba(255,255,255,0.8)"
        )
        
        # Bottom fiber
        bottom_stress = sigma_values[0]
        fig.add_annotation(
            x=L * 1.05,
            y=y_min,
            text=f"Bottom: σ={bottom_stress:.2f}",
            showarrow=False,
            font=dict(size=13, color='black'),
            bgcolor="rgba(255,255,255,0.8)"
        )
    
    # Calculate and show neutral axis if moment exists
    if abs(M) > 1e-12:
        y_neutral = N * section.inertia / (M * section.area)
        if y_min <= y_neutral <= y_max:
            # Draw neutral axis
            fig.add_trace(go.Scatter(
                x=[0, L],
                y=[y_neutral, y_neutral],
                mode='lines',
                line=dict(color='gray', dash='dot', width=2),
                name='Neutral Axis',
                hoverinfo='text',
                text=[f'Neutral Axis (y={y_neutral:.4f})', ''],
                showlegend=False
            ))
    
    # Add colorbar
    colorbar_vals = np.linspace(vmin, vmax, 100)
    fig.add_trace(go.Scatter(
        x=[None]*100,
        y=[None]*100,
        mode='markers',
        marker=dict(
            size=0.1,
            color=colorbar_vals,
            colorscale='rainbow',
            colorbar=dict(title="Normal Stress (σ)", x=1.15),
            showscale=True
        ),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Normal Stress Profile - Side View at x={x:.2f}",
        xaxis_title="Position along element",
        yaxis_title="Section height",
        width=900,
        height=500,
        showlegend=False,
        hovermode='closest'
    )
    
    return fig