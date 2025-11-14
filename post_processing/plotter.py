import numpy as np
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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