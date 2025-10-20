import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.collections import LineCollection

def plot_structure_diagram(structure_results, force_type="moment", n_points=50, scale=1.0, ax=None):
    """
    Plots the structure in 2D and overlays the force diagram (moment, shear, or normal) along each element.
    The diagram is projected along the element's physical path, colored by force value.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
    else:
        fig = ax.figure

    # Choose label and color
    force_labels = {
        "moment": "Bending Moment",
        "shear": "Shear Force",
        "normal": "Normal Force"
    }
    colormaps = {
        "moment": plt.cm.coolwarm,
        "shear": plt.cm.coolwarm,
        "normal": plt.cm.coolwarm
    }
    label = force_labels.get(force_type, force_type)
    cmap = colormaps.get(force_type, plt.cm.viridis)

    # Plot nodes
    for node in structure_results.mesh.nodes:
        ax.plot(node.x, node.y, 'ko')
        ax.text(node.x, node.y, f"{node.id}", fontsize=8, ha='right', va='bottom')

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
        else:
            raise ValueError("Unknown force_type")
        all_vals.extend(vals)
    all_vals = np.array(all_vals)
    vmin, vmax = np.min(all_vals), np.max(all_vals)
    norm = plt.Normalize(vmin, vmax)

    # Plot elements and force diagrams
    for el_result in structure_results.element_results:
        n1 = el_result.element.node_start
        n2 = el_result.element.node_end
        x1, y1 = n1.x, n1.y
        x2, y2 = n2.x, n2.y
        # Draw element line
        ax.plot([x1, x2], [y1, y2], 'k-', lw=2)

        # Force diagram along element
        L = el_result.length
        xs = np.linspace(0, L, n_points)
        if force_type == "moment":
            vals = np.array([el_result.bending_moment(x) for x in xs])
        elif force_type == "shear":
            vals = np.array([el_result.shear_force(x) for x in xs])
        elif force_type == "normal":
            vals = np.array([el_result.normal_force(x) for x in xs])
        else:
            raise ValueError("Unknown force_type")

        # Normalize/scale for visualization
        if np.max(np.abs(all_vals)) > 0:
            vals_scaled = scale * vals / np.max(np.abs(all_vals))
        else:
            vals_scaled = vals

        # Local to global coordinates
        dx = x2 - x1
        dy = y2 - y1
        perp = np.array([-dy, dx])
        norm_perp = np.linalg.norm(perp)
        if norm_perp > 0:
            perp = perp / norm_perp
        else:
            perp = np.array([0.0, 0.0])

        # Build segments for colored line
        points = []
        colors = []
        for i in range(n_points):
            t = xs[i] / L
            px = x1 + t * dx
            py = y1 + t * dy
            px_off = px + vals_scaled[i] * perp[0] * 0.1
            py_off = py + vals_scaled[i] * perp[1] * 0.1
            points.append([px_off, py_off])
            colors.append(vals[i])
        points = np.array(points)
        # Create line segments
        segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=2)
        lc.set_array(np.array(colors[:-1]))
        ax.add_collection(lc)

    # Add a single colorbar for all elements
    fig.colorbar(lc, ax=ax, label=label, pad=0.02, aspect=30)

    ax.set_aspect('equal')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Structure with {label} Diagram")
    ax.grid(True)
    plt.tight_layout()
    return fig, ax

def plot_normal_stress_distribution(element_result, x, n_points=100, ax=None):
    """
    Plots a 2D contour of normal stress over the section shape at position x along the element.
    Works for any section with xy_grid() implemented.
    """
    section = element_result.element.section
    if not hasattr(section, "xy_grid"):
        raise ValueError("Section type does not support 2D stress contour plotting.")

    X, Y, mask = section.xy_grid(n_points)
    N = element_result.normal_force(x)
    M = element_result.bending_moment(x)
    SIGMA = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if mask[i, j]:
                SIGMA[i, j] = section.normal_stress(N, M, Y[i, j])
            else:
                SIGMA[i, j] = np.nan

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    c = ax.contourf(X, Y, SIGMA, levels=30, cmap='RdBu_r')
    fig.colorbar(c, ax=ax, label="Normal Stress")
    ax.set_xlabel("Section x")
    ax.set_ylabel("Section y")
    ax.set_title(f"Normal Stress Contour at x={x:.2f}")
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig, ax