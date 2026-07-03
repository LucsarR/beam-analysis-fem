import numpy as np
from fem.analysis import get_element_dof_indices
from fem.element import _quadratic_shape_functions_3node, _quintic_bending_shapes_3node

class ElementResults:
    """
    Stores and computes results for a single element: bending moment, shear force, normal force.
    """
    def __init__(self, element, displacements):
        self.element = element
        self.displacements = displacements
        self.length = element.length
        self.compute_forces()

    def compute_forces(self):
        # Compute normal force, shear force, bending moment along the element
        # Example for 2-node Euler-Bernoulli:
        # Use shape functions and element DOF to calculate at any x in [0, L]
        pass

    def bending_moment(self, x):
        # Return bending moment at position x along the element
        return self.element.bending_moment(x, self.displacements)

    def shear_force(self, x):
        # Return shear force at position x along the element
        return self.element.shear_force(x, self.displacements)

    def normal_force(self, x):
        # Return normal force at position x along the element
        return self.element.normal_force(x, self.displacements)

    def axial_displacement(self, x):
        """Return local axial displacement u(x) at position x along the element."""
        L = self.length
        xi = x / L if L > 1e-14 else 0.0
        d = self.displacements
        class_name = type(self.element).__name__

        if "3Node" in class_name:
            # 9 DOFs: [u1, v1, θ1, u2, v2, θ2, u3, v3, θ3]
            u1, u2, u3 = d[0], d[3], d[6]
            n1, n2, n3 = _quadratic_shape_functions_3node(xi)
            return float(n1 * u1 + n2 * u2 + n3 * u3)
        elif "ReddyBickford" in class_name or "MRBT" in class_name:
            # 8 DOFs: [u1, v1, θ1, (dv/dx)1, u2, v2, θ2, (dv/dx)2]
            u1, u2 = d[0], d[4]
            return float((1.0 - xi) * u1 + xi * u2)
        else:
            # 6 DOFs: [u1, v1, θ1, u2, v2, θ2]
            u1, u2 = d[0], d[3]
            return float((1.0 - xi) * u1 + xi * u2)

    def transverse_displacement(self, x):
        """Return local transverse displacement v(x) at position x along the element."""
        L = self.length
        xi = x / L if L > 1e-14 else 0.0
        d = self.displacements
        class_name = type(self.element).__name__

        if "3Node" in class_name:
            # 9 DOFs: [u1, v1, θ1, u2, v2, θ2, u3, v3, θ3] – quintic Hermite
            bending_dofs = np.array([d[1], d[2], d[4], d[5], d[7], d[8]])
            n_w, _, _, _, _, _ = _quintic_bending_shapes_3node(xi, L)
            return float(np.dot(n_w, bending_dofs))
        elif "ReddyBickford" in class_name or "MRBT" in class_name:
            # 8 DOFs: [u1, v1, θ1, (dv/dx)1, u2, v2, θ2, (dv/dx)2]
            # Cubic Hermite using the (dv/dx) DOFs at each node
            v1, dvdx1 = d[1], d[3]
            v2, dvdx2 = d[5], d[7]
            H1 = 1.0 - 3.0 * xi**2 + 2.0 * xi**3
            H2 = L * xi * (1.0 - xi)**2
            H3 = 3.0 * xi**2 - 2.0 * xi**3
            H4 = L * xi**2 * (xi - 1.0)
            return float(H1 * v1 + H2 * dvdx1 + H3 * v2 + H4 * dvdx2)
        else:
            # 6 DOFs: [u1, v1, θ1, u2, v2, θ2] – cubic Hermite
            v1, theta1 = d[1], d[2]
            v2, theta2 = d[4], d[5]
            H1 = 1.0 - 3.0 * xi**2 + 2.0 * xi**3
            H2 = L * xi * (1.0 - xi)**2
            H3 = 3.0 * xi**2 - 2.0 * xi**3
            H4 = L * xi**2 * (xi - 1.0)
            return float(H1 * v1 + H2 * theta1 + H3 * v2 + H4 * theta2)

    def kinematic_normal_stress(self, x, y):
        """
        Compute the kinematic normal stress sigma_xx(x, y) directly from the
        displacement fields and kinematic relations of the specific beam theory.
        """
        el = self.element
        d = self.displacements
        E = el.material.E
        L = el.length
        xi = x / L if L > 1e-14 else 0.0
        class_name = type(el).__name__

        if class_name == "EulerBernoulliElement2Node":
            # DOFs: [u1, v1, theta1, u2, v2, theta2]
            u1, u2 = d[0], d[3]
            v1, theta1, v2, theta2 = d[1], d[2], d[4], d[5]
            
            # du/dx (linear/constant)
            du_dx = (u2 - u1) / L
            
            # d^2w/dx^2 (linear)
            d2N1 = (12.0 / L**2) * (xi - 0.5)
            d2N2 = (6.0 / L) * (xi - 2.0/3.0)
            d2N3 = (-12.0 / L**2) * (xi - 0.5)
            d2N4 = (6.0 / L) * (xi - 1.0/3.0)
            d2w_dx2 = d2N1 * v1 + d2N2 * theta1 + d2N3 * v2 + d2N4 * theta2
            
            # sigma = E * (du/dx - y * d2w/dx2)
            stress = E * (du_dx - y * d2w_dx2)

        elif class_name == "EulerBernoulliElement3Node":
            # DOFs: [u1, v1, theta1, u2, v2, theta2, u3, v3, theta3]
            u1, u2, u3 = d[0], d[3], d[6]
            d_bending = d[[1, 2, 4, 5, 7, 8]]
            
            # du/dx (linear)
            dN1_dxi = -3.0 + 4.0 * xi
            dN2_dxi = 4.0 - 8.0 * xi
            dN3_dxi = -1.0 + 4.0 * xi
            du_dx = (1.0 / L) * (dN1_dxi * u1 + dN2_dxi * u2 + dN3_dxi * u3)
            
            # d^2w/dx^2 (cubic)
            _, _, d2_w_dx2, _, _, _ = _quintic_bending_shapes_3node(xi, L)
            d2w_dx2 = np.dot(d2_w_dx2, d_bending)
            
            stress = E * (du_dx - y * d2w_dx2)

        elif class_name == "TimoshenkoElement2Node":
            # DOFs: [u1, v1, theta1, u2, v2, theta2]
            u1, u2 = d[0], d[3]
            theta1, theta2 = d[2], d[5]
            
            # du/dx (linear/constant)
            du_dx = (u2 - u1) / L
            
            # dtheta/dx (linear/constant)
            dtheta_dx = (theta2 - theta1) / L
            
            # sigma = E * (du/dx + y * dtheta/dx)
            stress = E * (du_dx + y * dtheta_dx)

        elif class_name == "TimoshenkoElement3Node":
            # DOFs: [u1, v1, theta1, u2, v2, theta2, u3, v3, theta3]
            u1, u2, u3 = d[0], d[3], d[6]
            theta1, theta2, theta3 = d[2], d[5], d[8]
            
            # du/dx (linear)
            dN1_dxi = -3.0 + 4.0 * xi
            dN2_dxi = 4.0 - 8.0 * xi
            dN3_dxi = -1.0 + 4.0 * xi
            du_dx = (1.0 / L) * (dN1_dxi * u1 + dN2_dxi * u2 + dN3_dxi * u3)
            
            # dtheta/dx (linear)
            dtheta_dx = (1.0 / L) * (dN1_dxi * theta1 + dN2_dxi * theta2 + dN3_dxi * theta3)
            
            # sigma = E * (du/dx + y * dtheta/dx)
            stress = E * (du_dx + y * dtheta_dx)

        elif class_name in ["ReddyBickfordElement2Node", "MRBTElement2Node"]:
            # DOFs: [u1, v1, theta1, dv_dx1, u2, v2, theta2, dv_dx2]
            u1, u2 = d[0], d[4]
            v1, dv_dx1 = d[1], d[3]
            theta1, theta2 = d[2], d[6]
            v2, dv_dx2 = d[5], d[7]
            
            # du/dx (linear/constant)
            du_dx = (u2 - u1) / L
            
            # dtheta/dx (linear/constant)
            dtheta_dx = (theta2 - theta1) / L
            
            # d^2v/dx^2 (linear)
            H1_pp = 6.0 * (2.0 * xi - 1.0) / L**2
            H2_pp = 2.0 * (3.0 * xi - 2.0) / L
            H3_pp = 6.0 * (1.0 - 2.0 * xi) / L**2
            H4_pp = 2.0 * (3.0 * xi - 1.0) / L
            d2v_dx2 = H1_pp * v1 + H2_pp * dv_dx1 + H3_pp * v2 + H4_pp * dv_dx2
            
            # Reddy-Bickford parameter c1 = 4 / (3 * h^2)
            h = el._get_height()
            c1 = 4.0 / (3.0 * h**2)
            
            # epsilon_xx = du/dx + y * dtheta/dx - c1 * y^3 * (dtheta/dx + d^2v/dx^2)
            epsilon_xx = du_dx + y * dtheta_dx - c1 * y**3 * (dtheta_dx + d2v_dx2)
            
            stress = E * epsilon_xx

        else:
            # Fallback to resultant normal stress using section formula
            N = self.normal_force(x)
            M = self.bending_moment(x)
            stress = self.element.section.normal_stress(N, M, y)

        if np.ndim(stress) == 0:
            return float(stress)
        return stress

    def _reddy_gamma_factor(self, x):
        """
        Compute -(θ(x) + dv₀/dx(x)) using shape function interpolation.
        Returns 0.0 if not a Reddy element.
        """
        class_name = type(self.element).__name__
        if "ReddyBickford" not in class_name and "MRBT" not in class_name:
            return 0.0
            
        # displacements in LOCAL coordinates:
        local_disps = self.displacements
        
        theta = self.element.interpolate_theta(x, local_disps)
        dv_dx = self.element.interpolate_dv_dx(x, local_disps)
        
        return - (theta + dv_dx)

    def reddy_shear_stress(self, x, y):
        """
        Compute τ_xy(x,y) = G·(θ(x) − dv₀/dx(x))·(3αy² − 1) for Reddy elements.
        """
        class_name = type(self.element).__name__
        if "ReddyBickford" not in class_name and "MRBT" not in class_name:
            return 0.0
            
        G = self.element.material.G
        h = self.element._get_height()
        alpha = 4.0 / (3.0 * h**2)
        
        gamma = self._reddy_gamma_factor(x)
        # Negate the sign to align with Jourawski shear stress convention
        return - G * gamma * (3.0 * alpha * y**2 - 1.0)

    def jourawski_shear_stress(self, x, y_val, n_points=100):
        """
        Compute approximate shear stress τ_xy(x, y_val) using Jourawski theory:
        τ = V · Q / (I · b)
        """
        section = self.element.section
        if not hasattr(section, "xy_grid"):
            return 0.0
        
        V = self.shear_force(x)
        if section.inertia is None or abs(section.inertia) < 1e-18:
            return 0.0

        cache_key = f"_jourawski_cache_{n_points}"
        if not hasattr(self, cache_key):
            X, Y, mask = section.xy_grid(n_points)
            if X is None or Y is None or mask is None:
                setattr(self, cache_key, None)
                return 0.0
                
            n_rows = Y.shape[0]
            if n_rows > 1:
                dy = abs(float(np.mean(np.diff(Y[:, 0]))))
            else:
                dy = 1.0

            row_width = np.zeros(n_rows)
            row_first_moment = np.zeros(n_rows)
            valid_rows = np.zeros(n_rows, dtype=bool)

            for i in range(n_rows):
                row_mask = mask[i, :]
                if not np.any(row_mask):
                    continue
                xi = np.sort(X[i, row_mask])
                if xi.size > 1:
                    dx = float(np.mean(np.diff(xi)))
                else:
                    x_full = X[i, :]
                    dx = float((np.max(x_full) - np.min(x_full)) / max(len(x_full) - 1, 1))
                if dx <= 0:
                    continue

                row_area = row_mask.sum() * dx * dy
                row_y = float(np.mean(Y[i, row_mask]))
                row_width[i] = row_mask.sum() * dx
                row_first_moment[i] = row_y * row_area
                valid_rows[i] = True

            Q = np.zeros(n_rows)
            running_Q = 0.0
            for i in range(n_rows - 1, -1, -1):
                if not valid_rows[i]:
                    continue
                running_Q += row_first_moment[i]
                Q[i] = running_Q

            # Find the row closest to y_val
            row_ys = np.zeros(n_rows)
            for i in range(n_rows):
                if valid_rows[i]:
                    row_ys[i] = float(np.mean(Y[i, mask[i, :]]))
                else:
                    row_ys[i] = Y[i, 0]

            setattr(self, cache_key, {
                'Q': Q,
                'row_width': row_width,
                'row_ys': row_ys,
                'valid_rows': valid_rows
            })

        cache = getattr(self, cache_key)
        if cache is None:
            return 0.0

        Q = cache['Q']
        row_width = cache['row_width']
        row_ys = cache['row_ys']
        valid_rows = cache['valid_rows']

        valid_y_coords = [row_ys[i] for i in range(len(row_ys)) if valid_rows[i]]
        if not valid_y_coords:
            return 0.0
        min_y, max_y = min(valid_y_coords), max(valid_y_coords)
        h = max_y - min_y
        if y_val < min_y - 0.01 * h or y_val > max_y + 0.01 * h:
            return 0.0

        idx = np.argmin(np.abs(row_ys - y_val))
        if valid_rows[idx] and row_width[idx] > 1e-12:
            return V * Q[idx] / (section.inertia * row_width[idx])
        return 0.0

class StructureResults:
    """
    Manages results for all elements in the mesh.
    """
    def __init__(self, mesh, displacements, reactions=None, dpn=None):
        self.mesh = mesh
        self.displacements = displacements
        self.reactions = reactions  # Dictionary: {(node_id, direction): reaction_force}
        if dpn is None:
            # Keep post-processing DOF extraction consistent with analysis assembly.
            self.dpn = max([getattr(el, "dofs_per_node", 3) for el in mesh.elements], default=3)
        else:
            self.dpn = dpn  # Global degrees of freedom per node
        self.element_results = [
            ElementResults(el, self._get_element_dofs(el)) for el in mesh.elements
        ]

    def _get_element_dofs(self, element):
        """Extract DOFs for the element from global displacement vector.

        Uses the same logic as BeamAnalysis._get_element_dof_indices to ensure
        consistency with the assembly process.
        """
        dof_indices = get_element_dof_indices(element, self.dpn)
        global_disp = self.displacements[dof_indices]
        # Transform to local coordinates
        R = element.R
        local_disp = R.T @ global_disp
        return local_disp

    def _find_element_at_x(self, x):
        """Find the element containing global coordinate x (for 1D beams).

        Returns (ElementResults, local_x) where local_x is the coordinate
        within the element measured from node_start.
        """
        for er in self.element_results:
            el = er.element
            x0 = el.node_start.x
            x1 = el.node_end.x
            x_lo, x_hi = min(x0, x1), max(x0, x1)
            if x_lo <= x <= x_hi + 1e-10:
                return er, x - x0
        # Fallback: clamp to last element
        last_er = self.element_results[-1]
        return last_er, last_er.element.length

    def M(self, x):
        """Return bending moment at global coordinate x."""
        er, local_x = self._find_element_at_x(x)
        return er.bending_moment(local_x)

    def V(self, x):
        """Return shear force at global coordinate x."""
        er, local_x = self._find_element_at_x(x)
        return er.shear_force(local_x)

    def N(self, x):
        """Return normal (axial) force at global coordinate x."""
        er, local_x = self._find_element_at_x(x)
        return er.normal_force(local_x)

    def get_diagram(self, force_type, n_points=50):
        # Returns arrays for plotting diagrams (moment, shear, normal)
        # force_type: "moment", "shear", "normal"
        pass
