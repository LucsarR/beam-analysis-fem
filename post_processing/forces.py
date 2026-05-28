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
        elif "ReddyBickford" in class_name:
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
        elif "ReddyBickford" in class_name:
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
