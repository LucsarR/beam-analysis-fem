import numpy as np

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

class StructureResults:
    """
    Manages results for all elements in the mesh.
    """
    def __init__(self, mesh, displacements, reactions=None):
        self.mesh = mesh
        self.displacements = displacements
        self.reactions = reactions  # Dictionary: {(node_id, direction): reaction_force}
        self.element_results = [
            ElementResults(el, self._get_element_dofs(el)) for el in mesh.elements
        ]

    def _get_element_dofs(self, element):
        """Extract DOFs for the element from global displacement vector."""
        # Handle both 2-node (6 or 8 DOF) and 3-node (8 DOF) elements
        if hasattr(element, 'node_center') and element.node_center is not None:
            # 3-node element with 8 DOFs: [u1, v1, θ1, u2, v2, u3, v3, θ3]
            # Center node has only u and v (no rotation)
            dof_indices = [
                3*(element.node_start.id-1), 3*(element.node_start.id-1)+1, 3*(element.node_start.id-1)+2,  # u1, v1, θ1
                3*(element.node_center.id-1), 3*(element.node_center.id-1)+1,  # u2, v2
                3*(element.node_end.id-1), 3*(element.node_end.id-1)+1, 3*(element.node_end.id-1)+2  # u3, v3, θ3
            ]
        else:
            # 2-node element: use element's dofs_per_node (3 for EB/Timoshenko, 4 for Reddy)
            elem_dpn = getattr(element, 'dofs_per_node', 3)
            node_ids = [element.node_start.id, element.node_end.id]
            dof_indices = []
            for nid in node_ids:
                dof_indices.extend([elem_dpn*(nid-1)+k for k in range(elem_dpn)])
        
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