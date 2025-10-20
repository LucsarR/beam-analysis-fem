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
    def __init__(self, mesh, displacements):
        self.mesh = mesh
        self.displacements = displacements
        self.element_results = [
            ElementResults(el, self._get_element_dofs(el)) for el in mesh.elements
        ]

    def _get_element_dofs(self, element):
        # Extract DOFs for the element from global displacement vector
        node_ids = [element.node_start.id, element.node_end.id]
        dof_indices = []
        for nid in node_ids:
            dof_indices.extend([3*(nid-1), 3*(nid-1)+1, 3*(nid-1)+2])
        return self.displacements[dof_indices]

    def get_diagram(self, force_type, n_points=50):
        # Returns arrays for plotting diagrams (moment, shear, normal)
        # force_type: "moment", "shear", "normal"
        pass