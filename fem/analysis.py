import numpy as np
from abc import ABC, abstractmethod

class Analysis(ABC):
    """
    Abstract base class for finite element analysis.
    """
    def __init__(self, mesh):
        self.mesh = mesh
        self.K_global = None
        self.F_global = None
        self.reactions = None  # Store reactions calculated during solve

    @abstractmethod
    def assemble(self):
        pass

    @abstractmethod
    def solve(self):
        pass
    
    def get_reactions(self):
        """
        Get the reaction forces calculated at constraints.
        Returns None if no reactions were calculated.
        """
        return self.reactions

class BeamAnalysis(Analysis):
    """
    Generic beam analysis class for 2D beam structures.
    
    This class works with all beam element types through polymorphism:
    - Euler-Bernoulli 2-node elements (3 DOFs/node, fully supported)
    - Euler-Bernoulli 3-node elements (3 DOFs/node, fully supported)
    - Timoshenko 2-node elements (3 DOFs/node, fully supported)
    - Reddy-Bickford 2-node elements (4 DOFs/node, fully supported)
    - Mixed element types in the same mesh
    
    The class assembles the global stiffness matrix and force vector by calling
    each element's stiffness_matrix() and force_vector() methods, which are
    implemented differently for each element type according to their respective
    beam theories.

    DOF convention:
      3-DOF nodes: [u, v, θ]          (Euler-Bernoulli, Timoshenko)
      4-DOF nodes: [u, v, θ, dv/dx]  (Reddy-Bickford)
    """
    def _dofs_per_node(self):
        """Return the number of DOFs per node for this mesh."""
        return max((getattr(el, 'dofs_per_node', 3)
                    for el in self.mesh.elements), default=3)

    def _get_element_dof_indices(self, element, dpn):
        """Return global DOF index list for the given element."""
        if hasattr(element, 'node_center') and element.node_center is not None:
            # 3-node element (Euler-Bernoulli or Timoshenko 3-node)
            dof_indices = []
            for nid in [element.node_start.id,
                        element.node_center.id,
                        element.node_end.id]:
                dof_indices.extend([dpn*(nid-1)+k for k in range(dpn)])
        else:
            # 2-node element
            dof_indices = []
            for nid in [element.node_start.id, element.node_end.id]:
                dof_indices.extend([dpn*(nid-1)+k for k in range(dpn)])
        return dof_indices

    def assemble(self):
        n_nodes = len(self.mesh.nodes)
        dpn = self._dofs_per_node()
        self.dpn = dpn                          # store for solve()
        n_dof = dpn * n_nodes
        self.K_global = np.zeros((n_dof, n_dof))
        self.F_global = np.zeros(n_dof)

        # Assemble element matrices
        for element in self.mesh.elements:
            k_local = element.stiffness_matrix()
            fe_local = element.force_vector()
            elem_dpn = getattr(element, 'dofs_per_node', 3)
            dof_indices = self._get_element_dof_indices(element, elem_dpn)
            n_elem_dof = len(dof_indices)
            for i in range(n_elem_dof):
                for j in range(n_elem_dof):
                    self.K_global[dof_indices[i], dof_indices[j]] += k_local[i, j]
                self.F_global[dof_indices[i]] += fe_local[i]

        # Apply point loads
        for load in getattr(self.mesh, "point_loads", []):
            load.apply(self.F_global, load.node, dpn)

        # Apply distributed loads
        for load in getattr(self.mesh, "distributed_loads", []):
            el = load.element
            fe_global = load.apply(el)
            elem_dpn = getattr(el, 'dofs_per_node', 3)
            dof_indices = self._get_element_dof_indices(el, elem_dpn)
            n_elem_dof = len(dof_indices)
            for i in range(n_elem_dof):
                self.F_global[dof_indices[i]] += fe_global[i]

    def solve(self):
        dpn = getattr(self, 'dpn', 3)
        # Apply constraints (modifies K_global and F_global with penalty method)
        if hasattr(self.mesh, "constraints"):
            self.mesh.constraints.apply_all(self.K_global, self.F_global, dpn)
        
        # Solve for displacements
        displacements = np.linalg.solve(self.K_global, self.F_global)
        
        # Calculate reactions at constraints and store in analysis object
        if hasattr(self.mesh, "constraints") and len(self.mesh.constraints.constraints) > 0:
            self.reactions = self.mesh.constraints.calculate_all_reactions(
                displacements, dpn)
        
        return displacements

# Alias for backward compatibility - old code may still use EulerBernoulliAnalysis
# For new code, use BeamAnalysis which better reflects the generic nature
EulerBernoulliAnalysis = BeamAnalysis