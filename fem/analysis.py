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

    @abstractmethod
    def assemble(self):
        pass

    @abstractmethod
    def solve(self):
        pass

class BeamAnalysis(Analysis):
    """
    Generic beam analysis class for 2D beam structures.
    
    This class works with all beam element types through polymorphism:
    - Euler-Bernoulli 2-node elements (fully supported)
    - Euler-Bernoulli 3-node elements (fully supported)
    - Timoshenko 2-node elements (fully supported)
    - Mixed element types in the same mesh
    
    The class assembles the global stiffness matrix and force vector by calling
    each element's stiffness_matrix() and force_vector() methods, which are
    implemented differently for each element type according to their respective
    beam theories.
    """
    def assemble(self):
        n_nodes = len(self.mesh.nodes)
        n_dof = 3 * n_nodes
        self.K_global = np.zeros((n_dof, n_dof))
        self.F_global = np.zeros(n_dof)

        # Assemble element matrices
        for element in self.mesh.elements:
            k_local = element.stiffness_matrix()
            fe_local = element.force_vector()
            
            # Get global DOF indices for the element
            # Handle 2-node (6 DOF), 3-node (8 DOF for Euler-Bernoulli), and 3-node (9 DOF for Timoshenko) elements
            if hasattr(element, 'node_center') and element.node_center is not None:
                # Check if it's a Timoshenko 3-node element (9 DOFs) or Euler-Bernoulli 3-node element (8 DOFs)
                from fem.element import TimoshenkoElement3Node
                if isinstance(element, TimoshenkoElement3Node):
                    # 3-node Timoshenko element with 9 DOFs: [u1, v1, θ1, u2, v2, θ2, u3, v3, θ3]
                    # Center node has rotation DOF
                    dof_indices = [
                        3*(element.node_start.id-1), 3*(element.node_start.id-1)+1, 3*(element.node_start.id-1)+2,  # u1, v1, θ1
                        3*(element.node_center.id-1), 3*(element.node_center.id-1)+1, 3*(element.node_center.id-1)+2,  # u2, v2, θ2
                        3*(element.node_end.id-1), 3*(element.node_end.id-1)+1, 3*(element.node_end.id-1)+2  # u3, v3, θ3
                    ]
                else:
                    # 3-node Euler-Bernoulli element with 8 DOFs: [u1, v1, θ1, u2, v2, u3, v3, θ3]
                    # Center node has only u and v (no rotation)
                    dof_indices = [
                        3*(element.node_start.id-1), 3*(element.node_start.id-1)+1, 3*(element.node_start.id-1)+2,  # u1, v1, θ1
                        3*(element.node_center.id-1), 3*(element.node_center.id-1)+1,  # u2, v2
                        3*(element.node_end.id-1), 3*(element.node_end.id-1)+1, 3*(element.node_end.id-1)+2  # u3, v3, θ3
                    ]
            else:
                # 2-node element with 6 DOFs
                node_ids = [element.node_start.id, element.node_end.id]
                dof_indices = []
                for nid in node_ids:
                    dof_indices.extend([3*(nid-1), 3*(nid-1)+1, 3*(nid-1)+2])
            
            # Assemble into global matrices
            n_elem_dof = len(dof_indices)
            for i in range(n_elem_dof):
                for j in range(n_elem_dof):
                    self.K_global[dof_indices[i], dof_indices[j]] += k_local[i, j]
                self.F_global[dof_indices[i]] += fe_local[i]

        # Apply point loads
        for load in getattr(self.mesh, "point_loads", []):
            load.apply(self.F_global, load.node)

        # Apply distributed loads
        for load in getattr(self.mesh, "distributed_loads", []):
            el = load.element
            fe_global = load.apply(el)
            
            # Handle 2-node, 3-node Euler-Bernoulli, and 3-node Timoshenko elements
            if hasattr(el, 'node_center') and el.node_center is not None:
                from fem.element import TimoshenkoElement3Node
                if isinstance(el, TimoshenkoElement3Node):
                    # 3-node Timoshenko element with 9 DOFs
                    dof_indices = [
                        3*(el.node_start.id-1), 3*(el.node_start.id-1)+1, 3*(el.node_start.id-1)+2,
                        3*(el.node_center.id-1), 3*(el.node_center.id-1)+1, 3*(el.node_center.id-1)+2,
                        3*(el.node_end.id-1), 3*(el.node_end.id-1)+1, 3*(el.node_end.id-1)+2
                    ]
                else:
                    # 3-node Euler-Bernoulli element with 8 DOFs
                    dof_indices = [
                        3*(el.node_start.id-1), 3*(el.node_start.id-1)+1, 3*(el.node_start.id-1)+2,
                        3*(el.node_center.id-1), 3*(el.node_center.id-1)+1,
                        3*(el.node_end.id-1), 3*(el.node_end.id-1)+1, 3*(el.node_end.id-1)+2
                    ]
            else:
                # 2-node element
                node_ids = [el.node_start.id, el.node_end.id]
                dof_indices = []
                for nid in node_ids:
                    dof_indices.extend([3*(nid-1), 3*(nid-1)+1, 3*(nid-1)+2])
            
            n_elem_dof = len(dof_indices)
            for i in range(n_elem_dof):
                self.F_global[dof_indices[i]] += fe_global[i]

    def solve(self):
        # Apply constraints
        if hasattr(self.mesh, "constraints"):
            self.mesh.constraints.apply_all(self.K_global, self.F_global)
        # Solve for displacements
        displacements = np.linalg.solve(self.K_global, self.F_global)
        return displacements

# Alias for backward compatibility - old code may still use EulerBernoulliAnalysis
# For new code, use BeamAnalysis which better reflects the generic nature
EulerBernoulliAnalysis = BeamAnalysis