import numpy as np
from abc import ABC, abstractmethod

def get_element_dof_indices(element, dpn):
    """Return global DOF indices for an element using the mesh/global DOF size."""
    elem_dpn = getattr(element, 'dofs_per_node', 3)
    node_ids = [element.node_start.id, element.node_end.id]

    if hasattr(element, 'node_center') and element.node_center is not None:
        node_ids = [element.node_start.id, element.node_center.id, element.node_end.id]

    dof_indices = []
    for nid in node_ids:
        dof_indices.extend([dpn * (nid - 1) + k for k in range(elem_dpn)])
    return dof_indices

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
    def __init__(self, mesh, structural_behavior="frame"):
        super().__init__(mesh)
        valid_behaviors = {"frame", "truss", "beam"}
        if structural_behavior not in valid_behaviors:
            raise ValueError(
                f"Unsupported structural_behavior '{structural_behavior}'. "
                f"Expected one of {sorted(valid_behaviors)}."
            )
        self.structural_behavior = structural_behavior
        self.active_global_dofs = None

    def _dofs_per_node(self):
        """Return the number of DOFs per node for this mesh."""
        return max((getattr(el, 'dofs_per_node', 3)
                    for el in self.mesh.elements), default=3)

    def _element_active_dof_mask(self, element, n_elem_dof):
        """
        Return a boolean mask selecting active local DOFs for the configured
        structural behavior (frame/truss/beam).
        """
        if self.structural_behavior == "frame":
            return np.ones(n_elem_dof, dtype=bool)

        elem_dpn = getattr(element, 'dofs_per_node', 3)
        n_nodes = 3 if hasattr(element, 'node_center') and element.node_center is not None else 2
        mask = np.zeros(n_elem_dof, dtype=bool)

        for node_i in range(n_nodes):
            base = node_i * elem_dpn
            if self.structural_behavior == "truss":
                mask[base] = True  # u DOF
            else:  # beam
                if elem_dpn >= 2:
                    mask[base + 1] = True  # v DOF
                if elem_dpn >= 3:
                    mask[base + 2:base + elem_dpn] = True  # rotation (+ higher-order bending DOFs)

        return mask

    def _apply_structural_behavior_to_element(self, element, k_local, fe_local):
        """
        Zero inactive DOF rows/cols and forces according to structural behavior.
        """
        n_elem_dof = len(fe_local)
        mask = self._element_active_dof_mask(element, n_elem_dof)
        if mask.all():
            return k_local, fe_local, mask

        k_filtered = np.array(k_local, copy=True)
        f_filtered = np.array(fe_local, copy=True)
        inactive = ~mask
        k_filtered[inactive, :] = 0.0
        k_filtered[:, inactive] = 0.0
        f_filtered[inactive] = 0.0
        return k_filtered, f_filtered, mask

    def _get_element_dof_indices(self, element, dpn):
        """Return global DOF index list for the given element.

        Args:
            element: The element object
            dpn: Global degrees of freedom per node

        Returns:
            list: Global DOF indices for this element
        """
        return get_element_dof_indices(element, dpn)

    def assemble(self):
        n_nodes = len(self.mesh.nodes)
        dpn = self._dofs_per_node()
        self.dpn = dpn                          # store for solve()
        n_dof = dpn * n_nodes
        self.K_global = np.zeros((n_dof, n_dof))
        self.F_global = np.zeros(n_dof)
        self.active_global_dofs = np.zeros(n_dof, dtype=bool)

        # Assemble element matrices
        for element in self.mesh.elements:
            k_local = element.stiffness_matrix()
            fe_local = element.force_vector()
            k_local, fe_local, active_mask = self._apply_structural_behavior_to_element(
                element, k_local, fe_local
            )
            dof_indices = self._get_element_dof_indices(element, dpn)
            n_elem_dof = len(dof_indices)
            self.active_global_dofs[np.asarray(dof_indices)[active_mask]] = True
            for i in range(n_elem_dof):
                for j in range(n_elem_dof):
                    self.K_global[dof_indices[i], dof_indices[j]] += k_local[i, j]
                self.F_global[dof_indices[i]] += fe_local[i]

        # Apply point loads
        for load in getattr(self.mesh, "point_loads", []):
            idx = dpn * (load.node.id - 1) + load.direction
            if (
                self.structural_behavior != "frame"
                and self.active_global_dofs is not None
                and 0 <= idx < len(self.active_global_dofs)
                and not self.active_global_dofs[idx]
                and abs(load.magnitude) > 0.0
            ):
                raise ValueError(
                    f"Point load applied to inactive DOF {load.direction} at node {load.node.id} "
                    f"for structural behavior '{self.structural_behavior}'."
                )
            load.apply(self.F_global, load.node, dpn)

        # Apply distributed loads
        for load in getattr(self.mesh, "distributed_loads", []):
            el = load.element
            fe_global = load.apply(el)
            dof_indices = self._get_element_dof_indices(el, dpn)
            _, fe_global, _ = self._apply_structural_behavior_to_element(
                el, np.zeros((len(fe_global), len(fe_global))), fe_global
            )
            n_elem_dof = len(dof_indices)
            for i in range(n_elem_dof):
                self.F_global[dof_indices[i]] += fe_global[i]

    def _stabilize_inactive_dofs(self, tol=1e-12):
        """Pin unused global DOFs so mixed-DOF meshes remain solvable.

        Mixed meshes can allocate the global system with more DOFs per node than
        some elements actually use. That leaves zero-stiffness rows/columns for
        unsupported DOFs on otherwise valid nodes. When such a DOF has no load,
        it is safely fixed to zero here so the linear solve remains well-posed.
        If a load was applied to one of those inactive DOFs, the model definition
        is inconsistent, so a ValueError is raised instead. The tolerance is used
        to identify numerically zero rows/columns robustly.

        Args:
            tol: Absolute tolerance used to classify a global DOF row/column as
                inactive. The default is intentionally small because assembled
                stiffness entries for active DOFs are many orders of magnitude
                larger in this application.
        """
        row_norms = np.sum(np.abs(self.K_global), axis=1)
        col_norms = np.sum(np.abs(self.K_global), axis=0)

        for i, (row_norm, col_norm) in enumerate(zip(row_norms, col_norms)):
            if row_norm <= tol and col_norm <= tol:
                if abs(self.F_global[i]) > tol:
                    raise ValueError(
                        f"Load applied to inactive DOF {i}. "
                        "Check constraints/loads for nodes that do not support that DOF."
                    )
                self.K_global[i, i] = 1.0
                self.F_global[i] = 0.0

    def solve(self):
        dpn = getattr(self, 'dpn', 3)
        # Apply constraints (modifies K_global and F_global with penalty method)
        if hasattr(self.mesh, "constraints"):
            self.mesh.constraints.apply_all(self.K_global, self.F_global, dpn)
        self._stabilize_inactive_dofs()
        
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
