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

class EulerBernoulliAnalysis(Analysis):
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
            node_ids = [element.node_start.id, element.node_end.id]
            dof_indices = []
            for nid in node_ids:
                dof_indices.extend([3*(nid-1), 3*(nid-1)+1, 3*(nid-1)+2])
            # Assemble into global matrices
            for i in range(6):
                for j in range(6):
                    self.K_global[dof_indices[i], dof_indices[j]] += k_local[i, j]
                self.F_global[dof_indices[i]] += fe_local[i]

        # Apply point loads
        for load in getattr(self.mesh, "point_loads", []):
            load.apply(self.F_global, load.node)

        # Apply distributed loads
        for load in getattr(self.mesh, "distributed_loads", []):
            el = load.element
            c = el.c
            s = el.s
            # Project distributed load according to direction
            if load.direction == 'x':
                q_ini = load.magnitude_start * c
                q_fim = load.magnitude_end * c
                p_ini = -load.magnitude_start * s
                p_fim = -load.magnitude_end * s
            elif load.direction == 'y':
                q_ini = load.magnitude_start * s
                q_fim = load.magnitude_end * s
                p_ini = load.magnitude_start * c
                p_fim = load.magnitude_end * c
            elif load.direction == 'l':
                q_ini = load.magnitude_start
                q_fim = load.magnitude_end
                p_ini = 0
                p_fim = 0
            elif load.direction == 't':
                q_ini = 0
                q_fim = 0
                p_ini = load.magnitude_start
                p_fim = load.magnitude_end
            else:
                q_ini = 0
                q_fim = 0
                p_ini = 0
                p_fim = 0
            fe_dist = el.force_vector(q_ini=q_ini, q_fim=q_fim, p_ini=p_ini, p_fim=p_fim)
            node_ids = [el.node_start.id, el.node_end.id]
            dof_indices = []
            for nid in node_ids:
                dof_indices.extend([3*(nid-1), 3*(nid-1)+1, 3*(nid-1)+2])
            for i in range(6):
                self.F_global[dof_indices[i]] += fe_dist[i]

    def solve(self):
        # Apply constraints
        if hasattr(self.mesh, "constraints"):
            self.mesh.constraints.apply_all(self.K_global, self.F_global)
        # Solve for displacements
        displacements = np.linalg.solve(self.K_global, self.F_global)
        return displacements