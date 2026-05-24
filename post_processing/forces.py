import numpy as np
from fem.analysis import get_element_dof_indices

class ElementResults:
    """
    Stores and computes results for a single element: bending moment, shear force, normal force.
    """
    def __init__(self, element, displacements, mesh=None):
        self.element = element
        self.displacements = displacements
        self.length = element.length
        self.mesh = mesh
        self._distributed_loads = []
        self._can_equilibrium_recover = getattr(element, "dofs_per_node", 3) == 3
        if mesh is not None:
            self._distributed_loads = [
                load for load in mesh.distributed_loads
                if getattr(load, "element", None) is element
            ]
        self._f_int_local = self._recover_local_internal_nodal_forces()
        self.compute_forces()

    def compute_forces(self):
        # Compute normal force, shear force, bending moment along the element
        # Example for 2-node Euler-Bernoulli:
        # Use shape functions and element DOF to calculate at any x in [0, L]
        pass

    def bending_moment(self, x):
        # Return bending moment at position x along the element
        if self._f_int_local is not None:
            m0 = -self._f_int_local[2]
            v0 = self._f_int_local[1]
            return m0 + v0 * x + self._integrate_transverse_load(x, weighted=True)
        return self.element.bending_moment(x, self.displacements)

    def shear_force(self, x):
        # Return shear force at position x along the element
        if self._f_int_local is not None:
            return self._f_int_local[1] + self._integrate_transverse_load(x)
        return self.element.shear_force(x, self.displacements)

    def normal_force(self, x):
        # Return normal force at position x along the element
        return self.element.normal_force(x, self.displacements)

    def _recover_local_internal_nodal_forces(self):
        if (
            not self._can_equilibrium_recover
            or not hasattr(self.element, "stiffness_matrix")
            or len(self.displacements) < 3
        ):
            return None

        d_local = np.asarray(self.displacements, dtype=float)
        k_local = self.element.R.T @ self.element.stiffness_matrix() @ self.element.R
        f_ext_local = np.zeros_like(d_local, dtype=float)

        for load in self._distributed_loads:
            fe_global = load.apply(self.element)
            f_ext_local += self.element.R.T @ np.asarray(fe_global, dtype=float)

        return k_local @ d_local - f_ext_local

    def _eval_distributed_load_local_transverse(self, distributed_load, x_local):
        L = self.length
        c = self.element.c
        s = self.element.s

        if distributed_load.func:
            x_global = self.element.node_start.x + x_local * c
            try:
                value = float(eval(distributed_load.func, {"np": np, "x": x_global, "L": L}))
            except Exception:
                value = 0.0
        elif (
            distributed_load.magnitude_start is not None
            and distributed_load.magnitude_end is not None
        ):
            a = float(distributed_load.magnitude_start)
            b = float(distributed_load.magnitude_end)
            value = a + (b - a) * (x_local / L)
        elif distributed_load.magnitude_start is not None:
            value = float(distributed_load.magnitude_start)
        else:
            value = 0.0

        if distributed_load.direction == "x":
            return -value * s
        if distributed_load.direction == "y":
            return value * c
        if distributed_load.direction == "t":
            return value
        return 0.0

    def _integrate_transverse_load(self, x, weighted=False):
        if not self._distributed_loads or x <= 0.0:
            return 0.0

        xi, wi = np.polynomial.legendre.leggauss(8)
        s = 0.5 * (xi + 1.0) * x
        w = 0.5 * wi * x

        total = 0.0
        for si, wi_scaled in zip(s, w):
            p = sum(
                self._eval_distributed_load_local_transverse(load, float(si))
                for load in self._distributed_loads
            )
            total += ((x - si) * p if weighted else p) * wi_scaled
        return float(total)

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
            ElementResults(el, self._get_element_dofs(el), mesh) for el in mesh.elements
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
