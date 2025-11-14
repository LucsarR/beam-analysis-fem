import numpy as np
from abc import ABC, abstractmethod

class Element(ABC):
    """
    Abstract base class for beam elements.
    """
    def __init__(self, id, node_start, node_end, material, section):
        self.id = id
        self.node_start = node_start
        self.node_end = node_end
        self.material = material
        self.section = section

    @abstractmethod
    def stiffness_matrix(self):
        pass

    @abstractmethod
    def force_vector(self, q_ini=0, q_fim=0, p_ini=0, p_fim=0):
        pass

class EulerBernoulliElement2Node(Element):
    def __init__(self, id, node_start, node_end, material, section):
        super().__init__(id, node_start, node_end, material, section)
        self.length, self.c, self.s, self.R = self._compute_geometry()

    def _compute_geometry(self):
        x1, y1 = self.node_start.x, self.node_start.y
        x2, y2 = self.node_end.x, self.node_end.y
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        c = (x2 - x1) / L
        s = (y2 - y1) / L
        # Transformation matrix
        R = np.array([
            [c, -s, 0, 0, 0, 0],
            [s, c, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, -s, 0],
            [0, 0, 0, s, c, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        return L, c, s, R

    def stiffness_matrix(self):
        E = self.material.E
        A = self.section.area
        I = self.section.inertia
        L = self.length
        R = self.R
        mu = (A * L**2) / (2 * I)
        # Local stiffness matrix
        k_local = 2 * E * I / L**3 * np.array([
            [mu, 0, 0, -mu, 0, 0],
            [0, 6, 3*L, 0, -6, 3*L],
            [0, 3*L, 2*L**2, 0, -3*L, L**2],
            [-mu, 0, 0, mu, 0, 0],
            [0, -6, -3*L, 0, 6, -3*L],
            [0, 3*L, L**2, 0, -3*L, 2*L**2]
        ])
        return R @ k_local @ R.T

    def force_vector(self, q_ini=0, q_fim=0, p_ini=0, p_fim=0):
        L = self.length
        R = self.R
        fe_local = L * R @ np.array([
            [(2*q_ini + q_fim) / 6],
            [(7*p_ini + 3*p_fim) / 20],
            [(3*p_ini + 2*p_fim) * L / 60],
            [(q_ini + 2*q_fim) / 6],
            [(3*p_ini + 7*p_fim) / 20],
            [-(2*p_ini + 3*p_fim) * L / 60]
        ])
        return fe_local.flatten()

    def compute_equivalent_nodal_loads(self, distributed_load, n_gauss=5):
        """
        Compute consistent nodal loads for a distributed load (constant, linear, or custom function).
        Returns a 6-vector in GLOBAL coordinates.
        """
        import numpy as np
        L = self.length
        c = self.c
        s = self.s

        # Build load function f(x) from distributed_load
        if distributed_load.func:
            # Custom function
            def f(x):
                try:
                    return float(eval(distributed_load.func, {"np": np, "x": x, "L": L}))
                except Exception as e:
                    print(f"Error evaluating custom function '{distributed_load.func}': {e}")
                    return 0.0  # or raise ValueError("Invalid custom function for distributed load")
        elif distributed_load.magnitude_start is not None and distributed_load.magnitude_end is not None:
            # Linear
            a = float(distributed_load.magnitude_start)
            b = float(distributed_load.magnitude_end)
            def f(x):
                return a + (b - a) * (x / L)
        elif distributed_load.magnitude_start is not None:
            # Constant
            a = float(distributed_load.magnitude_start)
            def f(x):
                return a
        else:
            def f(x):
                return 0.0

        # Project direction to local axes
        if distributed_load.direction == 'x':
            def q_local(x): return f(x) * c
            def p_local(x): return -f(x) * s
        elif distributed_load.direction == 'y':
            def q_local(x): return f(x) * s
            def p_local(x): return f(x) * c
        elif distributed_load.direction == 'l':
            def q_local(x): return f(x)
            def p_local(x): return 0.0
        elif distributed_load.direction == 't':
            def q_local(x): return 0.0
            def p_local(x): return f(x)
        else:
            def q_local(x): return 0.0
            def p_local(x): return 0.0

        # Gauss-Legendre quadrature points on [0,1] mapped to [0,L]
        xi, wi = np.polynomial.legendre.leggauss(n_gauss)
        t = 0.5 * (xi + 1.0)
        wt = 0.5 * wi

        ia1 = ia2 = iv1 = itheta1 = iv2 = itheta2 = 0.0
        for ti, wi_scaled in zip(t, wt):
            x = ti * L
            N1_ax = 1.0 - ti
            N2_ax = ti
            qx = q_local(x)
            ia1 += N1_ax * qx * wi_scaled * L
            ia2 += N2_ax * qx * wi_scaled * L

            Hv1 = 1 - 3*ti**2 + 2*ti**3
            Ht1 = L * (ti - 2*ti**2 + ti**3)
            Hv2 = 3*ti**2 - 2*ti**3
            Ht2 = L * (-ti**2 + ti**3)
            px = p_local(x)
            iv1 += Hv1 * px * wi_scaled * L
            itheta1 += Ht1 * px * wi_scaled * L
            iv2 += Hv2 * px * wi_scaled * L
            itheta2 += Ht2 * px * wi_scaled * L

        # local consistent vector in order [u1, v1, theta1, u2, v2, theta2]
        flocal = np.array([ia1, iv1, itheta1, ia2, iv2, itheta2], dtype=float)
        # Transform to global coordinates
        R = self.R
        fe_global = R @ flocal
        return fe_global.flatten()

    def bending_moment(self, x, displacements):
        """
        Returns bending moment M(x) at position x (in local coordinates, 0 <= x <= L).
        """
        E = self.material.E
        I = self.section.inertia
        L = self.length
        # Local DOFs: [u1, v1, theta1, u2, v2, theta2]
        v1 = displacements[1]
        theta1 = displacements[2]
        v2 = displacements[4]
        theta2 = displacements[5]
        xi = x / L
        # Hermite shape function second derivatives
        d2N1 = (12 / L**2) * (xi - 0.5)
        d2N2 = (6 / L) * (xi - 2/3)
        d2N3 = (-12 / L**2) * (xi - 0.5)
        d2N4 = (6 / L) * (xi - 1/3)
        # Bending moment: M(x) = E*I * w''(x)
        w_dd = d2N1 * v1 + d2N2 * theta1 + d2N3 * v2 + d2N4 * theta2
        return E * I * w_dd

    def shear_force(self, x, displacements):
        """
        Returns shear force V(x) at position x (in local coordinates, 0 <= x <= L).
        """
        E = self.material.E
        I = self.section.inertia
        L = self.length
        v1 = displacements[1]
        theta1 = displacements[2]
        v2 = displacements[4]
        theta2 = displacements[5]
        xi = x / L
        # Hermite shape function third derivatives
        d3N1 = (12 / L**3)
        d3N2 = (6 / L**2)
        d3N3 = (-12 / L**3)
        d3N4 = (6 / L**2)
        # Shear force: V(x) = E*I * w'''(x)
        w_ddd = d3N1 * v1 + d3N2 * theta1 + d3N3 * v2 + d3N4 * theta2
        return E * I * w_ddd

    def normal_force(self, x, displacements):
        """
        Returns normal (axial) force N(x) at position x (in local coordinates, 0 <= x <= L).
        """
        E = self.material.E
        A = self.section.area
        L = self.length
        u1 = displacements[0]
        u2 = displacements[3]
        xi = x / L
        # Linear shape function derivatives
        dN1 = -1 / L
        dN2 = 1 / L
        # Axial strain: epsilon(x) = du/dx = dN1*u1 + dN2*u2
        epsilon = dN1 * u1 + dN2 * u2
        return E * A * epsilon

class EulerBernoulliElement3Node(Element):
    def __init__(self, id, node_start, node_end, material, section):
        super().__init__(id, node_start, node_end, material, section)
        self.length, self.c, self.s = self._compute_geometry()

    def _compute_geometry(self):
        x1, y1 = self.node_start.x, self.node_start.y
        x2, y2 = (self.node_start.x + self.node_end.x) / 2, (self.node_start.y + self.node_end.y) / 2
        x3, y3 = self.node_end.x, self.node_end.y
        L = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
        c = (x3 - x1) / L
        s = (y3 - y1) / L
        return L, c, s

    def stiffness_matrix(self):
        E = self.material.E
        A = self.section.area
        I = self.section.inertia
        L = self.length
        c = self.c
        s = self.s

        # TODO: Implement the stiffness matrix for 3-node Euler-Bernoulli beam element

        return k

    def force_vector(self, q_ini=0, q_fim=0, p_ini=0, p_fim=0):
        L = self.length
        c = self.c
        s = self.s

        # TODO: Implement the force vector for 3-node Euler-Bernoulli beam element

        return fe_local.flatten()
    
class TimoshenkoElement2Node(Element):
    def __init__(self, id, node_start, node_end, material, section):
        super().__init__(id, node_start, node_end, material, section)
        self.length, self.c, self.s = self._compute_geometry()

    def _compute_geometry(self):
        x1, y1 = self.node_start.x, self.node_start.y
        x2, y2 = self.node_end.x, self.node_end.y
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        c = (x2 - x1) / L
        s = (y2 - y1) / L
        return L, c, s

    def stiffness_matrix(self):
        E = self.material.E
        G = self.material.G
        A = self.section.area
        I = self.section.inertia
        kappa = self.section.shear_coefficient
        L = self.length
        c = self.c
        s = self.s
        
        # Shear area
        As = kappa * A
        
        # Shear parameter (phi)
        phi = 12 * E * I / (G * As * L**2)
        
        # Local stiffness matrix for Timoshenko beam
        # DOFs: [u1, v1, theta1, u2, v2, theta2]
        k_local = np.zeros((6, 6))
        
        # Axial stiffness terms (same as Euler-Bernoulli)
        k_local[0, 0] = E * A / L
        k_local[0, 3] = -E * A / L
        k_local[3, 0] = -E * A / L
        k_local[3, 3] = E * A / L
        
        # Bending and shear stiffness terms
        # These terms account for shear deformation
        k_local[1, 1] = 12 * E * I / (L**3 * (1 + phi))
        k_local[1, 2] = 6 * E * I / (L**2 * (1 + phi))
        k_local[1, 4] = -12 * E * I / (L**3 * (1 + phi))
        k_local[1, 5] = 6 * E * I / (L**2 * (1 + phi))
        
        k_local[2, 1] = 6 * E * I / (L**2 * (1 + phi))
        k_local[2, 2] = (4 + phi) * E * I / (L * (1 + phi))
        k_local[2, 4] = -6 * E * I / (L**2 * (1 + phi))
        k_local[2, 5] = (2 - phi) * E * I / (L * (1 + phi))
        
        k_local[4, 1] = -12 * E * I / (L**3 * (1 + phi))
        k_local[4, 2] = -6 * E * I / (L**2 * (1 + phi))
        k_local[4, 4] = 12 * E * I / (L**3 * (1 + phi))
        k_local[4, 5] = -6 * E * I / (L**2 * (1 + phi))
        
        k_local[5, 1] = 6 * E * I / (L**2 * (1 + phi))
        k_local[5, 2] = (2 - phi) * E * I / (L * (1 + phi))
        k_local[5, 4] = -6 * E * I / (L**2 * (1 + phi))
        k_local[5, 5] = (4 + phi) * E * I / (L * (1 + phi))
        
        # Transformation matrix
        R = np.array([
            [c, -s, 0, 0, 0, 0],
            [s, c, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, -s, 0],
            [0, 0, 0, s, c, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Transform to global coordinates
        k = R @ k_local @ R.T
        
        return k

    def force_vector(self, q_ini=0, q_fim=0, p_ini=0, p_fim=0):
        L = self.length
        c = self.c
        s = self.s
        
        # Transformation matrix
        R = np.array([
            [c, -s, 0, 0, 0, 0],
            [s, c, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, -s, 0],
            [0, 0, 0, s, c, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Consistent nodal load vector for uniformly distributed loads
        # Similar to Euler-Bernoulli but with shear deformation effects
        fe_local = L * R @ np.array([
            [(2*q_ini + q_fim) / 6],
            [(7*p_ini + 3*p_fim) / 20],
            [(3*p_ini + 2*p_fim) * L / 60],
            [(q_ini + 2*q_fim) / 6],
            [(3*p_ini + 7*p_fim) / 20],
            [-(2*p_ini + 3*p_fim) * L / 60]
        ])
        
        return fe_local.flatten()
    
    def compute_equivalent_nodal_loads(self, distributed_load, n_gauss=5):
        """
        Compute consistent nodal loads for a distributed load (constant, linear, or custom function).
        Returns a 6-vector in GLOBAL coordinates.
        For Timoshenko beam elements, uses the same approach as Euler-Bernoulli.
        """
        import numpy as np
        L = self.length
        c = self.c
        s = self.s

        # Build load function f(x) from distributed_load
        if distributed_load.func:
            # Custom function
            def f(x):
                try:
                    return float(eval(distributed_load.func, {"np": np, "x": x, "L": L}))
                except Exception as e:
                    print(f"Error evaluating custom function '{distributed_load.func}': {e}")
                    return 0.0
        elif distributed_load.magnitude_start is not None and distributed_load.magnitude_end is not None:
            # Linear
            a = float(distributed_load.magnitude_start)
            b = float(distributed_load.magnitude_end)
            def f(x):
                return a + (b - a) * (x / L)
        elif distributed_load.magnitude_start is not None:
            # Constant
            a = float(distributed_load.magnitude_start)
            def f(x):
                return a
        else:
            def f(x):
                return 0.0

        # Project direction to local axes
        if distributed_load.direction == 'x':
            def q_local(x): return f(x) * c
            def p_local(x): return -f(x) * s
        elif distributed_load.direction == 'y':
            def q_local(x): return f(x) * s
            def p_local(x): return f(x) * c
        elif distributed_load.direction == 'l':
            def q_local(x): return f(x)
            def p_local(x): return 0.0
        elif distributed_load.direction == 't':
            def q_local(x): return 0.0
            def p_local(x): return f(x)
        else:
            def q_local(x): return 0.0
            def p_local(x): return 0.0

        # Gauss-Legendre quadrature points on [0,1] mapped to [0,L]
        xi, wi = np.polynomial.legendre.leggauss(n_gauss)
        t = 0.5 * (xi + 1.0)
        wt = 0.5 * wi

        ia1 = ia2 = iv1 = itheta1 = iv2 = itheta2 = 0.0
        for ti, wi_scaled in zip(t, wt):
            x = ti * L
            N1_ax = 1.0 - ti
            N2_ax = ti
            qx = q_local(x)
            ia1 += N1_ax * qx * wi_scaled * L
            ia2 += N2_ax * qx * wi_scaled * L

            # Use Hermite shape functions for transverse loads
            Hv1 = 1 - 3*ti**2 + 2*ti**3
            Ht1 = L * (ti - 2*ti**2 + ti**3)
            Hv2 = 3*ti**2 - 2*ti**3
            Ht2 = L * (-ti**2 + ti**3)
            px = p_local(x)
            iv1 += Hv1 * px * wi_scaled * L
            itheta1 += Ht1 * px * wi_scaled * L
            iv2 += Hv2 * px * wi_scaled * L
            itheta2 += Ht2 * px * wi_scaled * L

        # local consistent vector in order [u1, v1, theta1, u2, v2, theta2]
        flocal = np.array([ia1, iv1, itheta1, ia2, iv2, itheta2], dtype=float)
        # Transform to global coordinates
        R = np.array([
            [c, -s, 0, 0, 0, 0],
            [s, c, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, -s, 0],
            [0, 0, 0, s, c, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        fe_global = R @ flocal
        return fe_global.flatten()

    def bending_moment(self, x, displacements):
        """
        Returns bending moment M(x) at position x (in local coordinates, 0 <= x <= L).
        For Timoshenko beam theory.
        """
        E = self.material.E
        I = self.section.inertia
        L = self.length
        # Local DOFs: [u1, v1, theta1, u2, v2, theta2]
        v1 = displacements[1]
        theta1 = displacements[2]
        v2 = displacements[4]
        theta2 = displacements[5]
        xi = x / L
        
        # Linear interpolation for rotation (theta)
        theta = (1 - xi) * theta1 + xi * theta2
        
        # Shape functions for transverse displacement
        N1 = 1 - xi
        N2 = xi
        
        # Derivative of transverse displacement
        dv_dx = (-v1 + v2) / L
        
        # Bending moment: M(x) = E*I * d(theta)/dx for Timoshenko
        # For linear shape functions: dtheta/dx = (theta2 - theta1) / L
        dtheta_dx = (theta2 - theta1) / L
        
        return E * I * dtheta_dx
    
    def shear_force(self, x, displacements):
        """
        Returns shear force V(x) at position x (in local coordinates, 0 <= x <= L).
        For Timoshenko beam theory.
        """
        E = self.material.E
        G = self.material.G
        I = self.section.inertia
        A = self.section.area
        kappa = self.section.shear_coefficient
        L = self.length
        
        v1 = displacements[1]
        theta1 = displacements[2]
        v2 = displacements[4]
        theta2 = displacements[5]
        xi = x / L
        
        # Shear force: V = kappa*G*A*(dv/dx - theta)
        # Linear interpolation for rotation
        theta = (1 - xi) * theta1 + xi * theta2
        
        # Derivative of transverse displacement
        dv_dx = (-v1 + v2) / L
        
        # Shear force
        V = kappa * G * A * (dv_dx - theta)
        
        return V
    
    def normal_force(self, x, displacements):
        """
        Returns normal (axial) force N(x) at position x (in local coordinates, 0 <= x <= L).
        """
        E = self.material.E
        A = self.section.area
        L = self.length
        u1 = displacements[0]
        u2 = displacements[3]
        xi = x / L
        # Linear shape function derivatives
        dN1 = -1 / L
        dN2 = 1 / L
        # Axial strain: epsilon(x) = du/dx = dN1*u1 + dN2*u2
        epsilon = dN1 * u1 + dN2 * u2
        return E * A * epsilon

class ElementResults:
    ...
    def bending_moment(self, x):
        return self.element.bending_moment(x, self.displacements)
    def shear_force(self, x):
        return self.element.shear_force(x, self.displacements)
    def normal_force(self, x):
        return self.element.normal_force(x, self.displacements)