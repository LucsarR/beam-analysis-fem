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
    def __init__(self, id, node_start, node_end, material, section, node_center=None):
        super().__init__(id, node_start, node_end, material, section)
        self.node_center = node_center  # Central node
        self.length, self.c, self.s, self.R = self._compute_geometry()

    def _compute_geometry(self):
        x1, y1 = self.node_start.x, self.node_start.y
        x3, y3 = self.node_end.x, self.node_end.y
        L = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
        c = (x3 - x1) / L
        s = (y3 - y1) / L
        # Transformation matrix for 3-node element (8x8)
        # DOFs: [u1, v1, θ1, u2, v2, u3, v3, θ3]
        # Note: Central node has no rotation DOF
        R = np.zeros((8, 8))
        # Node 1: u1, v1, θ1
        R[0:2, 0:2] = np.array([[c, -s], [s, c]])
        R[2, 2] = 1
        # Node 2 (center): u2, v2 (no rotation)
        R[3:5, 3:5] = np.array([[c, -s], [s, c]])
        # Node 3: u3, v3, θ3
        R[5:7, 5:7] = np.array([[c, -s], [s, c]])
        R[7, 7] = 1
        return L, c, s, R

    def stiffness_matrix(self):
        """
        Stiffness matrix for 3-node Euler-Bernoulli beam element.
        
        The element has 8 DOFs: [u1, v1, θ1, u2, v2, u3, v3, θ3]
        - Axial: quadratic shape functions for u
        - Bending: Hermite cubics + central bubble function for v
        - Central node has no rotation DOF
        
        Shape functions:
        - H1 = 1 - 3ξ² + 2ξ³ (v1)
        - H2 = L(ξ - 2ξ² + ξ³) (θ1)
        - Hb = 16ξ²(1-ξ)² (v2, bubble function)
        - H3 = 3ξ² - 2ξ³ (v3)
        - H4 = L(-ξ² + ξ³) (θ3)
        
        References:
        - Reddy, J.N. "An Introduction to the Finite Element Method" (2006)
        - Bathe, K.J. "Finite Element Procedures" (1996)
        """
        E = self.material.E
        A = self.section.area
        I = self.section.inertia
        L = self.length
        R = self.R
        
        # Local stiffness matrix (8x8)
        k_local = np.zeros((8, 8))
        
        # Axial stiffness using quadratic shape functions
        # Shape functions: N1 = (1-ξ)(1-2ξ), N2 = 4ξ(1-ξ), N3 = ξ(2ξ-1)
        # where ξ = x/L
        k_axial = E * A / (3 * L) * np.array([
            [7, -8, 1],
            [-8, 16, -8],
            [1, -8, 7]
        ])
        
        # Assign axial stiffness to DOFs [u1, u2, u3] = [0, 3, 5]
        axial_dofs = [0, 3, 5]
        for i, ii in enumerate(axial_dofs):
            for j, jj in enumerate(axial_dofs):
                k_local[ii, jj] = k_axial[i, j]
        
        # Bending stiffness using Hermite shape functions with central bubble node
        # Computed numerically via integration
        # For 3-node element with v1, θ1, v2 (bubble), v3, θ3
        # DOFs: [v1, θ1, v2, v3, θ3] = indices [1, 2, 4, 6, 7]
        # Note: The coefficient 51.2 for the bubble function (v2) is derived from:
        # ∫₀¹ (d²Hb/dξ²)² dξ where Hb = 16ξ²(1-ξ)² and d²Hb/dξ² = 32(3ξ²-3ξ+0.5)
        # This evaluates to 51.2 when integrated over [0,1]
        k_bending = E * I / (L**3) * np.array([
            [12.0, 6.0*L, 0.0, -12.0, 6.0*L],
            [6.0*L, 4.0*L**2, 0.0, -6.0*L, 2.0*L**2],
            [0.0, 0.0, 51.2, 0.0, 0.0],  # 51.2 = ∫₀¹ (d²Hb/dξ²)² dξ
            [-12.0, -6.0*L, 0.0, 12.0, -6.0*L],
            [6.0*L, 2.0*L**2, 0.0, -6.0*L, 4.0*L**2]
        ])
        
        # Assign bending stiffness to DOFs [v1, θ1, v2, v3, θ3]
        bending_dofs = [1, 2, 4, 6, 7]
        for i, ii in enumerate(bending_dofs):
            for j, jj in enumerate(bending_dofs):
                k_local[ii, jj] = k_bending[i, j]
        
        # Transform to global coordinates
        return R @ k_local @ R.T

    def force_vector(self, q_ini=0, q_fim=0, p_ini=0, p_fim=0):
        """
        Consistent nodal load vector for 3-node Euler-Bernoulli element.
        
        Args:
            q_ini: Initial axial distributed load (force per unit length)
            q_fim: Final axial distributed load
            p_ini: Initial transverse distributed load
            p_fim: Final transverse distributed load
            
        Returns:
            8-element force vector in global coordinates [u1, v1, θ1, u2, v2, u3, v3, θ3]
        """
        L = self.length
        R = self.R
        
        # Consistent load vector for linearly varying distributed loads
        # For quadratic axial shape functions
        q_avg = (q_ini + q_fim) / 2
        fe_axial = L * np.array([
            q_ini / 6,
            2 * q_avg / 3,
            q_fim / 6
        ])
        
        # For bending with Hermite + central bubble node
        # Coefficients are derived from consistent load distribution:
        # ∫₀¹ Hᵢ(ξ) p(ξ) L dξ where p(ξ) is the distributed load
        # For uniform load: [7/20, 3L/60, 16/70, 3/20, -3L/60]
        p_avg = (p_ini + p_fim) / 2
        fe_bending = L * np.array([
            (7*p_ini + 3*p_fim) / 20,       # v1 contribution
            (3*p_ini + 2*p_fim) * L / 60,   # θ1 contribution
            (16*p_ini + 16*p_fim) / 70,     # v2 (bubble) contribution
            (3*p_ini + 7*p_fim) / 20,       # v3 contribution
            -(2*p_ini + 3*p_fim) * L / 60   # θ3 contribution
        ])
        
        # Assemble into 8-DOF vector [u1, v1, θ1, u2, v2, u3, v3, θ3]
        fe_local = np.zeros(8)
        fe_local[0] = fe_axial[0]  # u1
        fe_local[1] = fe_bending[0]  # v1
        fe_local[2] = fe_bending[1]  # θ1
        fe_local[3] = fe_axial[1]  # u2
        fe_local[4] = fe_bending[2]  # v2
        fe_local[5] = fe_axial[2]  # u3
        fe_local[6] = fe_bending[3]  # v3
        fe_local[7] = fe_bending[4]  # θ3
        
        # Transform to global coordinates
        fe_global = R @ fe_local
        return fe_global.flatten()
    
    def compute_equivalent_nodal_loads(self, distributed_load, n_gauss=5):
        """
        Compute consistent nodal loads for a distributed load using numerical integration.
        Returns an 8-vector in GLOBAL coordinates [u1, v1, θ1, u2, v2, u3, v3, θ3].
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

        # Gauss-Legendre quadrature
        xi, wi = np.polynomial.legendre.leggauss(n_gauss)
        t = 0.5 * (xi + 1.0)
        wt = 0.5 * wi

        # Initialize force components
        ia1 = ia2 = ia3 = 0.0
        iv1 = itheta1 = iv2 = iv3 = itheta3 = 0.0
        
        for ti, wi_scaled in zip(t, wt):
            x = ti * L
            
            # Quadratic shape functions for axial
            N1_ax = (1 - ti) * (1 - 2*ti)
            N2_ax = 4 * ti * (1 - ti)
            N3_ax = ti * (2*ti - 1)
            qx = q_local(x)
            ia1 += N1_ax * qx * wi_scaled * L
            ia2 += N2_ax * qx * wi_scaled * L
            ia3 += N3_ax * qx * wi_scaled * L

            # Hermite shape functions for bending
            Hv1 = 1 - 3*ti**2 + 2*ti**3
            Ht1 = L * (ti - 2*ti**2 + ti**3)
            Hv2 = 16 * ti**2 * (1 - ti)**2  # Central node shape function
            Hv3 = 3*ti**2 - 2*ti**3
            Ht3 = L * (-ti**2 + ti**3)
            px = p_local(x)
            iv1 += Hv1 * px * wi_scaled * L
            itheta1 += Ht1 * px * wi_scaled * L
            iv2 += Hv2 * px * wi_scaled * L
            iv3 += Hv3 * px * wi_scaled * L
            itheta3 += Ht3 * px * wi_scaled * L

        # Local consistent vector [u1, v1, theta1, u2, v2, u3, v3, theta3]
        flocal = np.array([ia1, iv1, itheta1, ia2, iv2, ia3, iv3, itheta3], dtype=float)
        # Transform to global coordinates
        R = self.R
        fe_global = R @ flocal
        return fe_global.flatten()

    def bending_moment(self, x, displacements):
        """
        Returns bending moment M(x) at position x (in local coordinates, 0 <= x <= L).
        Displacements should be in local coordinates: [u1, v1, theta1, u2, v2, u3, v3, theta3]
        """
        E = self.material.E
        I = self.section.inertia
        L = self.length
        # Local DOFs: [u1, v1, theta1, u2, v2, u3, v3, theta3]
        v1 = displacements[1]
        theta1 = displacements[2]
        v2 = displacements[4]
        v3 = displacements[6]
        theta3 = displacements[7]
        
        xi = x / L
        
        # Second derivatives of Hermite + central node shape functions w.r.t. ξ
        d2Hv1_dxi2 = -6 + 12*xi
        d2Ht1_dxi2 = L * (-4 + 6*xi)
        # Bubble function: d²Hb/dξ² (numerically validated)
        d2Hv2_dxi2 = 96 * (xi**2 - xi + 1/6)
        d2Hv3_dxi2 = 6 - 12*xi
        d2Ht3_dxi2 = L * (-2 + 6*xi)
        
        # w''(x) = d2w/dx2 = (1/L^2) * d2w/dxi2
        w_dd = (1/L**2) * (d2Hv1_dxi2 * v1 + d2Ht1_dxi2 * theta1 + 
                           d2Hv2_dxi2 * v2 + d2Hv3_dxi2 * v3 + d2Ht3_dxi2 * theta3)
        
        return E * I * w_dd

    def shear_force(self, x, displacements):
        """
        Returns shear force V(x) at position x (in local coordinates, 0 <= x <= L).
        Displacements should be in local coordinates: [u1, v1, theta1, u2, v2, u3, v3, theta3]
        """
        E = self.material.E
        I = self.section.inertia
        L = self.length
        v1 = displacements[1]
        theta1 = displacements[2]
        v2 = displacements[4]
        v3 = displacements[6]
        theta3 = displacements[7]
        
        xi = x / L
        
        # Third derivatives of Hermite + central node shape functions
        d3Hv1_dxi3 = 12
        d3Ht1_dxi3 = L * 6
        d3Hv2_dxi3 = 192 * (xi - 0.5)  # Simplified from 32*6*(xi-0.5)
        d3Hv3_dxi3 = -12
        d3Ht3_dxi3 = L * 6
        
        # w'''(x) = d3w/dx3 = (1/L^3) * d3w/dxi3
        w_ddd = (1/L**3) * (d3Hv1_dxi3 * v1 + d3Ht1_dxi3 * theta1 + 
                            d3Hv2_dxi3 * v2 + d3Hv3_dxi3 * v3 + d3Ht3_dxi3 * theta3)
        
        return E * I * w_ddd

    def normal_force(self, x, displacements):
        """
        Returns normal (axial) force N(x) at position x (in local coordinates, 0 <= x <= L).
        Displacements should be in local coordinates: [u1, v1, theta1, u2, v2, u3, v3, theta3]
        """
        E = self.material.E
        A = self.section.area
        L = self.length
        u1 = displacements[0]
        u2 = displacements[3]
        u3 = displacements[5]
        
        xi = x / L
        
        # Derivatives of quadratic shape functions
        dN1_dxi = -3 + 4*xi
        dN2_dxi = 4 - 8*xi
        dN3_dxi = -1 + 4*xi
        
        # du/dx = (1/L) * du/dxi
        epsilon = (1/L) * (dN1_dxi * u1 + dN2_dxi * u2 + dN3_dxi * u3)
        
        return E * A * epsilon
    
class TimoshenkoElement2Node(Element):
    def __init__(self, id, node_start, node_end, material, section):
        super().__init__(id, node_start, node_end, material, section)
        self.length, self.c, self.s, self.R = self._compute_geometry()

    def _compute_geometry(self):
        x1, y1 = self.node_start.x, self.node_start.y
        x2, y2 = self.node_end.x, self.node_end.y
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        c = (x2 - x1) / L
        s = (y2 - y1) / L
        # Transformation matrix (same as Euler-Bernoulli for consistency)
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
        G = self.material.G
        A = self.section.area
        I = self.section.inertia
        kappa = self.section.shear_coefficient
        L = self.length
        R = self.R
        
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
        
        # Transform to global coordinates
        return R @ k_local @ R.T

    def force_vector(self, q_ini=0, q_fim=0, p_ini=0, p_fim=0):
        L = self.length
        R = self.R
        
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
        # Transform to global coordinates using self.R
        fe_global = self.R @ flocal
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

class TimoshenkoElement3Node(Element):
    """
    3-node Timoshenko beam element with quadratic shape functions.
    
    The element has 9 DOFs: [u1, v1, θ1, u2, v2, θ2, u3, v3, θ3]
    - All three nodes have rotation DOFs (unlike Euler-Bernoulli 3-node)
    - Axial: quadratic shape functions for u
    - Bending: quadratic shape functions for both v and θ (independent)
    - Includes shear deformation effects
    
    Shape functions (ξ = x/L):
    - N1 = (1-ξ)(1-2ξ)  (node 1)
    - N2 = 4ξ(1-ξ)      (node 2, center)
    - N3 = ξ(2ξ-1)      (node 3)
    """
    def __init__(self, id, node_start, node_end, material, section, node_center=None):
        super().__init__(id, node_start, node_end, material, section)
        self.node_center = node_center
        self.length, self.c, self.s, self.R = self._compute_geometry()
    
    def _compute_geometry(self):
        x1, y1 = self.node_start.x, self.node_start.y
        x3, y3 = self.node_end.x, self.node_end.y
        L = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
        c = (x3 - x1) / L
        s = (y3 - y1) / L
        
        # Transformation matrix for 3-node element (9x9)
        # DOFs: [u1, v1, θ1, u2, v2, θ2, u3, v3, θ3]
        # All nodes have rotation DOF
        R = np.zeros((9, 9))
        # Node 1: u1, v1, θ1
        R[0:2, 0:2] = np.array([[c, -s], [s, c]])
        R[2, 2] = 1
        # Node 2 (center): u2, v2, θ2
        R[3:5, 3:5] = np.array([[c, -s], [s, c]])
        R[5, 5] = 1
        # Node 3: u3, v3, θ3
        R[6:8, 6:8] = np.array([[c, -s], [s, c]])
        R[8, 8] = 1
        return L, c, s, R
    
    def stiffness_matrix(self):
        """
        Stiffness matrix for 3-node Timoshenko beam element.
        
        Uses quadratic shape functions for both displacement and rotation.
        Includes shear deformation effects through the shear coefficient.
        
        The stiffness matrix is computed using numerical integration (Gauss quadrature)
        of the strain energy contributions from:
        - Axial deformation: E*A*∫(du/dx)² dx
        - Bending: E*I*∫(dθ/dx)² dx
        - Shear: κ*G*A*∫(dv/dx - θ)² dx
        """
        E = self.material.E
        G = self.material.G
        A = self.section.area
        I = self.section.inertia
        kappa = self.section.shear_coefficient
        L = self.length
        R = self.R
        
        # Shear area
        As = kappa * A
        
        # Local stiffness matrix (9x9)
        k_local = np.zeros((9, 9))
        
        # Axial stiffness using quadratic shape functions
        # Shape functions: N1 = (1-ξ)(1-2ξ), N2 = 4ξ(1-ξ), N3 = ξ(2ξ-1)
        # where ξ = x/L
        k_axial = E * A / (3 * L) * np.array([
            [7, -8, 1],
            [-8, 16, -8],
            [1, -8, 7]
        ])
        
        # Assign axial stiffness to DOFs [u1, u2, u3] = [0, 3, 6]
        axial_dofs = [0, 3, 6]
        for i, ii in enumerate(axial_dofs):
            for j, jj in enumerate(axial_dofs):
                k_local[ii, jj] = k_axial[i, j]
        
        # For Timoshenko beam, we need to compute bending and shear stiffness
        # using numerical integration
        # Use selective reduced integration to avoid shear locking:
        # - Full integration (3-point Gauss) for bending stiffness
        # - Reduced integration (2-point Gauss) for shear stiffness
        
        # DOFs for bending: [v1, θ1, v2, θ2, v3, θ3] = indices [1, 2, 4, 5, 7, 8]
        bending_dofs = [1, 2, 4, 5, 7, 8]
        n_bending = len(bending_dofs)
        k_bending_shear = np.zeros((n_bending, n_bending))
        
        # Full integration (3-point) for bending stiffness
        # 3-point Gauss-Legendre quadrature points and weights for interval [-1,1]
        xi_gauss_full = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
        w_gauss_full = np.array([5/9, 8/9, 5/9])
        
        for xi_g, w_g in zip(xi_gauss_full, w_gauss_full):
            # Map from [-1,1] to [0,1]
            xi = (xi_g + 1) / 2
            
            # Quadratic shape functions for θ
            dN1_theta = (-3 + 4*xi) / L
            dN2_theta = (4 - 8*xi) / L
            dN3_theta = (-1 + 4*xi) / L
            
            # Bending stiffness: E*I*(dθ/dx)²
            dtheta_vec = np.array([0, dN1_theta, 0, dN2_theta, 0, dN3_theta])
            k_bending_shear += E * I * np.outer(dtheta_vec, dtheta_vec) * (L/2) * w_g
        
        # Reduced integration (2-point) for shear stiffness to avoid shear locking
        # 2-point Gauss-Legendre quadrature points and weights for interval [-1,1]
        xi_gauss_reduced = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        w_gauss_reduced = np.array([1.0, 1.0])
        
        for xi_g, w_g in zip(xi_gauss_reduced, w_gauss_reduced):
            # Map from [-1,1] to [0,1]
            xi = (xi_g + 1) / 2
            
            # Quadratic shape functions and derivatives for v
            N1_v = (1 - xi) * (1 - 2*xi)
            N2_v = 4 * xi * (1 - xi)
            N3_v = xi * (2*xi - 1)
            
            dN1_v = (-3 + 4*xi) / L
            dN2_v = (4 - 8*xi) / L
            dN3_v = (-1 + 4*xi) / L
            
            # Quadratic shape functions for θ
            N1_theta = (1 - xi) * (1 - 2*xi)
            N2_theta = 4 * xi * (1 - xi)
            N3_theta = xi * (2*xi - 1)
            
            # Shape function vectors for v and θ
            # v = [N1_v, 0, N2_v, 0, N3_v, 0]
            # θ = [0, N1_theta, 0, N2_theta, 0, N3_theta]
            # dv/dx = [dN1_v, 0, dN2_v, 0, dN3_v, 0]
            
            # Shear stiffness: κ*G*A*(dv/dx - θ)²
            dv_vec = np.array([dN1_v, 0, dN2_v, 0, dN3_v, 0])
            theta_vec = np.array([0, -N1_theta, 0, -N2_theta, 0, -N3_theta])
            gamma_vec = dv_vec + theta_vec  # dv/dx - θ
            k_bending_shear += As * G * np.outer(gamma_vec, gamma_vec) * (L/2) * w_g
        
        # Assign bending and shear stiffness
        for i, ii in enumerate(bending_dofs):
            for j, jj in enumerate(bending_dofs):
                k_local[ii, jj] = k_bending_shear[i, j]
        
        # Transform to global coordinates
        return R @ k_local @ R.T
    
    def force_vector(self, q_ini=0, q_fim=0, p_ini=0, p_fim=0):
        """
        Consistent nodal load vector for 3-node Timoshenko element.
        
        Args:
            q_ini: Initial axial distributed load (force per unit length)
            q_fim: Final axial distributed load
            p_ini: Initial transverse distributed load
            p_fim: Final transverse distributed load
            
        Returns:
            9-element force vector in global coordinates [u1, v1, θ1, u2, v2, θ2, u3, v3, θ3]
        """
        L = self.length
        R = self.R
        
        # Consistent load vector for linearly varying distributed loads
        # For quadratic axial shape functions
        q_avg = (q_ini + q_fim) / 2
        fe_axial = L * np.array([
            q_ini / 6,
            2 * q_avg / 3,
            q_fim / 6
        ])
        
        # For bending with quadratic shape functions
        # Using consistent load distribution for transverse load
        p_avg = (p_ini + p_fim) / 2
        fe_bending_v = L * np.array([
            (7*p_ini + 3*p_fim) / 20,
            (16*p_ini + 16*p_fim) / 70,
            (3*p_ini + 7*p_fim) / 20
        ])
        
        # For rotation DOFs, the consistent loads are typically zero
        # unless there are distributed moments
        fe_bending_theta = np.zeros(3)
        
        # Assemble into 9-DOF vector [u1, v1, θ1, u2, v2, θ2, u3, v3, θ3]
        fe_local = np.zeros(9)
        fe_local[0] = fe_axial[0]  # u1
        fe_local[1] = fe_bending_v[0]  # v1
        fe_local[2] = fe_bending_theta[0]  # θ1
        fe_local[3] = fe_axial[1]  # u2
        fe_local[4] = fe_bending_v[1]  # v2
        fe_local[5] = fe_bending_theta[1]  # θ2
        fe_local[6] = fe_axial[2]  # u3
        fe_local[7] = fe_bending_v[2]  # v3
        fe_local[8] = fe_bending_theta[2]  # θ3
        
        # Transform to global coordinates
        fe_global = R @ fe_local
        return fe_global.flatten()
    
    def compute_equivalent_nodal_loads(self, distributed_load, n_gauss=5):
        """
        Compute consistent nodal loads for a distributed load using numerical integration.
        Returns a 9-vector in GLOBAL coordinates [u1, v1, θ1, u2, v2, θ2, u3, v3, θ3].
        """
        import numpy as np
        L = self.length
        c = self.c
        s = self.s

        # Build load function f(x) from distributed_load
        if distributed_load.func:
            def f(x):
                try:
                    return float(eval(distributed_load.func, {"np": np, "x": x, "L": L}))
                except Exception as e:
                    print(f"Error evaluating custom function '{distributed_load.func}': {e}")
                    return 0.0
        elif distributed_load.magnitude_start is not None and distributed_load.magnitude_end is not None:
            a = float(distributed_load.magnitude_start)
            b = float(distributed_load.magnitude_end)
            def f(x):
                return a + (b - a) * (x / L)
        elif distributed_load.magnitude_start is not None:
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

        # Gauss-Legendre quadrature
        xi, wi = np.polynomial.legendre.leggauss(n_gauss)
        t = 0.5 * (xi + 1.0)
        wt = 0.5 * wi

        # Initialize force components
        ia1 = ia2 = ia3 = 0.0
        iv1 = itheta1 = iv2 = itheta2 = iv3 = itheta3 = 0.0
        
        for ti, wi_scaled in zip(t, wt):
            x = ti * L
            
            # Quadratic shape functions for axial
            N1_ax = (1 - ti) * (1 - 2*ti)
            N2_ax = 4 * ti * (1 - ti)
            N3_ax = ti * (2*ti - 1)
            qx = q_local(x)
            ia1 += N1_ax * qx * wi_scaled * L
            ia2 += N2_ax * qx * wi_scaled * L
            ia3 += N3_ax * qx * wi_scaled * L

            # Quadratic shape functions for bending (v and θ independent)
            N1_v = (1 - ti) * (1 - 2*ti)
            N2_v = 4 * ti * (1 - ti)
            N3_v = ti * (2*ti - 1)
            
            # For rotation, distributed loads typically don't contribute
            # unless there are distributed moments
            N1_theta = 0
            N2_theta = 0
            N3_theta = 0
            
            px = p_local(x)
            iv1 += N1_v * px * wi_scaled * L
            itheta1 += N1_theta * px * wi_scaled * L
            iv2 += N2_v * px * wi_scaled * L
            itheta2 += N2_theta * px * wi_scaled * L
            iv3 += N3_v * px * wi_scaled * L
            itheta3 += N3_theta * px * wi_scaled * L

        # Local consistent vector [u1, v1, theta1, u2, v2, theta2, u3, v3, theta3]
        flocal = np.array([ia1, iv1, itheta1, ia2, iv2, itheta2, ia3, iv3, itheta3], dtype=float)
        # Transform to global coordinates
        R = self.R
        fe_global = R @ flocal
        return fe_global.flatten()
    
    def bending_moment(self, x, displacements):
        """
        Returns bending moment M(x) at position x (in local coordinates, 0 <= x <= L).
        For Timoshenko beam theory: M(x) = E*I * dθ/dx
        """
        E = self.material.E
        I = self.section.inertia
        L = self.length
        
        # Local DOFs: [u1, v1, theta1, u2, v2, theta2, u3, v3, theta3]
        theta1 = displacements[2]
        theta2 = displacements[5]
        theta3 = displacements[8]
        
        xi = x / L
        
        # Derivatives of quadratic shape functions
        dN1_theta = (-3 + 4*xi) / L
        dN2_theta = (4 - 8*xi) / L
        dN3_theta = (-1 + 4*xi) / L
        
        # dθ/dx
        dtheta_dx = dN1_theta * theta1 + dN2_theta * theta2 + dN3_theta * theta3
        
        return E * I * dtheta_dx
    
    def shear_force(self, x, displacements):
        """
        Returns shear force V(x) at position x (in local coordinates, 0 <= x <= L).
        For Timoshenko beam theory: V = κ*G*A*(dv/dx - θ)
        """
        G = self.material.G
        A = self.section.area
        kappa = self.section.shear_coefficient
        L = self.length
        
        # Local DOFs: [u1, v1, theta1, u2, v2, theta2, u3, v3, theta3]
        v1 = displacements[1]
        theta1 = displacements[2]
        v2 = displacements[4]
        theta2 = displacements[5]
        v3 = displacements[7]
        theta3 = displacements[8]
        
        xi = x / L
        
        # Derivatives of quadratic shape functions for v
        dN1_v = (-3 + 4*xi) / L
        dN2_v = (4 - 8*xi) / L
        dN3_v = (-1 + 4*xi) / L
        
        # Quadratic shape functions for θ
        N1_theta = (1 - xi) * (1 - 2*xi)
        N2_theta = 4 * xi * (1 - xi)
        N3_theta = xi * (2*xi - 1)
        
        # dv/dx
        dv_dx = dN1_v * v1 + dN2_v * v2 + dN3_v * v3
        
        # θ
        theta = N1_theta * theta1 + N2_theta * theta2 + N3_theta * theta3
        
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
        
        # Local DOFs: [u1, v1, theta1, u2, v2, theta2, u3, v3, theta3]
        u1 = displacements[0]
        u2 = displacements[3]
        u3 = displacements[6]
        
        xi = x / L
        
        # Derivatives of quadratic shape functions
        dN1 = (-3 + 4*xi) / L
        dN2 = (4 - 8*xi) / L
        dN3 = (-1 + 4*xi) / L
        
        # du/dx
        epsilon = dN1 * u1 + dN2 * u2 + dN3 * u3
        
        return E * A * epsilon

class ElementResults:
    ...
    def bending_moment(self, x):
        return self.element.bending_moment(x, self.displacements)
    def shear_force(self, x):
        return self.element.shear_force(x, self.displacements)
    def normal_force(self, x):
        return self.element.normal_force(x, self.displacements)