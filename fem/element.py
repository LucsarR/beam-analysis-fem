import numpy as np
from abc import ABC, abstractmethod

def _quadratic_shape_functions_3node(xi):
    """Quadratic Lagrange shape functions on ξ ∈ [0, 1] for nodes [0, 0.5, 1]."""
    n1 = (1 - xi) * (1 - 2 * xi)
    n2 = 4 * xi * (1 - xi)
    n3 = xi * (2 * xi - 1)
    return np.array([n1, n2, n3], dtype=float)

def _quintic_bending_shapes_3node(xi, L):
    """
    Quintic Hermite-like interpolation for 3-node bending DOFs.

    Returns shape vectors for DOFs [v1, θ1, v2, θ2, v3, θ3]:
      n_w:      w interpolation
      dn_w_dx:  dw/dx interpolation
      d2_w_dx2: d²w/dx² interpolation
      d3_w_dx3: d³w/dx³ interpolation

    Also returns (n_theta, dn_theta_dx) for a quintic interpolation of
    θ using nodal rotations [θ1, θ2, θ3].
    """
    # Quintic basis (ξ = x/L)
    n1 = 24 * xi**5 - 68 * xi**4 + 66 * xi**3 - 23 * xi**2 + 1
    m1 = 4 * xi**5 - 12 * xi**4 + 13 * xi**3 - 6 * xi**2 + xi
    n2 = 16 * xi**4 - 32 * xi**3 + 16 * xi**2
    m2 = 16 * xi**5 - 40 * xi**4 + 32 * xi**3 - 8 * xi**2
    n3 = -24 * xi**5 + 52 * xi**4 - 34 * xi**3 + 7 * xi**2
    m3 = 4 * xi**5 - 8 * xi**4 + 5 * xi**3 - xi**2

    dn1 = 120 * xi**4 - 272 * xi**3 + 198 * xi**2 - 46 * xi
    dm1 = 20 * xi**4 - 48 * xi**3 + 39 * xi**2 - 12 * xi + 1
    dn2 = 64 * xi**3 - 96 * xi**2 + 32 * xi
    dm2 = 80 * xi**4 - 160 * xi**3 + 96 * xi**2 - 16 * xi
    dn3 = -120 * xi**4 + 208 * xi**3 - 102 * xi**2 + 14 * xi
    dm3 = 20 * xi**4 - 32 * xi**3 + 15 * xi**2 - 2 * xi

    d2n1 = 480 * xi**3 - 816 * xi**2 + 396 * xi - 46
    d2m1 = 80 * xi**3 - 144 * xi**2 + 78 * xi - 12
    d2n2 = 192 * xi**2 - 192 * xi + 32
    d2m2 = 320 * xi**3 - 480 * xi**2 + 192 * xi - 16
    d2n3 = -480 * xi**3 + 624 * xi**2 - 204 * xi + 14
    d2m3 = 80 * xi**3 - 96 * xi**2 + 30 * xi - 2

    d3n1 = 1440 * xi**2 - 1632 * xi + 396
    d3m1 = 240 * xi**2 - 288 * xi + 78
    d3n2 = 384 * xi - 192
    d3m2 = 960 * xi**2 - 960 * xi + 192
    d3n3 = -1440 * xi**2 + 1248 * xi - 204
    d3m3 = 240 * xi**2 - 192 * xi + 30

    n_w = np.array([n1, L * m1, n2, L * m2, n3, L * m3], dtype=float)
    dn_w_dx = np.array([dn1 / L, dm1, dn2 / L, dm2, dn3 / L, dm3], dtype=float)
    d2_w_dx2 = np.array([d2n1 / (L**2), d2m1 / L, d2n2 / (L**2), d2m2 / L, d2n3 / (L**2), d2m3 / L], dtype=float)
    d3_w_dx3 = np.array([d3n1 / (L**3), d3m1 / (L**2), d3n2 / (L**3), d3m2 / (L**2), d3n3 / (L**3), d3m3 / (L**2)], dtype=float)

    n_theta = np.array([n1, n2, n3], dtype=float)
    dn_theta_dx = np.array([dn1 / L, dn2 / L, dn3 / L], dtype=float)

    return n_w, dn_w_dx, d2_w_dx2, d3_w_dx3, n_theta, dn_theta_dx

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

    def _recover_local_nodal_forces(self, displacements):
        """Recover local nodal force vector f_local = K_local · d_local."""
        d_local = np.asarray(displacements, dtype=float)
        k_local = self.R.T @ self.stiffness_matrix() @ self.R
        return k_local @ d_local


class EulerBernoulliElement2Node(Element):
    dofs_per_node = 3  # Each node has [u, v, θ] DOFs

    def __init__(self, id, node_start, node_end, material, section, stiffness_integration="analytical"):
        super().__init__(id, node_start, node_end, material, section)
        self.stiffness_integration = stiffness_integration
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
        if self.stiffness_integration == "numerical":
            return self._stiffness_matrix_numerical()
        if self.stiffness_integration != "analytical":
            raise ValueError(
                f"Unsupported stiffness integration mode '{self.stiffness_integration}' "
                f"for {self.__class__.__name__}"
            )

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

    def _stiffness_matrix_numerical(self):
        E = self.material.E
        A = self.section.area
        I = self.section.inertia
        L = self.length
        R = self.R

        k_local = np.zeros((6, 6))

        # 3-point Gauss-Legendre is exact here for the polynomial terms from
        # Hermite-cubic Euler-Bernoulli interpolation.
        xi_gauss = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
        w_gauss = np.array([5/9, 8/9, 5/9])

        for xi_g, w_g in zip(xi_gauss, w_gauss):
            xi = 0.5 * (xi_g + 1.0)
            jac = L / 2.0

            b_axial = np.array([-1.0 / L, 0.0, 0.0, 1.0 / L, 0.0, 0.0])

            d2n1 = (-6.0 + 12.0 * xi) / (L ** 2)
            d2n2 = (-4.0 + 6.0 * xi) / L
            d2n3 = (6.0 - 12.0 * xi) / (L ** 2)
            d2n4 = (-2.0 + 6.0 * xi) / L
            b_bending = np.array([0.0, d2n1, d2n2, 0.0, d2n3, d2n4])

            k_local += E * A * np.outer(b_axial, b_axial) * jac * w_g
            k_local += E * I * np.outer(b_bending, b_bending) * jac * w_g

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

        For custom functions (func), the variable ``x`` passed to the expression
        is the **global** position along the beam (i.e. the x-coordinate of the
        point on the element in the global frame).  This allows a single
        expression such as ``60000*(3*(x/4)**2 - 2*(x/4)**3)`` to describe a
        load that varies over the full beam length even when the mesh has
        multiple elements.  ``L`` in the expression refers to the length of
        the current element.
        """
        import numpy as np
        L = self.length
        c = self.c
        s = self.s

        # Build load function f(x_local) from distributed_load.
        # x_local is the local coordinate within the element (0 to L).
        if distributed_load.func:
            # Custom function: evaluate using the global x position so that
            # load expressions written in terms of the full-beam coordinate
            # work correctly for any element in a multi-element mesh.
            x_start = self.node_start.x
            def f(x_local):
                x_global = x_start + x_local * c
                try:
                    return float(eval(distributed_load.func, {"np": np, "x": x_global, "L": L}))
                except Exception as e:
                    print(f"Error evaluating custom function '{distributed_load.func}': {e}")
                    return 0.0
        elif distributed_load.magnitude_start is not None and distributed_load.magnitude_end is not None:
            # Linear
            a = float(distributed_load.magnitude_start)
            b = float(distributed_load.magnitude_end)
            def f(x_local):
                return a + (b - a) * (x_local / L)
        elif distributed_load.magnitude_start is not None:
            # Constant
            a = float(distributed_load.magnitude_start)
            def f(x_local):
                return a
        else:
            def f(x_local):
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
    dofs_per_node = 3  # Each node has [u, v, θ] DOFs

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
        # Transformation matrix for 3-node element (9x9)
        # DOFs: [u1, v1, θ1, u2, v2, θ2, u3, v3, θ3]
        # Central node now has rotation DOF
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
        Stiffness matrix for 3-node Euler-Bernoulli beam element.
        
        The element has 9 DOFs: [u1, v1, θ1, u2, v2, θ2, u3, v3, θ3]
        - Axial: quadratic shape functions for u
        - Bending: quintic interpolation for [v1, θ1, v2, θ2, v3, θ3]
          with Euler-Bernoulli curvature κ = d²v/dx²
        
        References:
        - Reddy, J.N. "An Introduction to the Finite Element Method" (2006)
        """
        E = self.material.E
        A = self.section.area
        I = self.section.inertia
        L = self.length
        R = self.R
        
        # Local stiffness matrix (9x9)
        k_local = np.zeros((9, 9))
        
        # Axial stiffness using quadratic shape functions
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
        
        # Bending stiffness for Euler-Bernoulli beam with quintic bending interpolation
        bending_dofs = [1, 2, 4, 5, 7, 8]
        n_bending = len(bending_dofs)
        k_bending = np.zeros((n_bending, n_bending))
        
        # 5-point Gauss integration on ξ ∈ [0, 1]
        xi_g, w_g = np.polynomial.legendre.leggauss(5)
        t = 0.5 * (xi_g + 1.0)
        wt = 0.5 * w_g
        for xi, wi in zip(t, wt):
            _, _, d2_w_dx2, _, _, _ = _quintic_bending_shapes_3node(xi, L)
            k_bending += E * I * np.outer(d2_w_dx2, d2_w_dx2) * wi * L
        
        # Assign bending stiffness to DOFs [v1, θ1, v2, θ2, v3, θ3]
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
            9-element force vector in global coordinates [u1, v1, θ1, u2, v2, θ2, u3, v3, θ3]
        """
        L = self.length
        R = self.R
        
        fe_local = np.zeros(9)
        xi_g, w_g = np.polynomial.legendre.leggauss(5)
        t = 0.5 * (xi_g + 1.0)
        wt = 0.5 * w_g

        for xi, wi in zip(t, wt):
            qx = q_ini + (q_fim - q_ini) * xi
            px = p_ini + (p_fim - p_ini) * xi

            n_ax = _quadratic_shape_functions_3node(xi)
            n_w, _, _, _, _, _ = _quintic_bending_shapes_3node(xi, L)

            fe_local[[0, 3, 6]] += n_ax * qx * wi * L
            fe_local[[1, 2, 4, 5, 7, 8]] += n_w * px * wi * L
        
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

        # Build load function f(x_local) from distributed_load.
        # x_local is the local coordinate within the element (0 to L).
        if distributed_load.func:
            # Custom function: evaluate using the global x position so that
            # load expressions written in terms of the full-beam coordinate
            # work correctly for any element in a multi-element mesh.
            x_start = self.node_start.x
            def f(x_local):
                x_global = x_start + x_local * c
                try:
                    return float(eval(distributed_load.func, {"np": np, "x": x_global, "L": L}))
                except Exception as e:
                    print(f"Error evaluating custom function '{distributed_load.func}': {e}")
                    return 0.0
        elif distributed_load.magnitude_start is not None and distributed_load.magnitude_end is not None:
            # Linear
            a = float(distributed_load.magnitude_start)
            b = float(distributed_load.magnitude_end)
            def f(x_local):
                return a + (b - a) * (x_local / L)
        elif distributed_load.magnitude_start is not None:
            # Constant
            a = float(distributed_load.magnitude_start)
            def f(x_local):
                return a
        else:
            def f(x_local):
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
            N1_ax, N2_ax, N3_ax = _quadratic_shape_functions_3node(ti)
            qx = q_local(x)
            ia1 += N1_ax * qx * wi_scaled * L
            ia2 += N2_ax * qx * wi_scaled * L
            ia3 += N3_ax * qx * wi_scaled * L

            # Quintic bending interpolation for [v1, θ1, v2, θ2, v3, θ3]
            n_w, _, _, _, _, _ = _quintic_bending_shapes_3node(ti, L)
            px = p_local(x)
            iv1 += n_w[0] * px * wi_scaled * L
            itheta1 += n_w[1] * px * wi_scaled * L
            iv2 += n_w[2] * px * wi_scaled * L
            itheta2 += n_w[3] * px * wi_scaled * L
            iv3 += n_w[4] * px * wi_scaled * L
            itheta3 += n_w[5] * px * wi_scaled * L

        # Local consistent vector [u1, v1, theta1, u2, v2, theta2, u3, v3, theta3]
        flocal = np.array([ia1, iv1, itheta1, ia2, iv2, itheta2, ia3, iv3, itheta3], dtype=float)
        # Transform to global coordinates
        R = self.R
        fe_global = R @ flocal
        return fe_global.flatten()

    def bending_moment(self, x, displacements):
        """
        Returns bending moment M(x) at position x (in local coordinates, 0 <= x <= L).
        For the 3-node Euler-Bernoulli element, recover end moments from element
        equilibrium (K_local · d_local) and linearly interpolate between ends.
        """
        E = self.material.E
        I = self.section.inertia
        L = self.length
        xi = x / L
        d_local = np.asarray(displacements, dtype=float)
        d_bending = d_local[[1, 2, 4, 5, 7, 8]]
        _, _, d2_w_dx2, _, _, _ = _quintic_bending_shapes_3node(xi, L)
        return E * I * np.dot(d2_w_dx2, d_bending)

    def shear_force(self, x, displacements):
        """
        Returns shear force V(x) at position x (in local coordinates, 0 <= x <= L).
        Recover end shears from element equilibrium (K_local · d_local) and
        linearly interpolate between element ends.
        """
        E = self.material.E
        I = self.section.inertia
        L = self.length
        xi = x / L
        d_local = np.asarray(displacements, dtype=float)
        d_bending = d_local[[1, 2, 4, 5, 7, 8]]
        _, _, _, d3_w_dx3, _, _ = _quintic_bending_shapes_3node(xi, L)
        return E * I * np.dot(d3_w_dx3, d_bending)

    def _recover_local_nodal_forces(self, displacements):
        """Recover local nodal force vector f_local = K_local · d_local."""
        d_local = np.asarray(displacements, dtype=float)
        k_local = self.R.T @ self.stiffness_matrix() @ self.R
        return k_local @ d_local

    def normal_force(self, x, displacements):
        """
        Returns normal (axial) force N(x) at position x (in local coordinates, 0 <= x <= L).
        Displacements should be in local coordinates: [u1, v1, theta1, u2, v2, theta2, u3, v3, theta3]
        """
        E = self.material.E
        A = self.section.area
        L = self.length
        u1 = displacements[0]
        u2 = displacements[3]
        u3 = displacements[6]
        
        xi = x / L
        
        # Derivatives of quadratic shape functions
        dN1_dxi = -3 + 4*xi
        dN2_dxi = 4 - 8*xi
        dN3_dxi = -1 + 4*xi
        
        # du/dx = (1/L) * du/dxi
        epsilon = (1/L) * (dN1_dxi * u1 + dN2_dxi * u2 + dN3_dxi * u3)
        
        return E * A * epsilon
    
class TimoshenkoElement2Node(Element):
    dofs_per_node = 3  # Each node has [u, v, θ] DOFs

    def __init__(self, id, node_start, node_end, material, section, stiffness_integration="analytical"):
        super().__init__(id, node_start, node_end, material, section)
        self.stiffness_integration = stiffness_integration
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
        if self.stiffness_integration == "numerical":
            return self._stiffness_matrix_numerical()
        if self.stiffness_integration != "analytical":
            raise ValueError(
                f"Unsupported stiffness integration mode '{self.stiffness_integration}' "
                f"for {self.__class__.__name__}"
            )

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

    def _stiffness_matrix_numerical(self):
        E = self.material.E
        G = self.material.G
        A = self.section.area
        I = self.section.inertia
        kappa = self.section.shear_coefficient
        L = self.length
        R = self.R

        k_local = np.zeros((6, 6))
        As = kappa * A

        # Axial + bending terms with 2-point Gauss integration.
        # This is exact for the linear shape-function derivatives used here.
        xi_gauss = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
        w_gauss = np.array([1.0, 1.0])
        for xi_g, w_g in zip(xi_gauss, w_gauss):
            xi = 0.5 * (xi_g + 1.0)
            jac = L / 2.0

            b_axial = np.array([-1.0 / L, 0.0, 0.0, 1.0 / L, 0.0, 0.0])
            b_bending = np.array([0.0, 0.0, -1.0 / L, 0.0, 0.0, 1.0 / L])

            k_local += E * A * np.outer(b_axial, b_axial) * jac * w_g
            k_local += E * I * np.outer(b_bending, b_bending) * jac * w_g

        # Reduced integration for shear term (single point) to mitigate
        # shear locking (artificially stiff transverse response in slender beams).
        xi = 0.5
        jac = L
        n1 = 1.0 - xi
        n2 = xi
        b_shear = np.array([0.0, -1.0 / L, -n1, 0.0, 1.0 / L, -n2])
        k_local += As * G * np.outer(b_shear, b_shear) * jac

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

        For custom functions (func), the variable ``x`` passed to the expression
        is the **global** position along the beam (i.e. the x-coordinate of the
        point on the element in the global frame).  ``L`` in the expression
        refers to the length of the current element.
        """
        import numpy as np
        L = self.length
        c = self.c
        s = self.s

        # Build load function f(x_local) from distributed_load.
        # x_local is the local coordinate within the element (0 to L).
        if distributed_load.func:
            # Custom function: evaluate using the global x position so that
            # load expressions written in terms of the full-beam coordinate
            # work correctly for any element in a multi-element mesh.
            x_start = self.node_start.x
            def f(x_local):
                x_global = x_start + x_local * c
                try:
                    return float(eval(distributed_load.func, {"np": np, "x": x_global, "L": L}))
                except Exception as e:
                    print(f"Error evaluating custom function '{distributed_load.func}': {e}")
                    return 0.0
        elif distributed_load.magnitude_start is not None and distributed_load.magnitude_end is not None:
            # Linear
            a = float(distributed_load.magnitude_start)
            b = float(distributed_load.magnitude_end)
            def f(x_local):
                return a + (b - a) * (x_local / L)
        elif distributed_load.magnitude_start is not None:
            # Constant
            a = float(distributed_load.magnitude_start)
            def f(x_local):
                return a
        else:
            def f(x_local):
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
        For the 2-node Timoshenko beam element.

        The nodal bending moments are recovered from the element equilibrium
        (K_local · d_local) and then linearly interpolated:

            M(0) = −f[2]  (start-node moment from stiffness row 2)
            M(L) =  f[5]  (end-node moment from stiffness row 5)
            M(x) = M(0)·(1−ξ) + M(L)·ξ,   ξ = x/L

        This is consistent with the constant field-consistent shear force
        (dM/dx = V = const) and reproduces the correct linear variation for
        beams without distributed loads.  The midpoint value equals EI·dθ/dx,
        so existing midpoint-based checks are unaffected.
        """
        E = self.material.E
        G = self.material.G
        I = self.section.inertia
        A = self.section.area
        kappa = self.section.shear_coefficient
        L = self.length
        # Local DOFs: [u1, v1, theta1, u2, v2, theta2]
        v1 = displacements[1]
        theta1 = displacements[2]
        v2 = displacements[4]
        theta2 = displacements[5]

        # Shear-flexibility parameter (same as in stiffness_matrix)
        phi = 12.0 * E * I / (G * kappa * A * L**2)

        # Nodal moments from K_local · d_local (rows 2 and 5):
        #   f[2] = EI/(L(1+φ)) · [6/L·(v1−v2) + (4+φ)·θ1 + (2−φ)·θ2]
        #   f[5] = EI/(L(1+φ)) · [6/L·(v1−v2) + (2−φ)·θ1 + (4+φ)·θ2]
        # Sign convention (same as Euler-Bernoulli): M(0) = −f[2], M(L) = f[5]
        coeff = E * I / (L * (1.0 + phi))
        f2 = coeff * (6.0 / L * (v1 - v2) + (4.0 + phi) * theta1 + (2.0 - phi) * theta2)
        f5 = coeff * (6.0 / L * (v1 - v2) + (2.0 - phi) * theta1 + (4.0 + phi) * theta2)

        M0 = -f2   # internal moment at x = 0
        ML = f5    # internal moment at x = L

        xi = x / L
        return M0 * (1.0 - xi) + ML * xi
    
    def shear_force(self, x, displacements):
        """
        Returns shear force V(x) at position x (in local coordinates, 0 <= x <= L).
        For the field-consistent 2-node Timoshenko beam element.

        The raw kinematic formula V = kGA*(dw/dx - θ) is not used here because
        the 2-node element uses linear interpolation for both w and θ, making the
        shear strain artificially large (shear locking in the post-processing sense).
        Instead the shear force is recovered from the element stiffness equilibrium:

            V = 12·EI/(L³·(1+φ)) · (v₁ − v₂)  +  6·EI/(L²·(1+φ)) · (θ₁ + θ₂)

        where  φ = 12·EI / (κGA·L²)  is the shear-flexibility parameter of the
        element.  This equals the y-direction nodal force from K·u at node 1, and
        is constant along the element (consistent with linear shear interpolation).
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

        # Shear-flexibility parameter (same as in stiffness_matrix)
        phi = 12.0 * E * I / (G * kappa * A * L ** 2)

        # Field-consistent shear force (constant along element)
        V = (12.0 * E * I / (L ** 3 * (1.0 + phi))) * (v1 - v2) \
            + (6.0 * E * I / (L ** 2 * (1.0 + phi))) * (theta1 + theta2)
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
    3-node Timoshenko beam element with decoupled interpolation orders.

    The element has 9 DOFs: [u1, v1, θ1, u2, v2, θ2, u3, v3, θ3]
    - All three nodes have rotation DOFs (unlike Euler-Bernoulli 3-node)
    - Axial: quadratic shape functions for u
    - Bending: quintic interpolation for v and quadratic interpolation for θ
    - Includes shear deformation effects
    """
    dofs_per_node = 3  # Each node has [u, v, θ] DOFs

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
        
        Uses decoupled interpolation:
        - axial u: quadratic
        - bending v: quintic
        - rotation θ: quadratic
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
            dn_theta_dx = np.array([(-3 + 4*xi) / L, (4 - 8*xi) / L, (-1 + 4*xi) / L], dtype=float)
            
            # Bending stiffness: E*I*(dθ/dx)²
            dtheta_vec = np.array([0, dn_theta_dx[0], 0, dn_theta_dx[1], 0, dn_theta_dx[2]])
            k_bending_shear += E * I * np.outer(dtheta_vec, dtheta_vec) * (L/2) * w_g
        
        # Reduced integration (2-point) for shear stiffness to avoid shear locking
        # 2-point Gauss-Legendre quadrature points and weights for interval [-1,1]
        xi_gauss_reduced = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        w_gauss_reduced = np.array([1.0, 1.0])
        
        for xi_g, w_g in zip(xi_gauss_reduced, w_gauss_reduced):
            # Map from [-1,1] to [0,1]
            xi = (xi_g + 1) / 2
            
            _, dn_w_dx, _, _, _, _ = _quintic_bending_shapes_3node(xi, L)
            n_theta = _quadratic_shape_functions_3node(xi)

            # Shear stiffness: κ*G*A*(dv/dx - θ)²
            dv_vec = dn_w_dx
            theta_vec = np.array([0, n_theta[0], 0, n_theta[1], 0, n_theta[2]])
            gamma_vec = dv_vec - theta_vec
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
        
        fe_local = np.zeros(9)
        xi_g, w_g = np.polynomial.legendre.leggauss(5)
        t = 0.5 * (xi_g + 1.0)
        wt = 0.5 * w_g

        for xi, wi in zip(t, wt):
            qx = q_ini + (q_fim - q_ini) * xi
            px = p_ini + (p_fim - p_ini) * xi

            n_ax = _quadratic_shape_functions_3node(xi)
            n_w, _, _, _, _, _ = _quintic_bending_shapes_3node(xi, L)

            fe_local[[0, 3, 6]] += n_ax * qx * wi * L
            fe_local[[1, 2, 4, 5, 7, 8]] += n_w * px * wi * L
        
        # Transform to global coordinates
        fe_global = R @ fe_local
        return fe_global.flatten()
    
    def compute_equivalent_nodal_loads(self, distributed_load, n_gauss=5):
        """
        Compute consistent nodal loads for a distributed load using numerical integration.
        Returns a 9-vector in GLOBAL coordinates [u1, v1, θ1, u2, v2, θ2, u3, v3, θ3].

        For custom functions (func), the variable ``x`` passed to the expression
        is the **global** position along the beam (i.e. the x-coordinate of the
        point on the element in the global frame).  ``L`` in the expression
        refers to the length of the current element.
        """
        import numpy as np
        L = self.length
        c = self.c
        s = self.s

        # Build load function f(x_local) from distributed_load.
        # x_local is the local coordinate within the element (0 to L).
        if distributed_load.func:
            # Custom function: evaluate using the global x position so that
            # load expressions written in terms of the full-beam coordinate
            # work correctly for any element in a multi-element mesh.
            x_start = self.node_start.x
            def f(x_local):
                x_global = x_start + x_local * c
                try:
                    return float(eval(distributed_load.func, {"np": np, "x": x_global, "L": L}))
                except Exception as e:
                    print(f"Error evaluating custom function '{distributed_load.func}': {e}")
                    return 0.0
        elif distributed_load.magnitude_start is not None and distributed_load.magnitude_end is not None:
            a = float(distributed_load.magnitude_start)
            b = float(distributed_load.magnitude_end)
            def f(x_local):
                return a + (b - a) * (x_local / L)
        elif distributed_load.magnitude_start is not None:
            a = float(distributed_load.magnitude_start)
            def f(x_local):
                return a
        else:
            def f(x_local):
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
            N1_ax, N2_ax, N3_ax = _quadratic_shape_functions_3node(ti)
            qx = q_local(x)
            ia1 += N1_ax * qx * wi_scaled * L
            ia2 += N2_ax * qx * wi_scaled * L
            ia3 += N3_ax * qx * wi_scaled * L

            # Quintic bending interpolation for [v1, θ1, v2, θ2, v3, θ3]
            n_w, _, _, _, _, _ = _quintic_bending_shapes_3node(ti, L)
            px = p_local(x)
            iv1 += n_w[0] * px * wi_scaled * L
            itheta1 += n_w[1] * px * wi_scaled * L
            iv2 += n_w[2] * px * wi_scaled * L
            itheta2 += n_w[3] * px * wi_scaled * L
            iv3 += n_w[4] * px * wi_scaled * L
            itheta3 += n_w[5] * px * wi_scaled * L

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
        
        dn_theta_dx = np.array([(-3 + 4*xi) / L, (4 - 8*xi) / L, (-1 + 4*xi) / L], dtype=float)
        
        # dθ/dx
        dtheta_dx = dn_theta_dx[0] * theta1 + dn_theta_dx[1] * theta2 + dn_theta_dx[2] * theta3
        
        return E * I * dtheta_dx
    
    def shear_force(self, x, displacements):
        """
        Returns shear force V(x) at position x (in local coordinates, 0 <= x <= L).
        For the 3-node Timoshenko element, recover end shears from element
        equilibrium (K_local · d_local) and linearly interpolate between ends.
        """
        L = self.length
        xi = x / L

        f_local = self._recover_local_nodal_forces(displacements)
        V0 = f_local[1]
        VL = -f_local[7]
        return V0 * (1.0 - xi) + VL * xi
    
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
class ReddyBickfordElement2Node(Element):
    """
    2-node Reddy-Bickford (RBT) beam element.

    Based on Reddy's third-order shear deformation theory (TSDT) with
    *independent* (uncoupled) interpolations: Hermite cubics for transverse
    displacement v and linear shape functions for the rotation θ.

    Each node has 4 DOFs in global ordering [u, v, θ, dv/dx]:
      - u     : axial displacement
      - v     : transverse displacement
      - θ     : independent rotation of the cross-section
      - dv/dx : slope (derivative of transverse displacement)

    Element local DOF vector (8 components):
      [u₁, v₁, θ₁, (dv/dx)₁, u₂, v₂, θ₂, (dv/dx)₂]

    The 8×8 stiffness matrix is derived from Reddy's TSDT strain energy.

    Reference: Heyliger, P.R. and Reddy, J.N., "A Higher Order Beam Finite
    Element for Bending and Vibration Problems," Journal of Sound and
    Vibration, 126(2), 309-326, 1988.
    """

    dofs_per_node = 4

    def __init__(self, id, node_start, node_end, material, section):
        super().__init__(id, node_start, node_end, material, section)
        x1, y1 = node_start.x, node_start.y
        x2, y2 = node_end.x, node_end.y
        L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        c = (x2 - x1) / L
        s = (y2 - y1) / L
        # 8×8 transformation matrix
        R = np.array([
            [c, -s, 0, 0, 0,  0, 0, 0],
            [s,  c, 0, 0, 0,  0, 0, 0],
            [0,  0, 1, 0, 0,  0, 0, 0],
            [0,  0, 0, 1, 0,  0, 0, 0],
            [0,  0, 0, 0, c, -s, 0, 0],
            [0,  0, 0, 0, s,  c, 0, 0],
            [0,  0, 0, 0, 0,  0, 1, 0],
            [0,  0, 0, 0, 0,  0, 0, 1],
        ], dtype=float)
        self.length = L
        self.c = c
        self.s = s
        self.R = R

    def _get_height(self):
        """Return section height used in Reddy TSDT parameter c₁ = 4/(3h²)."""
        sec = self.section
        if hasattr(sec, 'height'):
            return float(sec.height)
        return float(np.sqrt(12.0 * sec.inertia / sec.area))

    def _get_reddy_params(self):
        """
        Compute the four modified stiffness parameters for the Reddy TSDT.
        """
        E = self.material.E
        G = self.material.G
        A = self.section.area
        I = self.section.inertia
        h = self._get_height()
        b = A / h

        c1 = 4.0 / (3.0 * h ** 2)
        c3 = 3.0 * c1

        I4 = b * h ** 5 / 80.0
        I6 = b * h ** 7 / 448.0

        A_bar = A - 2.0 * c3 * I + c3 ** 2 * I4

        D1 = E * (I - 2.0 * c1 * I4 + c1 ** 2 * I6)
        E1 = E * c1 * (I4 - c1 * I6)
        F1 = E * c1 ** 2 * I6
        G1 = G * A_bar

        return D1, E1, F1, G1

    # ------------------------------------------------------------------
    # Core stiffness assembly – extracted into a single method so that
    # subclasses (MRBTElement2Node) can override only this while inheriting
    # all other behaviour (force vectors, moment / shear recovery, etc.).
    # ------------------------------------------------------------------

    def _build_local_stiffness(self):
        """
        Build and return the full 8×8 LOCAL stiffness matrix (before rotation).

        Local DOF ordering: [u₁, v₁, θ₁, (dv/dx)₁, u₂, v₂, θ₂, (dv/dx)₂]

        The 6×6 bending sub-matrix K_b uses the sequential node-grouped DOF ordering:
          [v₁, θ₁, (dv/dx)₁, v₂, θ₂, (dv/dx)₂]
        and is scattered into the 8×8 via b_dofs = [1, 2, 3, 5, 6, 7].

        Strain energy terms:
          U = ½∫[D1·(θ')² − 2E1·θ'·v'' + F1·(v'')² + G1·(θ − v')²] dx
        """
        E = self.material.E
        A = self.section.area
        L = self.length

        D1, E1_param, F1, G1 = self._get_reddy_params()

        # Sequential bending DOF ordering → 8×8 index mapping
        b_dofs = [1, 2, 3, 5, 6, 7]
        K_b = np.zeros((6, 6))

        # --- D1 term: ∫D1·(dθ/dx)² dx  (linear θ → constant derivative) ---
        K_b[1, 1] += D1 / L
        K_b[1, 4] += -D1 / L
        K_b[4, 1] += -D1 / L
        K_b[4, 4] += D1 / L

        # --- F1 term: ∫F1·(v'')² dx  (Hermite cubic → EB-like 4×4 mapped to 0, 2, 3, 5) ---
        f = F1 / L ** 3
        K_b[0, 0] += 12.0 * f;  K_b[0, 2] += 6.0 * L * f
        K_b[0, 3] += -12.0 * f; K_b[0, 5] += 6.0 * L * f
        K_b[2, 0] += 6.0 * L * f;  K_b[2, 2] += 4.0 * L ** 2 * f
        K_b[2, 3] += -6.0 * L * f; K_b[2, 5] += 2.0 * L ** 2 * f
        K_b[3, 0] += -12.0 * f; K_b[3, 2] += -6.0 * L * f
        K_b[3, 3] += 12.0 * f;  K_b[3, 5] += -6.0 * L * f
        K_b[5, 0] += 6.0 * L * f;  K_b[5, 2] += 2.0 * L ** 2 * f
        K_b[5, 3] += -6.0 * L * f; K_b[5, 5] += 4.0 * L ** 2 * f

        # --- E1 cross term: −∫E1·(dθ/dx·d²v/dx² + d²v/dx²·dθ/dx) dx ---
        K_b[1, 2] += -E1_param / L;   K_b[2, 1] += -E1_param / L
        K_b[1, 5] += E1_param / L;    K_b[5, 1] += E1_param / L
        K_b[2, 4] += E1_param / L;    K_b[4, 2] += E1_param / L
        K_b[4, 5] += -E1_param / L;   K_b[5, 4] += -E1_param / L

        # --- G1 shear term: ∫G1·(θ − v')² dx ---
        K_b[0, 0] += 6.0 * G1 / (5.0 * L)
        K_b[0, 1] += -G1 / 2.0
        K_b[0, 2] += G1 / 10.0
        K_b[0, 3] += -6.0 * G1 / (5.0 * L)
        K_b[0, 4] += -G1 / 2.0
        K_b[0, 5] += G1 / 10.0

        K_b[1, 0] += -G1 / 2.0
        K_b[1, 1] += G1 * L / 3.0
        K_b[1, 2] += G1 * L / 12.0
        K_b[1, 3] += G1 / 2.0
        K_b[1, 4] += G1 * L / 6.0
        K_b[1, 5] += -G1 * L / 12.0

        K_b[2, 0] += G1 / 10.0
        K_b[2, 1] += G1 * L / 12.0
        K_b[2, 2] += 2.0 * G1 * L / 15.0
        K_b[2, 3] += -G1 / 10.0
        K_b[2, 4] += -G1 * L / 12.0
        K_b[2, 5] += -G1 * L / 30.0

        K_b[3, 0] += -6.0 * G1 / (5.0 * L)
        K_b[3, 1] += G1 / 2.0
        K_b[3, 2] += -G1 / 10.0
        K_b[3, 3] += 6.0 * G1 / (5.0 * L)
        K_b[3, 4] += G1 / 2.0
        K_b[3, 5] += -G1 / 10.0

        K_b[4, 0] += -G1 / 2.0
        K_b[4, 1] += G1 * L / 6.0
        K_b[4, 2] += -G1 * L / 12.0
        K_b[4, 3] += G1 / 2.0
        K_b[4, 4] += G1 * L / 3.0
        K_b[4, 5] += G1 * L / 12.0

        K_b[5, 0] += G1 / 10.0
        K_b[5, 1] += -G1 * L / 12.0
        K_b[5, 2] += -G1 * L / 30.0
        K_b[5, 3] += -G1 / 10.0
        K_b[5, 4] += G1 * L / 12.0
        K_b[5, 5] += 2.0 * G1 * L / 15.0

        # Build full 8×8
        k_local = np.zeros((8, 8))
        k_local[0, 0] = E * A / L
        k_local[0, 4] = -E * A / L
        k_local[4, 0] = -E * A / L
        k_local[4, 4] = E * A / L

        for i_b, i_g in enumerate(b_dofs):
            for j_b, j_g in enumerate(b_dofs):
                k_local[i_g, j_g] = K_b[i_b, j_b]

        return k_local

    def stiffness_matrix(self):
        """Build the 8×8 element stiffness matrix in global coordinates."""
        return self.R @ self._build_local_stiffness() @ self.R.T

    def force_vector(self, q_ini=0, q_fim=0, p_ini=0, p_fim=0):
        """
        Consistent element load vector for uniformly or linearly varying loads.

        q_ini, q_fim : axial distributed load at start / end node (per unit length)
        p_ini, p_fim : transverse distributed load at start / end (per unit length)

        Returns the 8-vector in GLOBAL coordinates.
        Local DOF ordering: [u₁, v₁, θ₁, (dv/dx)₁, u₂, v₂, θ₂, (dv/dx)₂]
        """
        L = self.length
        fe = np.zeros(8)

        # Axial (linear shape functions for u)
        fe[0] = L * (2.0 * q_ini + q_fim) / 6.0
        fe[4] = L * (q_ini + 2.0 * q_fim) / 6.0

        # Transverse – Hermite shape functions for v; θ gets zero load
        fe[1] = L * (7.0 * p_ini + 3.0 * p_fim) / 20.0        # v₁
        fe[3] = L ** 2 * (3.0 * p_ini + 2.0 * p_fim) / 60.0   # (dv/dx)₁
        fe[5] = L * (3.0 * p_ini + 7.0 * p_fim) / 20.0        # v₂
        fe[7] = -L ** 2 * (2.0 * p_ini + 3.0 * p_fim) / 60.0  # (dv/dx)₂

        return (self.R @ fe).flatten()

    def compute_equivalent_nodal_loads(self, distributed_load, n_gauss=5):
        """
        Compute consistent nodal loads for a DistributedLoad object using
        numerical Gauss-Legendre integration.
        """
        L = self.length
        c = self.c
        s = self.s

        # Build scalar load function f(x_local)
        if distributed_load.func:
            x_start = self.node_start.x
            def f(x_local):
                x_global = x_start + x_local * c
                try:
                    return float(eval(distributed_load.func,
                                      {"np": np, "x": x_global, "L": L}))
                except Exception as e:
                    print(f"Error evaluating load function '{distributed_load.func}': {e}")
                    return 0.0
        elif (distributed_load.magnitude_start is not None
              and distributed_load.magnitude_end is not None):
            a = float(distributed_load.magnitude_start)
            b_val = float(distributed_load.magnitude_end)
            def f(x_local):
                return a + (b_val - a) * (x_local / L)
        elif distributed_load.magnitude_start is not None:
            a = float(distributed_load.magnitude_start)
            def f(x_local):
                return a
        else:
            def f(x_local):
                return 0.0

        # Project global load direction onto local axes
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

        # Gauss-Legendre quadrature (on [0, L])
        xi_pts, wi = np.polynomial.legendre.leggauss(n_gauss)
        t = 0.5 * (xi_pts + 1.0)
        wt = 0.5 * wi

        fe = np.zeros(8)
        for ti, wi_s in zip(t, wt):
            x = ti * L
            N1u = 1.0 - ti
            N2u = ti
            qx = q_local(x)
            fe[0] += N1u * qx * wi_s * L
            fe[4] += N2u * qx * wi_s * L

            H1 = 1.0 - 3.0 * ti ** 2 + 2.0 * ti ** 3
            H2 = L * ti * (1.0 - ti) ** 2
            H3 = 3.0 * ti ** 2 - 2.0 * ti ** 3
            H4 = L * ti ** 2 * (ti - 1.0)
            px = p_local(x)
            fe[1] += H1 * px * wi_s * L
            fe[3] += H2 * px * wi_s * L
            fe[5] += H3 * px * wi_s * L
            fe[7] += H4 * px * wi_s * L

        return (self.R @ fe).flatten()

    def bending_moment(self, x, displacements):
        """
        Bending moment M(x) recovered from nodal forces at local position x.
        where f = K_local @ displacements is the element nodal force vector.
        This method is exact at the nodes and provides linear variation between them.

        `displacements` must be in LOCAL coordinates:
        [u₁, v₁, θ₁, (dv/dx)₁, u₂, v₂, θ₂, (dv/dx)₂]
        """
        L = self.length
        xi = x / L

        k_local = self._build_local_stiffness()
        f = k_local @ displacements

        # Extract moments at left and right ends
        # M_left = Fθ₁ − F(dv/dx)₁  (work-conjugate to curvature)
        # M_right = -(Fθ₂ − F(dv/dx)₂)
        M_left = f[2] - f[3]
        M_right = -(f[6] - f[7])

        # Linear interpolation
        return M_left * (1.0 - xi) + M_right * xi

    def shear_force(self, x, displacements):
        """
        Shear force V(x) recovered from nodal forces.

        For the Reddy-Bickford element with no distributed load, the shear force
        is constant along the element and is computed from the moment gradient:

            V = (M_left - M_right) / L = -dM/dx

        where M_left and M_right are recovered from nodal forces.

        `displacements` in LOCAL coordinates:
        [u₁, v₁, θ₁, (dv/dx)₁, u₂, v₂, θ₂, (dv/dx)₂]
        """
        L = self.length

        k_local = self._build_local_stiffness()
        f_vec = k_local @ displacements

        # Extract moments at left and right ends
        M_left = f_vec[2] - f_vec[3]
        M_right = -(f_vec[6] - f_vec[7])

        # Shear force (constant): V = -dM/dx
        return (M_left - M_right) / L

    def normal_force(self, x, displacements):
        """
        Normal (axial) force N(x) = EA·(u₂−u₁)/L (constant along element).
        `displacements` in LOCAL coordinates.
        """
        E = self.material.E
        A = self.section.area
        L = self.length
        u1 = displacements[0]
        u2 = displacements[4]
        return E * A * (u2 - u1) / L

    def interpolate_theta(self, x, displacements):
        """Interpolate θ(x) at local position x."""
        L = self.length
        xi = x / L
        # displacements local: [u1, v1, theta1, (dv/dx)1, u2, v2, theta2, (dv/dx)2]
        theta1 = displacements[2]
        theta2 = displacements[6]
        return (1.0 - xi) * theta1 + xi * theta2

    def interpolate_dv_dx(self, x, displacements):
        """Interpolate dv/dx(x) at local position x using Hermite cubic derivative."""
        L = self.length
        ti = x / L
        H1_p = (-6.0 * ti + 6.0 * ti**2) / L
        H2_p = 1.0 - 4.0 * ti + 3.0 * ti**2
        H3_p = (6.0 * ti - 6.0 * ti**2) / L
        H4_p = -2.0 * ti + 3.0 * ti**2
        
        v1 = displacements[1]
        dv_dx1 = displacements[3]
        v2 = displacements[5]
        dv_dx2 = displacements[7]
        
        return H1_p * v1 + H2_p * dv_dx1 + H3_p * v2 + H4_p * dv_dx2


def _get_mrbt_X_vectors(x, L, h, nu, E, I, mu, D1, E1, G1):
    # Using scaled constants c_i^* = c_i / (E * I) for i in {1, 5, 6}
    # This eliminates E*I from the denominators of components 0, 4, 5
    factor_scaled = h**2 * (1.0 + nu) / 420.0
    
    # X_v
    X_v = np.array([
        -factor_scaled * x,
        1.0,
        1.0 - mu * x + 0.5 * mu**2 * x**2,
        1.0 + mu * x + 0.5 * mu**2 * x**2,
        -x**3 / 6.0,
        -x**2 / 2.0
    ], dtype=float)
    
    # X_v_prime
    X_v_prime = np.array([
        -factor_scaled,
        0.0,
        -mu + mu**2 * x,
        mu + mu**2 * x,
        -x**2 / 2.0,
        -x
    ], dtype=float)
    
    # X_v_double_prime
    X_v_double_prime = np.array([
        0.0,
        0.0,
        mu**2,
        mu**2,
        -x,
        -1.0
    ], dtype=float)
    
    # X_theta
    c_theta_3 = -0.25 * mu + 0.25 * mu**2 * x - 0.125 * mu**3 * x**2
    c_theta_4 = 0.25 * mu + 0.25 * mu**2 * x + 0.125 * mu**3 * x**2
    
    X_theta = np.array([
        -factor_scaled,
        0.0,
        c_theta_3,
        c_theta_4,
        -x**2 / 2.0 - (D1 + E1) / G1,
        -x
    ], dtype=float)
    
    # X_theta_prime
    X_theta_prime = np.array([
        0.0,
        0.0,
        0.25 * mu**2 - 0.25 * mu**3 * x,
        0.25 * mu**2 + 0.25 * mu**3 * x,
        -x,
        -1.0
    ], dtype=float)
    
    return X_v, X_v_prime, X_v_double_prime, X_theta, X_theta_prime


class MRBTElement2Node(ReddyBickfordElement2Node):
    """
    2-node Modified Reddy-Bickford (MRBT) beam element.
    Uses coupled shape functions derived from the exact homogeneous solution,
    truncated using Taylor series expansion to the second order.
    """
    def _build_local_stiffness(self):
        """
        Build and return the 8×8 LOCAL stiffness matrix using MRBT kinematics
        and 5-point Gauss-Legendre quadrature.
        """
        E = self.material.E
        A = self.section.area
        I = self.section.inertia
        L = self.length
        h = self._get_height()
        nu = self.material.nu

        mu = 2.0 * np.sqrt(105.0) / (h * np.sqrt(1.0 + nu))
        D1, E1, F1, G1 = self._get_reddy_params()

        # Build H matrix mapping constants to nodal displacements (CORRECT unpacking!)
        X_v_0, X_v_prime_0, _, X_theta_0, _ = _get_mrbt_X_vectors(0.0, L, h, nu, E, I, mu, D1, E1, G1)
        X_v_L, X_v_prime_L, _, X_theta_L, _ = _get_mrbt_X_vectors(L, L, h, nu, E, I, mu, D1, E1, G1)

        H = np.zeros((6, 6))
        H[0] = X_v_0
        H[1] = X_theta_0
        H[2] = X_v_prime_0
        H[3] = X_v_L
        H[4] = X_theta_L
        H[5] = X_v_prime_L

        # Stable numerical inversion using column scaling / equilibration
        col_scales = np.max(np.abs(H), axis=0)
        col_scales[col_scales == 0] = 1.0
        H_scaled = H / col_scales
        H_scaled_inv = np.linalg.inv(H_scaled)
        H_inv = H_scaled_inv / col_scales[:, np.newaxis]

        # 5-point Gauss-Legendre quadrature
        xi_pts, wi = np.polynomial.legendre.leggauss(5)
        # map from [-1, 1] to [0, L]
        t = 0.5 * (xi_pts + 1.0)
        wt = 0.5 * wi

        D = np.zeros((4, 4))
        D[0, 0] = E * A
        D[1, 1] = D1
        D[1, 2] = E1
        D[2, 1] = E1
        D[2, 2] = F1
        D[3, 3] = G1

        k_local = np.zeros((8, 8))
        bending_dofs = [1, 2, 3, 5, 6, 7]

        for ti, wi_s in zip(t, wt):
            x = ti * L
            X_v, X_v_prime, X_v_double_prime, X_theta, X_theta_prime = _get_mrbt_X_vectors(
                x, L, h, nu, E, I, mu, D1, E1, G1
            )

            # Build strain-displacement matrix B (4x8)
            B = np.zeros((4, 8))
            B[0, 0] = -1.0 / L
            B[0, 4] = 1.0 / L

            # B[1, :] corresponds to theta'
            N_theta_prime = X_theta_prime @ H_inv
            for idx, gd in enumerate(bending_dofs):
                B[1, gd] = N_theta_prime[idx]

            # B[2, :] corresponds to v''
            N_v_double_prime = X_v_double_prime @ H_inv
            for idx, gd in enumerate(bending_dofs):
                B[2, gd] = N_v_double_prime[idx]

            # B[3, :] corresponds to theta - v'
            N_shear = (X_theta - X_v_prime) @ H_inv
            for idx, gd in enumerate(bending_dofs):
                B[3, gd] = N_shear[idx]

            # Integrate: K_local += B^T @ D @ B * weight * L
            k_local += (B.T @ D @ B) * (wi_s * L)

        k_local = 0.5 * (k_local + k_local.T)
        return k_local

    def interpolate_theta(self, x, displacements):
        """Interpolate θ(x) at local position x using MRBT coupled shape functions."""
        L = self.length
        h = self._get_height()
        nu = self.material.nu
        E = self.material.E
        I = self.section.inertia

        mu = 2.0 * np.sqrt(105.0) / (h * np.sqrt(1.0 + nu))
        D1, E1, F1, G1 = self._get_reddy_params()

        # Build H matrix and invert (CORRECT unpacking!)
        X_v_0, X_v_prime_0, _, X_theta_0, _ = _get_mrbt_X_vectors(0.0, L, h, nu, E, I, mu, D1, E1, G1)
        X_v_L, X_v_prime_L, _, X_theta_L, _ = _get_mrbt_X_vectors(L, L, h, nu, E, I, mu, D1, E1, G1)

        H = np.zeros((6, 6))
        H[0] = X_v_0
        H[1] = X_theta_0
        H[2] = X_v_prime_0
        H[3] = X_v_L
        H[4] = X_theta_L
        H[5] = X_v_prime_L

        col_scales = np.max(np.abs(H), axis=0)
        col_scales[col_scales == 0] = 1.0
        H_scaled = H / col_scales
        H_scaled_inv = np.linalg.inv(H_scaled)
        H_inv = H_scaled_inv / col_scales[:, np.newaxis]

        _, _, _, X_theta, _ = _get_mrbt_X_vectors(x, L, h, nu, E, I, mu, D1, E1, G1)
        N_theta = X_theta @ H_inv

        # Bending DOFs are [v1, theta1, (dv/dx)1, v2, theta2, (dv/dx)2]
        bending_disps = displacements[[1, 2, 3, 5, 6, 7]]
        return N_theta @ bending_disps

    def interpolate_dv_dx(self, x, displacements):
        """Interpolate dv/dx(x) at local position x using MRBT coupled shape functions."""
        L = self.length
        h = self._get_height()
        nu = self.material.nu
        E = self.material.E
        I = self.section.inertia

        mu = 2.0 * np.sqrt(105.0) / (h * np.sqrt(1.0 + nu))
        D1, E1, F1, G1 = self._get_reddy_params()

        # Build H matrix and invert (CORRECT unpacking!)
        X_v_0, X_v_prime_0, _, X_theta_0, _ = _get_mrbt_X_vectors(0.0, L, h, nu, E, I, mu, D1, E1, G1)
        X_v_L, X_v_prime_L, _, X_theta_L, _ = _get_mrbt_X_vectors(L, L, h, nu, E, I, mu, D1, E1, G1)

        H = np.zeros((6, 6))
        H[0] = X_v_0
        H[1] = X_theta_0
        H[2] = X_v_prime_0
        H[3] = X_v_L
        H[4] = X_theta_L
        H[5] = X_v_prime_L

        col_scales = np.max(np.abs(H), axis=0)
        col_scales[col_scales == 0] = 1.0
        H_scaled = H / col_scales
        H_scaled_inv = np.linalg.inv(H_scaled)
        H_inv = H_scaled_inv / col_scales[:, np.newaxis]

        _, X_v_prime, _, _, _ = _get_mrbt_X_vectors(x, L, h, nu, E, I, mu, D1, E1, G1)
        N_v_prime = X_v_prime @ H_inv

        # Bending DOFs are [v1, theta1, (dv/dx)1, v2, theta2, (dv/dx)2]
        bending_disps = displacements[[1, 2, 3, 5, 6, 7]]
        return N_v_prime @ bending_disps



class ElementResults:
    ...
    def bending_moment(self, x):
        return self.element.bending_moment(x, self.displacements)
    def shear_force(self, x):
        return self.element.shear_force(x, self.displacements)
    def normal_force(self, x):
        return self.element.normal_force(x, self.displacements)
