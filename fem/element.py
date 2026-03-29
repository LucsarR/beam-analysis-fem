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
        - Bending: quadratic shape functions for rotation θ
        - Displacement v is derived from θ using the Euler-Bernoulli relation: θ = dw/dx
        - Central node now has rotation DOF
        
        For Euler-Bernoulli beams: M = EI * dθ/dx
        Strain energy: U = ∫(EI/2)*(dθ/dx)² dx
        
        We use quadratic shape functions for θ:
        - θ(ξ) = N1(ξ)*θ1 + N2(ξ)*θ2 + N3(ξ)*θ3
        Where N1 = (1-ξ)(1-2ξ), N2 = 4ξ(1-ξ), N3 = ξ(2ξ-1)
        
        The displacement is obtained by integrating: w = ∫θ dx + constant
        
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
        
        # Bending stiffness for Euler-Bernoulli beam
        # Using quadratic shape functions for rotation θ
        # DOFs for bending: [v1, θ1, v2, θ2, v3, θ3] = indices [1, 2, 4, 5, 7, 8]
        bending_dofs = [1, 2, 4, 5, 7, 8]
        n_bending = len(bending_dofs)
        k_bending = np.zeros((n_bending, n_bending))
        
        # Numerical integration using 3-point Gauss quadrature
        xi_gauss = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
        w_gauss = np.array([5/9, 8/9, 5/9])
        
        for xi_g, w_g in zip(xi_gauss, w_gauss):
            # Map from [-1,1] to [0,1]
            xi = (xi_g + 1) / 2
            
            # Quadratic shape function derivatives for θ
            # dN/dξ
            dN1_dxi = -3 + 4*xi
            dN2_dxi = 4 - 8*xi
            dN3_dxi = -1 + 4*xi
            
            # dθ/dx = (1/L) * dθ/dξ = (1/L) * (dN1*θ1 + dN2*θ2 + dN3*θ3)
            # Bending strain energy density: (EI/2) * (dθ/dx)²
            
            # Shape function vector for dθ/dx (on rotation DOFs only)
            # We only put derivatives on θ DOFs, not v DOFs
            dtheta_dx_vec = np.zeros(n_bending)
            dtheta_dx_vec[1] = dN1_dxi / L  # θ1 position
            dtheta_dx_vec[3] = dN2_dxi / L  # θ2 position
            dtheta_dx_vec[5] = dN3_dxi / L  # θ3 position
            
            # Add contribution to stiffness: ∫ EI * (dθ/dx)² dx
            # Jacobian for transformation: dx = (L/2) dξ_gauss
            k_bending += E * I * np.outer(dtheta_dx_vec, dtheta_dx_vec) * (L/2) * w_g
            
            # Now enforce compatibility: dw/dx = θ
            # Using penalty method with carefully chosen penalty parameter
            # The penalty value of 10000*EI/L was chosen empirically to:
            # 1. Enforce the Euler-Bernoulli constraint θ = dw/dx reasonably well
            # 2. Avoid numerical ill-conditioning from too large a penalty
            # 3. Balance constraint enforcement with solution accuracy
            # For better accuracy, use finer meshes rather than increasing penalty
            penalty = 10000 * E * I / L  # Empirically chosen for constraint/conditioning balance
            
            # Quadratic shape functions for v
            N1_v = (1 - xi) * (1 - 2*xi)
            N2_v = 4 * xi * (1 - xi)
            N3_v = xi * (2*xi - 1)
            
            # dw/dx = (1/L) * dw/dξ
            dw_dx_vec = np.zeros(n_bending)
            dw_dx_vec[0] = dN1_dxi / L  # v1 position
            dw_dx_vec[2] = dN2_dxi / L  # v2 position
            dw_dx_vec[4] = dN3_dxi / L  # v3 position
            
            # θ shape functions
            theta_vec = np.zeros(n_bending)
            theta_vec[1] = N1_v  # θ1 position
            theta_vec[3] = N2_v  # θ2 position
            theta_vec[5] = N3_v  # θ3 position
            
            # Constraint: (dw/dx - θ) = 0
            constraint_vec = dw_dx_vec - theta_vec
            k_bending += penalty * np.outer(constraint_vec, constraint_vec) * (L/2) * w_g
        
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
        
        # Consistent load vector for linearly varying distributed loads
        # For quadratic axial shape functions
        q_avg = (q_ini + q_fim) / 2
        fe_axial = L * np.array([
            q_ini / 6,
            2 * q_avg / 3,
            q_fim / 6
        ])
        
        # For bending with quadratic shape functions for displacement
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
        For Euler-Bernoulli: M(x) = E*I * dθ/dx
        Displacements should be in local coordinates: [u1, v1, theta1, u2, v2, theta2, u3, v3, theta3]
        """
        E = self.material.E
        I = self.section.inertia
        L = self.length
        
        # Local DOFs: [u1, v1, theta1, u2, v2, theta2, u3, v3, theta3]
        theta1 = displacements[2]
        theta2 = displacements[5]
        theta3 = displacements[8]
        
        xi = x / L
        
        # Derivatives of quadratic shape functions for rotation
        dN1_theta = (-3 + 4*xi) / L
        dN2_theta = (4 - 8*xi) / L
        dN3_theta = (-1 + 4*xi) / L
        
        # dθ/dx
        dtheta_dx = dN1_theta * theta1 + dN2_theta * theta2 + dN3_theta * theta3
        
        return E * I * dtheta_dx

    def shear_force(self, x, displacements):
        """
        Returns shear force V(x) at position x (in local coordinates, 0 <= x <= L).
        For Euler-Bernoulli: V(x) = E*I * d²θ/dx² = E*I * d³w/dx³
        Displacements should be in local coordinates: [u1, v1, theta1, u2, v2, theta2, u3, v3, theta3]
        """
        E = self.material.E
        I = self.section.inertia
        L = self.length
        
        # Local DOFs: [u1, v1, theta1, u2, v2, theta2, u3, v3, theta3]
        theta1 = displacements[2]
        theta2 = displacements[5]
        theta3 = displacements[8]
        
        xi = x / L
        
        # Second derivatives of quadratic shape functions for rotation
        # d²θ/dx² = (1/L²) * d²θ/dξ²
        # For quadratic: dN/dξ = a + bξ, so d²N/dξ² = b (constant)
        d2N1_theta_dxi2 = 4
        d2N2_theta_dxi2 = -8
        d2N3_theta_dxi2 = 4
        
        d2theta_dx2 = (1/L**2) * (d2N1_theta_dxi2 * theta1 + d2N2_theta_dxi2 * theta2 + d2N3_theta_dxi2 * theta3)
        
        # Shear force: V = -EI * d²θ/dx²
        # Sign convention: negative because V = -dM/dx for Euler-Bernoulli beams
        # Positive shear causes counterclockwise rotation of the element
        return -E * I * d2theta_dx2

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

class ReddyBickfordElement2Node(Element):
    """
    2-node Reddy-Bickford (modified Reddy) beam element.

    Based on Reddy's third-order shear deformation theory (TSDT).
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
        # 8×8 transformation matrix: [u,v] rotate as a vector; θ and dv/dx are
        # scalar quantities that do not change under coordinate transformation.
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
        # Fallback: equivalent height for rectangular section (I = A·h²/12)
        return float(np.sqrt(12.0 * sec.inertia / sec.area))

    def _get_reddy_params(self):
        """
        Compute the four modified stiffness parameters for the Reddy TSDT.

        Assumes a rectangular cross-section of width b = A/h and height h.
        Returns (D1, E1, F1, G1) where:
          D1 = E·(I − 2c₁·I₄ + c₁²·I₆)   [modified bending, 68EI/105 for rect.]
          E1 = E·c₁·(I₄ − c₁·I₆)           [coupling term,   16EI/105 for rect.]
          F1 = E·c₁²·I₆                      [higher-order,     EI/21   for rect.]
          G1 = G·Ā  where Ā = A−2c₃·I+c₃²·I₄ [eff. shear,   8GA/15  for rect.]
        """
        E = self.material.E
        G = self.material.G
        A = self.section.area
        I = self.section.inertia
        h = self._get_height()
        b = A / h                            # effective width

        c1 = 4.0 / (3.0 * h ** 2)           # Reddy c₁ parameter
        c3 = 3.0 * c1                        # = 4/h²

        I4 = b * h ** 5 / 80.0              # ∫z⁴ dA (rectangular)
        I6 = b * h ** 7 / 448.0             # ∫z⁶ dA (rectangular)

        A_bar = A - 2.0 * c3 * I + c3 ** 2 * I4  # effective shear area (= 8A/15 rect.)

        D1 = E * (I - 2.0 * c1 * I4 + c1 ** 2 * I6)
        E1 = E * c1 * (I4 - c1 * I6)
        F1 = E * c1 ** 2 * I6
        G1 = G * A_bar

        return D1, E1, F1, G1

    def stiffness_matrix(self):
        """
        Build the 8×8 element stiffness matrix in global coordinates.

        Local DOF ordering: [u₁, v₁, θ₁, (dv/dx)₁, u₂, v₂, θ₂, (dv/dx)₂]
        (index 2 = θ, index 3 = dv/dx  – matches the global 4-DOF-per-node
        convention used throughout the assembly, so direction 2 = rotation
        remains consistent with the Euler-Bernoulli / Timoshenko elements.)
        """
        E = self.material.E
        A = self.section.area
        L = self.length
        R = self.R

        D1, E1, F1, G1 = self._get_reddy_params()

        # ------------------------------------------------------------------
        # 8×8 local stiffness matrix K_local.
        # The 6×6 bending sub-matrix K_b is derived analytically from the
        # Reddy TSDT strain energy:
        #   U_bend = ½∫[D1·(θ')² − 2E1·θ'·v'' + F1·(v'')² + G1·(θ+v')²] dx
        #
        # Hermite cubic shape functions for v: H1…H4
        # Linear shape functions for θ:       N1 = 1−ξ, N2 = ξ  (ξ = x/L)
        #
        # K_b DOF ordering (bending-only, 6×6):
        #   b0 = v₁,  b1 = v₁' = (dv/dx)₁,  b2 = θ₁,
        #   b3 = v₂,  b4 = v₂' = (dv/dx)₂,  b5 = θ₂
        #
        # In the 8×8 matrix these bending indices map to:
        #   b0=v₁  → 1,  b1=(dv/dx)₁ → 3,  b2=θ₁  → 2,
        #   b3=v₂  → 5,  b4=(dv/dx)₂ → 7,  b5=θ₂  → 6
        # ------------------------------------------------------------------
        b_dofs = [1, 3, 2, 5, 7, 6]   # map bending index → 8×8 index

        K_b = np.zeros((6, 6))

        # --- D1 term: ψ'·ψ'^T where ψ' = dθ/dx = (θ₂−θ₁)/L (constant) ---
        K_b[2, 2] += D1 / L
        K_b[2, 5] += -D1 / L
        K_b[5, 2] += -D1 / L
        K_b[5, 5] += D1 / L

        # --- F1 term: v''·v''^T  (standard Euler-Bernoulli stiffness form) ---
        f = F1 / L ** 3
        K_b[0, 0] += 12.0 * f;  K_b[0, 1] += 6.0 * L * f
        K_b[0, 3] += -12.0 * f; K_b[0, 4] += 6.0 * L * f
        K_b[1, 0] += 6.0 * L * f;  K_b[1, 1] += 4.0 * L ** 2 * f
        K_b[1, 3] += -6.0 * L * f; K_b[1, 4] += 2.0 * L ** 2 * f
        K_b[3, 0] += -12.0 * f; K_b[3, 1] += -6.0 * L * f
        K_b[3, 3] += 12.0 * f;  K_b[3, 4] += -6.0 * L * f
        K_b[4, 0] += 6.0 * L * f;  K_b[4, 1] += 2.0 * L ** 2 * f
        K_b[4, 3] += -6.0 * L * f; K_b[4, 4] += 4.0 * L ** 2 * f

        # --- E1 cross term: −E1·(ψ'·κ^T + κ·ψ'^T) ---
        # From −E1·∫dθ/dx·d²v/dx² dx, analytically integrated:
        #   K[v₁', θ₁] = K[θ₁, v₁'] = −E1/L
        #   K[v₁', θ₂] = K[θ₂, v₁'] = +E1/L
        #   K[θ₁, v₂'] = K[v₂', θ₁] = +E1/L
        #   K[v₂', θ₂] = K[θ₂, v₂'] = −E1/L
        # b-index mapping: v₁'=1, θ₁=2, v₂'=4, θ₂=5
        K_b[1, 2] += -E1 / L;   K_b[2, 1] += -E1 / L
        K_b[1, 5] += E1 / L;    K_b[5, 1] += E1 / L
        K_b[2, 4] += E1 / L;    K_b[4, 2] += E1 / L
        K_b[4, 5] += -E1 / L;   K_b[5, 4] += -E1 / L

        # --- G1 shear term: G1·L·∫γᵢ·γⱼ dξ ---
        # γ = [N_θ,ᵢ + dN_v,ᵢ/dx] – see analytical integral table in derivation.
        K_b[0, 0] += 6.0 * G1 / (5.0 * L)
        K_b[0, 1] += G1 / 10.0
        K_b[0, 2] += -G1 / 2.0
        K_b[0, 3] += -6.0 * G1 / (5.0 * L)
        K_b[0, 4] += G1 / 10.0
        K_b[0, 5] += -G1 / 2.0

        K_b[1, 0] += G1 / 10.0
        K_b[1, 1] += 2.0 * G1 * L / 15.0
        K_b[1, 2] += G1 * L / 12.0
        K_b[1, 3] += -G1 / 10.0
        K_b[1, 4] += -G1 * L / 30.0
        K_b[1, 5] += -G1 * L / 12.0

        K_b[2, 0] += -G1 / 2.0
        K_b[2, 1] += G1 * L / 12.0
        K_b[2, 2] += G1 * L / 3.0
        K_b[2, 3] += G1 / 2.0
        K_b[2, 4] += -G1 * L / 12.0
        K_b[2, 5] += G1 * L / 6.0

        K_b[3, 0] += -6.0 * G1 / (5.0 * L)
        K_b[3, 1] += -G1 / 10.0
        K_b[3, 2] += G1 / 2.0
        K_b[3, 3] += 6.0 * G1 / (5.0 * L)
        K_b[3, 4] += -G1 / 10.0
        K_b[3, 5] += G1 / 2.0

        K_b[4, 0] += G1 / 10.0
        K_b[4, 1] += -G1 * L / 30.0
        K_b[4, 2] += -G1 * L / 12.0
        K_b[4, 3] += -G1 / 10.0
        K_b[4, 4] += 2.0 * G1 * L / 15.0
        K_b[4, 5] += G1 * L / 12.0

        K_b[5, 0] += -G1 / 2.0
        K_b[5, 1] += -G1 * L / 12.0
        K_b[5, 2] += G1 * L / 6.0
        K_b[5, 3] += G1 / 2.0
        K_b[5, 4] += G1 * L / 12.0
        K_b[5, 5] += G1 * L / 3.0

        # Build the full 8×8 local stiffness matrix
        k_local = np.zeros((8, 8))

        # Axial part (DOFs 0 and 4)
        k_local[0, 0] = E * A / L
        k_local[0, 4] = -E * A / L
        k_local[4, 0] = -E * A / L
        k_local[4, 4] = E * A / L

        # Map bending sub-matrix into full 8×8
        for i_b, i_g in enumerate(b_dofs):
            for j_b, j_g in enumerate(b_dofs):
                k_local[i_g, j_g] = K_b[i_b, j_b]

        # Transform to global coordinates: K_global = R · K_local · R^T
        return R @ k_local @ R.T

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
        # v₁  (H1):  L·(7p_ini + 3p_fim)/20
        # v₁' (H2):  L²·(3p_ini + 2p_fim)/60
        # v₂  (H3):  L·(3p_ini + 7p_fim)/20
        # v₂' (H4): −L²·(2p_ini + 3p_fim)/60
        fe[1] = L * (7.0 * p_ini + 3.0 * p_fim) / 20.0    # v₁
        fe[3] = L ** 2 * (3.0 * p_ini + 2.0 * p_fim) / 60.0  # (dv/dx)₁
        fe[5] = L * (3.0 * p_ini + 7.0 * p_fim) / 20.0    # v₂
        fe[7] = -L ** 2 * (2.0 * p_ini + 3.0 * p_fim) / 60.0  # (dv/dx)₂
        # fe[2] = fe[6] = 0  (θ₁, θ₂ – no direct transverse load contribution)

        return (self.R @ fe).flatten()

    def compute_equivalent_nodal_loads(self, distributed_load, n_gauss=5):
        """
        Compute consistent nodal loads for a DistributedLoad object using
        numerical Gauss-Legendre integration.

        Returns an 8-vector in GLOBAL coordinates
        [u₁, v₁, θ₁, (dv/dx)₁, u₂, v₂, θ₂, (dv/dx)₂].
        """
        L = self.length
        c = self.c
        s = self.s

        # Build scalar load function f(x)
        if distributed_load.func:
            def f(x):
                try:
                    return float(eval(distributed_load.func,
                                      {"np": np, "x": x, "L": L}))
                except Exception as e:
                    print(f"Error evaluating load function '{distributed_load.func}': {e}")
                    return 0.0
        elif (distributed_load.magnitude_start is not None
              and distributed_load.magnitude_end is not None):
            a = float(distributed_load.magnitude_start)
            b_val = float(distributed_load.magnitude_end)
            def f(x):
                return a + (b_val - a) * (x / L)
        elif distributed_load.magnitude_start is not None:
            a = float(distributed_load.magnitude_start)
            def f(x):
                return a
        else:
            def f(x):
                return 0.0

        # Project global load direction onto local axes
        if distributed_load.direction == 'x':
            def q_local(x): return f(x) * c      # axial
            def p_local(x): return -f(x) * s     # transverse
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
        t = 0.5 * (xi_pts + 1.0)           # map [−1,1] → [0,1]
        wt = 0.5 * wi

        fe = np.zeros(8)

        for ti, wi_s in zip(t, wt):
            x = ti * L

            # --- Axial: linear shape functions ---
            N1u = 1.0 - ti
            N2u = ti
            qx = q_local(x)
            fe[0] += N1u * qx * wi_s * L    # u₁
            fe[4] += N2u * qx * wi_s * L    # u₂

            # --- Transverse: Hermite shape functions for v; θ gets zero ---
            H1 = 1.0 - 3.0 * ti ** 2 + 2.0 * ti ** 3
            H2 = L * ti * (1.0 - ti) ** 2                # shape fn for v₁' = dv/dx₁
            H3 = 3.0 * ti ** 2 - 2.0 * ti ** 3
            H4 = L * ti ** 2 * (ti - 1.0)                # shape fn for v₂' = dv/dx₂
            px = p_local(x)
            fe[1] += H1 * px * wi_s * L   # v₁
            fe[3] += H2 * px * wi_s * L   # (dv/dx)₁
            fe[5] += H3 * px * wi_s * L   # v₂
            fe[7] += H4 * px * wi_s * L   # (dv/dx)₂
            # fe[2] = fe[6] = 0  (θ₁, θ₂ – no load contribution)

        # Transform to global
        return (self.R @ fe).flatten()

    def bending_moment(self, x, displacements):
        """
        Effective bending moment M̂(x) = D1·θ'(x) − E1·v''(x) at local position x.

        `displacements` must be in LOCAL coordinates:
        [u₁, v₁, θ₁, (dv/dx)₁, u₂, v₂, θ₂, (dv/dx)₂]
        """
        D1, E1, _F1, _G1 = self._get_reddy_params()
        L = self.length
        xi = x / L

        # θ is linear: θ'(x) = (θ₂ − θ₁) / L
        theta1 = displacements[2]
        theta2 = displacements[6]
        dtheta_dx = (theta2 - theta1) / L

        # v is cubic Hermite; v'' = d²v/dx² evaluated at ξ
        v1 = displacements[1]
        dvdx1 = displacements[3]   # dv/dx at node 1
        v2 = displacements[5]
        dvdx2 = displacements[7]   # dv/dx at node 2

        d2H1 = 6.0 * (2.0 * xi - 1.0) / L ** 2
        d2H2 = 2.0 * (3.0 * xi - 2.0) / L
        d2H3 = 6.0 * (1.0 - 2.0 * xi) / L ** 2
        d2H4 = 2.0 * (3.0 * xi - 1.0) / L

        v_ddx = d2H1 * v1 + d2H2 * dvdx1 + d2H3 * v2 + d2H4 * dvdx2

        return D1 * dtheta_dx - E1 * v_ddx

    def shear_force(self, x, displacements):
        """
        Effective shear force V̂ = −dM̂/dx = E1·v'''(x) at local position x.

        For a cubic Hermite v, v''' is constant along the element.
        `displacements` in LOCAL coordinates.
        """
        _D1, E1, _F1, _G1 = self._get_reddy_params()
        L = self.length

        v1 = displacements[1]
        dvdx1 = displacements[3]
        v2 = displacements[5]
        dvdx2 = displacements[7]

        # v''' = (1/L³)·[12·v₁ + 6L·v₁' − 12·v₂ + 6L·v₂']
        v_dddx = (12.0 * v1 + 6.0 * L * dvdx1 - 12.0 * v2 + 6.0 * L * dvdx2) / L ** 3

        return E1 * v_dddx

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


class ElementResults:
    ...
    def bending_moment(self, x):
        return self.element.bending_moment(x, self.displacements)
    def shear_force(self, x):
        return self.element.shear_force(x, self.displacements)
    def normal_force(self, x):
        return self.element.normal_force(x, self.displacements)