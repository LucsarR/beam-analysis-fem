import numpy as np
from abc import ABC, abstractmethod

def gauss_legendre(n):
    # Points and weights on [-1, 1]
    from numpy.polynomial.legendre import leggauss
    return leggauss(n)

class Interpolation1D(ABC):
    @property
    @abstractmethod
    def order(self) -> int:
        ...

    @property
    @abstractmethod
    def xi_nodes(self) -> np.ndarray:
        ...

    @abstractmethod
    def N(self, xi: float) -> np.ndarray:
        ...

    @abstractmethod
    def dN_dxi(self, xi: float) -> np.ndarray:
        ...

class Lagrange1D(Interpolation1D):
    """
    1D Lagrange interpolation on equally spaced nodes in [-1, 1].
    Supports order=1 (2-node) and order=2 (3-node).
    """
    def __init__(self, order: int):
        assert order in (1, 2), "Lagrange1D supports order 1 or 2"
        self._order = order
        self._xi_nodes = np.linspace(-1.0, 1.0, order + 1)
        # Build square Vandermonde and its inverse once
        m = order + 1
        d = order
        V = np.vander(self._xi_nodes, N=d + 1, increasing=True)  # shape (m, d+1) with m=d+1
        self._V_inv = np.linalg.inv(V)  # (m, m)
        # Precompute derivative coefficient operator
        self._pow_idx = np.arange(d + 1)  # [0..d]

    @property
    def order(self) -> int:
        return self._order

    @property
    def xi_nodes(self) -> np.ndarray:
        return self._xi_nodes

    def _poly_vec(self, xi: float) -> np.ndarray:
        # [1, xi, xi^2, ... xi^d]
        return xi ** self._pow_idx

    def _dpoly_vec(self, xi: float) -> np.ndarray:
        # [0, 1, 2 xi, ... d xi^(d-1)]
        if self._pow_idx.size == 0:
            return np.zeros(0)
        dcoef = self._pow_idx.copy()
        dcoef[0] = 0
        return dcoef * (xi ** np.maximum(self._pow_idx - 1, 0))

    def N(self, xi: float) -> np.ndarray:
        # Evaluate all basis at xi: v(xi) @ V^{-1}
        v = self._poly_vec(xi)  # (m,)
        return v @ self._V_inv  # (m,)

    def dN_dxi(self, xi: float) -> np.ndarray:
        dv = self._dpoly_vec(xi)  # (m,)
        return dv @ self._V_inv  # (m,)

class HermiteBeam1D:
    """
    Generalized C1 Hermite interpolation for Euler-Bernoulli bending.
    - order=1 -> 2 nodes (xi = [-1, 1]), degree 3 (classic cubic Hermite).
    - order=2 -> 3 nodes (xi = [-1, 0, 1]), degree 5 (quintic Hermite).
    Produces shape functions for v with nodal DOFs [v_i, theta_i] at each node.
    v(xi) = sum_j h_disp_j(xi) * v_j + (L/2) * h_rot_j(xi) * theta_j
    """
    def __init__(self, order: int):
        assert order in (1, 2), "HermiteBeam1D supports order 1 or 2"
        self._order = order
        self._xi_nodes = np.linspace(-1.0, 1.0, order + 1)
        n = self._xi_nodes.size  # number of nodes
        d = 2 * n - 1            # polynomial degree
        m = 2 * n                # number of basis functions (v & theta per node)

        # Build constraint matrix M (m x m) for basis coefficients (degree d => d+1=m)
        # Rows 0..n-1: value constraints at nodes
        # Rows n..2n-1: derivative constraints at nodes
        M = np.zeros((m, m))
        pow_idx = np.arange(d + 1)

        # fill value constraints
        for r, xi_k in enumerate(self._xi_nodes):
            M[r, :] = xi_k ** pow_idx

        # fill derivative constraints
        for r, xi_k in enumerate(self._xi_nodes, start=n):
            # derivative vector [0,1,2 xi, ..., d xi^(d-1)]
            drow = pow_idx.copy()
            drow[0] = 0
            drow = drow * (xi_k ** np.maximum(pow_idx - 1, 0))
            M[r, :] = drow

        # Inverse once. Columns of A are coefficient vectors for each basis function.
        self._A = np.linalg.inv(M)  # (m, m)

        # Cache power indices for evaluation
        self._pow_idx = pow_idx
        self._d = d
        self._n = n
        self._m = m

    @property
    def order(self) -> int:
        return self._order

    @property
    def xi_nodes(self) -> np.ndarray:
        return self._xi_nodes

    def _poly_vec(self, xi: float) -> np.ndarray:
        return xi ** self._pow_idx  # (m,)

    def _dpoly_vec(self, xi: float) -> np.ndarray:
        dcoef = self._pow_idx.copy()
        dcoef[0] = 0
        return dcoef * (xi ** np.maximum(self._pow_idx - 1, 0))

    def _d2poly_vec(self, xi: float) -> np.ndarray:
        # second derivative coefficients: k*(k-1)*xi^(k-2)
        k = self._pow_idx
        coef = k * np.maximum(k - 1, 0)
        # For k=0,1 value is zero automatically
        return coef * (xi ** np.maximum(k - 2, 0))

    def _basis_values(self, xi: float):
        # Compute all m Hermite basis functions in canonical order:
        # [h_disp_0, ..., h_disp_{n-1}, h_rot_0, ..., h_rot_{n-1}]
        v = self._poly_vec(xi)        # (m,)
        dv = self._dpoly_vec(xi)      # (m,)
        d2v = self._d2poly_vec(xi)    # (m,)

        # Evaluate h = v @ A, h' = dv @ A, h'' = d2v @ A
        H = v @ self._A
        dH = dv @ self._A
        d2H = d2v @ self._A
        # Split into displacement and rotation families
        h_disp = H[: self._n]
        h_rot = H[self._n :]
        dh_disp = dH[: self._n]
        dh_rot = dH[self._n :]
        d2h_disp = d2H[: self._n]
        d2h_rot = d2H[self._n :]
        return h_disp, h_rot, dh_disp, dh_rot, d2h_disp, d2h_rot

    def Nv(self, xi: float, L: float) -> np.ndarray:
        """
        Shape functions for v mapping to [v1, theta1, v2, theta2, ...].
        Returns size (2*n,) vector evaluated at xi.
        """
        h_disp, h_rot, _, _, _, _ = self._basis_values(xi)
        Nv = np.zeros(2 * self._n)
        for j in range(self._n):
            Nv[2 * j] = h_disp[j]
            Nv[2 * j + 1] = (L / 2.0) * h_rot[j]
        return Nv

    def d2Nv_dx2(self, xi: float, L: float) -> np.ndarray:
        """
        Second derivative of v-shape functions w.r.t physical x.
        Returns size (2*n,) vector for curvature kappa = d2v/dx2 mapping.
        """
        _, _, _, _, d2h_disp, d2h_rot = self._basis_values(xi)
        # d2/dx2 = (2/L)^2 d2/dxi2; account for (L/2) factor in rotation shapes.
        disp_part = (4.0 / L**2) * d2h_disp
        rot_part = (2.0 / L) * d2h_rot  # (4/L^2)*(L/2) = 2/L
        kappa_vec = np.zeros(2 * self._n)
        for j in range(self._n):
            kappa_vec[2 * j] = disp_part[j]
            kappa_vec[2 * j + 1] = rot_part[j]
        return kappa_vec