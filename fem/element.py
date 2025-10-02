import numpy as np
from abc import ABC, abstractmethod

class Element(ABC):
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

class EulerBernoulliElement(Element):
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
        A = self.section.area
        I = self.section.inertia
        L = self.length
        c = self.c
        s = self.s
        mu = (A * L**2) / (2 * I)
        aux_1 = mu * c**2 + 6 * s**2
        aux_2 = mu * s**2 + 6 * c**2
        aux_3 = (mu - 6) * c * s
        aux_4 = 3 * L * c
        aux_5 = 3 * L * s
        aux_6 = L**2
        k_local = 2 * E * I / L**3 * np.array([
            [aux_1, aux_3, -aux_5, -aux_1, -aux_3, -aux_5],
            [aux_3, aux_2, aux_4, -aux_3, -aux_2, aux_4],
            [-aux_5, aux_4, 2*aux_6, aux_5, -aux_4, aux_6],
            [-aux_1, -aux_3, aux_5, aux_1, aux_3, aux_5],
            [-aux_3, -aux_2, -aux_4, aux_3, aux_2, -aux_4],
            [-aux_5, aux_4, aux_6, aux_5, -aux_4, 2*aux_6]
        ])
        return k_local

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
        fe_local = L * R @ np.array([
            [(2*q_ini + q_fim) / 6],
            [(7*p_ini + 3*p_fim) / 20],
            [(3*p_ini + 2*p_fim) * L / 60],
            [(q_ini + 2*q_fim) / 6],
            [(3*p_ini + 7*p_fim) / 20],
            [-(2*p_ini + 3*p_fim) * L / 60]
        ])
        return fe_local.flatten()