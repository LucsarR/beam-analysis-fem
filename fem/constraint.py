import numpy as np

class Constraint:
    """
    Represents a boundary condition (constraint) applied to a node.
    direction: 0 = x, 1 = y, 2 = rotation
    value: prescribed displacement or rotation
    """
    def __init__(self, node, direction, value=0.0):
        self.node = node
        self.direction = direction
        self.value = value

    def apply(self, K_global, F_global, penalty):
        """
        Apply the constraint to the global stiffness matrix and force vector using penalty method.
        """
        idx = 3 * (self.node.id - 1) + self.direction
        K_global[idx, idx] += penalty
        F_global[idx] += penalty * self.value

class ConstraintSet:
    """
    Manages a collection of constraints.
    """
    def __init__(self):
        self.constraints = []
        self.penalty = None

    def add(self, constraint):
        self.constraints.append(constraint)

    def apply_all(self, K_global, F_global):
        if self.penalty is None:
            self.penalty = np.max(np.abs(K_global)) * 1e4 # Set penalty based on max stiffness to ensure numerical stability
        for constraint in self.constraints:
            constraint.apply(K_global, F_global, self.penalty)

    def get_penalty(self):
        return self.penalty