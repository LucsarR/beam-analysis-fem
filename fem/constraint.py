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

    def apply(self, K_global, F_global):
        """
        Apply the constraint to the global stiffness matrix and force vector.
        """
        idx = 3 * (self.node.id - 1) + self.direction
        K_global[idx, :] = 0
        K_global[:, idx] = 0
        K_global[idx, idx] = 1
        F_global[idx] = self.value

class ConstraintSet:
    """
    Manages a collection of constraints.
    """
    def __init__(self):
        self.constraints = []

    def add(self, constraint):
        self.constraints.append(constraint)

    def apply_all(self, K_global, F_global):
        for constraint in self.constraints:
            constraint.apply(K_global, F_global)