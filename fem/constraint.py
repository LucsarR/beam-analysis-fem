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
        self.penalty = None  # Will be set when constraint is applied

    def apply(self, K_global, F_global, penalty):
        """
        Apply the constraint to the global stiffness matrix and force vector using penalty method.
        """
        self.penalty = penalty  # Store penalty value for reaction calculation
        idx = 3 * (self.node.id - 1) + self.direction
        K_global[idx, idx] += penalty
        F_global[idx] += penalty * self.value
    
    def calculate_reaction(self, displacement_vector):
        """
        Calculate reaction force at this constraint using penalty method.
        
        The reaction force is the force required to maintain the constraint:
        R = penalty * (u_actual - u_prescribed)
        
        Args:
            displacement_vector: Global displacement vector
            
        Returns:
            Reaction force value (positive means force in positive direction)
        """
        if self.penalty is None:
            raise ValueError("Constraint must be applied before calculating reactions")
        
        idx = 3 * (self.node.id - 1) + self.direction
        actual_displacement = displacement_vector[idx]
        reaction = self.penalty * (actual_displacement - self.value)
        return reaction

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
    
    def calculate_all_reactions(self, displacement_vector):
        """
        Calculate reaction forces at all constraints using penalty method.
        
        Returns:
            Dictionary mapping (node_id, direction) to reaction force value
            direction: 0=x, 1=y, 2=rotation
        """
        reactions = {}
        for constraint in self.constraints:
            key = (constraint.node.id, constraint.direction)
            reactions[key] = constraint.calculate_reaction(displacement_vector)
        return reactions