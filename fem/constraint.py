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

    def apply(self, K_global, F_global, penalty, dofs_per_node=3):
        """
        Apply the constraint to the global stiffness matrix and force vector using penalty method.
        """
        self.penalty = penalty  # Store penalty value for reaction calculation
        idx = dofs_per_node * (self.node.id - 1) + self.direction
        K_global[idx, idx] += penalty
        F_global[idx] += penalty * self.value
    
    def calculate_reaction(self, displacement_vector, dofs_per_node=3):
        """
        Calculate reaction force at this constraint using penalty method.
        
        The reaction force is the force that the support applies to the structure:
        R = penalty * (u_prescribed - u_actual)
        
        This represents the restoring force from the penalty spring. When the actual
        displacement deviates from the prescribed value, the spring applies a force
        to restore it, which is the reaction force.
        
        Args:
            displacement_vector: Global displacement vector
            dofs_per_node: Number of DOFs per node (3 for EB/Timoshenko, 4 for Reddy-Bickford)
            
        Returns:
            Reaction force value (positive means force in positive direction)
        """
        if self.penalty is None:
            raise ValueError("Constraint must be applied before calculating reactions")
        
        idx = dofs_per_node * (self.node.id - 1) + self.direction
        actual_displacement = displacement_vector[idx]
        
        # Reaction is the force that maintains the constraint
        # Spring force: F = k * delta = penalty * (prescribed - actual)
        reaction = self.penalty * (self.value - actual_displacement)
        
        return reaction

class ConstraintSet:
    """
    Manages a collection of constraints.
    """
    def __init__(self):
        self.constraints = []
        self.penalty = None
        self._dofs_per_node = 3

    def add(self, constraint):
        self.constraints.append(constraint)

    def apply_all(self, K_global, F_global, dofs_per_node=3):
        if self.penalty is None:
            self.penalty = np.max(np.abs(K_global)) * 1e4 # Set penalty based on max stiffness to ensure numerical stability
        self._dofs_per_node = dofs_per_node
        for constraint in self.constraints:
            constraint.apply(K_global, F_global, self.penalty, dofs_per_node)

    def get_penalty(self):
        return self.penalty
    
    def calculate_all_reactions(self, displacement_vector, dofs_per_node=None):
        """
        Calculate reaction forces at all constraints using penalty method.
        
        Args:
            displacement_vector: Global displacement vector
            dofs_per_node: Number of DOFs per node (3 for EB/Timoshenko, 4 for Reddy-Bickford)
        
        Returns:
            Dictionary mapping (node_id, direction) to reaction force value
            direction: 0=u, 1=v, 2=rotation (or dv/dx for Reddy dir 3)
        """
        reactions = {}
        for constraint in self.constraints:
            key = (constraint.node.id, constraint.direction)
            reactions[key] = constraint.calculate_reaction(displacement_vector, dofs_per_node)
        return reactions