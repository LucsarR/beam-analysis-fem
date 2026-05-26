class Spring:
    def __init__(self, node, stiffness, direction):
        self.node = node
        self.stiffness = stiffness
        self.direction = direction

    def apply(self, K_global, dofs_per_node=3):
        idx = dofs_per_node * (self.node.id - 1) + self.direction
        K_global[idx, idx] += self.stiffness
