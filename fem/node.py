class Node:
    """
    Class representing a node in the finite element mesh.
    """
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.loads = []
        self.springs = []