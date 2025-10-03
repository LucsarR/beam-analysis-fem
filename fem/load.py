class Load:
    """
    Abstract base class for loads applied to nodes or elements.
    """
    def __init__(self, magnitude):
        self.magnitude = magnitude

    def apply(self, F_global, node):
        raise NotImplementedError("Implement in subclass.")

class PointLoad(Load):
    """
    Represents a point load applied to a node.
    direction: 0 = x, 1 = y, 2 = moment
    """
    def __init__(self, magnitude, direction):
        super().__init__(magnitude)
        self.direction = direction

    def apply(self, F_global, node):
        idx = 3 * (node.id - 1) + self.direction
        F_global[idx] += self.magnitude

class DistributedLoad(Load):
    """
    Represents a distributed load applied to an element.
    direction: 'x', 'y', 'l', 't'
    """
    def __init__(self, magnitude_start, magnitude_end, direction):
        super().__init__(None)
        self.magnitude_start = magnitude_start
        self.magnitude_end = magnitude_end
        self.direction = direction

    def apply(self, element):
        # Implementation depends on element type
        # Check the force_vector method in Element classes
        pass

class MomentLoad(Load):
    """
    Represents a moment load applied to a node.
    """
    def __init__(self, magnitude):
        super().__init__(magnitude)

    def apply(self, F_global, node):
        idx = 3 * (node.id - 1) + 2
        F_global[idx] += self.magnitude