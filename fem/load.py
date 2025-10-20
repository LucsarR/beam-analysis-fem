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
    Supports constant, linear, or custom function (func) for the load distribution.
    """
    def __init__(self, magnitude_start=None, magnitude_end=None, direction='l', func=None):
        super().__init__(None)
        self.magnitude_start = magnitude_start
        self.magnitude_end = magnitude_end
        self.direction = direction
        self.func = func  # string, e.g. "1000*np.sin(np.pi*x/L)"

    def apply(self, element):
        """
        Calls the element's method to compute equivalent nodal loads.
        """
        return element.compute_equivalent_nodal_loads(self)