class Load:
    def __init__(self, node, value, direction):
        """
        value: float or callable (e.g., lambda x: ...)
        direction: 'x', 'y', or 'moment'
        """
        self.node = node
        self.value = value
        self.direction = direction

    def get_value(self, x=None):
        if callable(self.value):
            return self.value(x)
        return self.value

class DistributedLoad:
    def __init__(self, element, q_func, direction):
        """
        q_func: callable (e.g., lambda x: ...) or float
        direction: 'x', 'y', or 'moment'
        """
        self.element = element
        self.q_func = q_func
        self.direction = direction

    def get_value(self, x):
        if callable(self.q_func):
            return self.q_func(x)
        return self.q_func