class Material:
    """
    Class representing a material with its properties.
    """
    def __init__(self, id, E, nu = 0.3):
        self.id = id
        self.E = E
        self.nu = nu