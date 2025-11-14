class Material:
    """
    Class representing a material with its properties.
    """
    def __init__(self, id, E, nu = 0.3):
        self.id = id
        self.E = E
        self.nu = nu
    
    @property
    def G(self):
        """
        Shear modulus computed from Young's modulus and Poisson's ratio.
        G = E / (2 * (1 + nu))
        """
        return self.E / (2 * (1 + self.nu))