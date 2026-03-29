class Material:
    """
    Class representing a material with its properties.

    Exactly two of E, G, and nu must be specified; the third is computed from
    the standard isotropic elasticity relation:

        G = E / (2 * (1 + nu))

    Parameters
    ----------
    id : int
        Unique identifier for the material.
    E : float, optional
        Young's modulus (elastic modulus).
    nu : float, optional
        Poisson's ratio.  Defaults to 0.3 when only E is supplied (backward
        compatibility).
    G : float, optional
        Shear modulus.

    Raises
    ------
    ValueError
        If all three parameters are provided simultaneously (only two are
        allowed — the third is always derived).
    ValueError
        If fewer than two parameters are provided and no default can be
        applied.
    """

    def __init__(self, id, E=None, nu=None, G=None):
        self.id = id

        n_provided = sum(v is not None for v in [E, G, nu])

        if n_provided == 3:
            raise ValueError(
                "Only two of E, G, and nu may be specified at the same time; "
                "the third is computed automatically from "
                "G = E / (2 * (1 + nu))."
            )

        if E is not None and G is not None:
            # Compute nu from E and G
            self.E = float(E)
            self.G = float(G)
            self.nu = self.E / (2.0 * self.G) - 1.0
        elif G is not None and nu is not None:
            # Compute E from G and nu
            self.G = float(G)
            self.nu = float(nu)
            self.E = 2.0 * self.G * (1.0 + self.nu)
        else:
            # Default path: E and nu → compute G
            if E is None:
                raise ValueError(
                    "At least two of E, G, and nu must be provided."
                )
            if nu is None:
                nu = 0.3  # backward-compatible default
            self.E = float(E)
            self.nu = float(nu)
            self.G = self.E / (2.0 * (1.0 + self.nu))