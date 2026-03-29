# Default material properties (unit-system-agnostic; see Help tab for unit reference)
DEFAULT_E = 200e3  # Young's modulus — default corresponds to steel (e.g. 200 000 MPa or 200 GPa)
DEFAULT_NU = 0.3    # Poisson's ratio
DEFAULT_G = DEFAULT_E / (2 * (1 + DEFAULT_NU))  # Shear modulus (computed from E and ν)

# Default section dimensions (unused placeholders; actual defaults are set per section type in app.py)
DEFAULT_WIDTH = 10
DEFAULT_HEIGHT = 10

# Supported section types
SECTION_TYPES = [
    "rectangular_bar", "rectangular_tube", "circular_bar", "circular_tube",
    "trapezoidal_bar", "trapezoidal_tube", "hexagonal_bar", "hexagonal_tube",
    "ibeam", "c_section", "l_section", "t_section", "z_section", "hat_section", "general"
]

# Supported element types
ELEMENT_TYPES = {
    "Euler-Bernoulli 2-node": "euler_bernoulli_2node",
    "Euler-Bernoulli 3-node": "euler_bernoulli_3node",
    "Timoshenko 2-node": "timoshenko_2node",
    "Timoshenko 3-node": "timoshenko_3node",
    "Reddy-Bickford 2-node": "reddy_bickford_2node"
}

# Numerical tolerance
TOL = 1e-8

# Output directory
RESULTS_DIR = "results/"