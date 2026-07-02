import numpy as np

# Parameters
L = 1.0
E = 10000000.0
NU = 0.3
b = 0.5
h = 0.1
P = -1030.0
M = 1.03

A = b * h
I = b * h**3 / 12
G = E / (2 * (1 + NU))
alpha = 4 / (3 * h**2)
mu = (2 * np.sqrt(105)) / (h * np.sqrt(1 + NU))

# Analytical solution at x = L
# dv_dx = - P * (2L - x)*x / (2*E*I) + 12*P*(1+nu)/(5*E*A) - 12*P*(1+nu)*sech(mu*L)*cosh(mu*(L-x))/(5*E*A)
# At x = L:
dv_dx_L = - P * L**2 / (2 * E * I) + 12 * P * (1 + NU) / (5 * E * A) - (12 * P * (1 + NU) / (5 * E * A)) * (1.0 / np.cosh(mu * L))

# theta = - P * (2L - x)*x / (2*E*I) + 3*P*(1+nu)/(5*E*A) - 3*P*(1+nu)*sech(mu*L)*cosh(mu*(L-x))/(5*E*A)
# At x = L:
theta_L = - P * L**2 / (2 * E * I) + 3 * P * (1 + NU) / (5 * E * A) - (3 * P * (1 + NU) / (5 * E * A)) * (1.0 / np.cosh(mu * L))

# Since the moment M is also applied at the end:
# Wait! Do equations 29-31 include the effect of the moment M?
# Let's read the paper description for equations 29-31:
# "According with Ruocco and Reddy (2023), the analytical solution of the problem considering the Reddy beam model is given by:"
# This is for a clamped beam with a concentrated load P at the end. It does not mention M in equations 29-31!
# Wait! Let's check if the analytical solution is only for P.
# In the paper, under section 7.1:
# "The applied load was P = 1030 kN, and M = 1.03 kNm."
# If there is also an applied moment M, the analytical solution would change.
# Let's calculate the analytical solution with M = 0 and M = 1.03 to see what the values are.

print(f"mu = {mu}")
print(f"sech(mu*L) = {1.0/np.cosh(mu*L)}")
print(f"dv_dx_L (P only) = {dv_dx_L}")
print(f"theta_L (P only) = {theta_L}")

# Shear stress at x=L, y=0:
# tau = G * (theta_L - dv_dx_L) * (3*alpha*0**2 - 1) = - G * (theta_L - dv_dx_L)
tau_L_y0 = - G * (theta_L - dv_dx_L)
print(f"tau_L_y0 (P only) = {tau_L_y0}")
