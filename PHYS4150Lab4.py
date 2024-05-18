import numpy as np
from scipy.integrate import quad

# Define the constants
hbar = 1.054571817e-34  # Planck's constant over 2π in J·s
c = 3.0e8  # Speed of light in m/s
k_B = 1.380649e-23  # Boltzmann's constant in J/K
T = 300  # Temperature in Kelvin (example value)

# Define the integrand
def integrand(x):
    return x**3 / (np.exp(x) - 1)

# Compute the integral from 0 to infinity
integral_value, error = quad(integrand, 0, np.inf)

# Compute the total energy per unit area
W = (k_B**4 * T**4) / (4 * np.pi**2 * c**2 * hbar**3) * integral_value

# Print the integral value
print(f"Integral value: {integral_value:.5e}")

# Print the total energy per unit area
print(f"Total energy per unit area radiated by the black body: {W:.5e} W/m^2")

# Compute the Stefan-Boltzmann constant
sigma = (k_B**4) / (4 * np.pi**2 * c**2 * hbar**3) * integral_value

# Print the computed Stefan-Boltzmann constant
print(f"Computed Stefan-Boltzmann constant: {sigma:.5e} W/m^2/K^4")

# Known value of the Stefan-Boltzmann constant for comparison
sigma_known = 5.670374419e-8  # in W/m^2/K^4
print(f"Known Stefan-Boltzmann constant: {sigma_known:.5e} W/m^2/K^4")

# Compare the values
difference = abs(sigma - sigma_known)
print(f"Difference: {difference:.5e} W/m^2/K^4")


