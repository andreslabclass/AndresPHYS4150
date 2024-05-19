import numpy as np
from scipy.integrate import quad

# Part (a)
# Define the weighting function w(x) = x^(-1/2)
def w(x):
    return x**(-1/2)

# Normalize w(x) over the interval [0, 1]
integral_w, _ = quad(w, 0, 1)
print(f"Integral of w(x) over [0, 1]: {integral_w}")

# Define the probability distribution p(x)
def p(x):
    return w(x) / integral_w

# Define the CDF P(x) of p(x)
def P(x):
    return np.sqrt(x)

# Define the inverse CDF P^-1(y)
def P_inv(y):
    return y**2

# Test the inverse CDF transformation
np.random.seed(0)  # For reproducibility
U = np.random.uniform(0, 1, 10)
X = P_inv(U)
print(f"Sampled values from p(x): {X}")


# Part (b)
# Number of random points
N = 1_000_000

# Generate N random points from the distribution p(x) using the inverse transform method
U = np.random.uniform(0, 1, N)
X = U**2

# Evaluate the function at these points
f_X = 1 / (np.exp(X) + 1)

# Compute the integral as the mean value of the function
I = np.mean(f_X)

# Print the result
print(f"Estimated value of the integral: {I:.5f}")

