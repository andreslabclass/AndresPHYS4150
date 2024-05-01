import numpy as np
import matplotlib.pyplot as plt

# Define the parameter range
theta = np.linspace(0, 2*np.pi, 1000)

# Parametric equations
x = 2 * np.cos(theta) + np.cos(2 * theta)
y = 2 * np.sin(theta) - np.sin(2 * theta)

# Plot the deltoid curve
plt.plot(x, y)
plt.title('Deltoid Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.grid(True)
plt.show()


# Define the parameter range
theta = np.linspace(0, 10*np.pi, 1000)

# Define the function r = f(theta)
r = theta**2

# Convert polar coordinates to Cartesian coordinates
x = r * np.cos(theta)
y = r * np.sin(theta)

# Plot the Galilean spiral
plt.plot(x, y)
plt.title('Galilean Spiral')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.grid(True)
plt.show()


# Define the parameter range
theta = np.linspace(0, 24*np.pi, 1000)  # Adjust the range as needed for a good plot

# Convert polar coordinates to Cartesian coordinates
x = r * np.cos(theta)
y = r * np.sin(theta)

# Plot Fey's function
plt.figure(figsize=(8, 8))  # Adjust figure size as needed
plt.plot(x, y)
plt.title("Fey's Function")
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.grid(True)
plt.show()



