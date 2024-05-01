import numpy as np
import matplotlib.pyplot as plt 

def pot(r):
    biga = q/4*pi(eps0)*r
    return biga



# constants
k = 8.99*10**9 # Coulumb's constant in N m**2/c**2
q1 = 1.0 # charge 1 in C 
q2 = -1.0 # charge 2 in C
d = 0.1  # Distance between charges in meters


# Define the square plane
side_length = 1.0  # Side length of the square plane in meters
num_points = 100   # Number of points on each side of the square



# Create a grid of points on the square plane
x = np.linspace(-side_length/2, side_length/2, num_points)
y = np.linspace(-side_length/2, side_length/2, num_points)
X, Y = np.meshgrid(x, y)



# Calculate the electric potential at each point
r1 = np.sqrt((X - d/2)**2 + Y**2)
r2 = np.sqrt((X + d/2)**2 + Y**2)
V1 = k * q1 / r1
V2 = k * q2 / r2
V_total = V1 + V2


# Create a density plot of the electric potential
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, V_total, cmap='viridis')
plt.colorbar(contour, label='Electric Potential (V)')
plt.title('Electric Potential due to ±1 C Charges')
plt.xlabel('X-axis (m)')
plt.ylabel('Y-axis (m)')
plt.show()









