import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 1000  # Number of steps
L = 10    # Range
dx = L / N
x = np.linspace(-L/2, L/2, N)  # Centering the x range around zero


# Potential function (harmonic oscillator potential)
def V(x):
    return 0.5 * x**2

# Initial wave function (Gaussian wave packet)
def initial_wave_packet(x, x0, sigma, k0):
    return np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)

# Initial parameters for the wave packet
x0 = 0   # Center the initial position
sigma = 1.0  # Width of the wave packet
k0 = 5.0  # Initial momentum

# Initialize wave function as complex
psi = initial_wave_packet(x, x0, sigma, k0).astype(np.complex128)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)  # Normalize the wave function

# Plot the wave function and potential at t = 0
fig, ax = plt.subplots()
ax.plot(x, np.abs(psi)**2, label='Probability Density at t = 0')
ax.plot(x, V(x), 'r--', label='Potential Energy')
ax.set_title("Quantum Harmonic Oscillator at t = 0")
ax.set_xlabel("Position")
ax.set_ylabel("Probability / Energy")
ax.set_ylim(0, 1)
ax.legend()
plt.show()



# Parameters
N = 1000  # Number of steps
L = 10    # Range
dx = L / N
x = np.linspace(-L/2, L/2, N)  # Centering the x range around zero

# Potential function (harmonic oscillator potential)
def V(x):
    return 0.5 * x**2

# Hamiltonian operator (kinetic and potential terms)
def H_operator(psi):
    # Second derivative (finite difference method)
    d2psi_dx2 = (np.roll(psi, -1) - 2 * psi + np.roll(psi, 1)) / dx**2
    return -0.5 * d2psi_dx2 + V(x) * psi

# Initial wave function (Gaussian wave packet)
def initial_wave_packet(x, x0, sigma):
    return np.exp(-(x - x0)**2 / (2 * sigma**2))

# Initial parameters for the wave packet
x0 = 0   # Center the initial position
sigma = 1.0  # Width of the wave packet

# Initialize wave function as complex
psi = initial_wave_packet(x, x0, sigma).astype(np.complex128)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)  # Normalize the wave function

# Time evolution parameters
dt = 0.001  # Time step
t_max = 1  # Maximum time
t_values = np.arange(0, t_max, dt)

# Evolve the wave function to t = 1
for t in t_values:
    psi += -1j * dt * H_operator(psi)  # Imaginary unit for time evolution in quantum mechanics
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)  # Normalize the wave function

# Plot the wave function and potential at t = 1
fig, ax = plt.subplots()
ax.plot(x, np.abs(psi)**2, label='Probability Density at t = 1')
ax.plot(x, V(x), 'r--', label='Potential Energy')
ax.set_title("Quantum Harmonic Oscillator at t = 1")
ax.set_xlabel("Position")
ax.set_ylabel("Probability / Energy")
ax.set_ylim(0, 1)
ax.legend()
plt.show()



# Parameters
N = 1000  # Number of steps
L = 10    # Range
dx = L / N
x = np.linspace(-L/2, L/2, N)  # Centering the x range around zero

# Potential function (harmonic oscillator potential)
def V(x):
    return 0.5 * x**2

# Hamiltonian operator (kinetic and potential terms)
def H_operator(psi):
    # Second derivative (finite difference method)
    d2psi_dx2 = (np.roll(psi, -1) - 2 * psi + np.roll(psi, 1)) / dx**2
    return -0.5 * d2psi_dx2 + V(x) * psi

# Initial wave function (Gaussian wave packet)
def initial_wave_packet(x, x0, sigma):
    return np.exp(-(x - x0)**2 / (2 * sigma**2))

# Initial parameters for the wave packet
x0 = 0   # Center the initial position
sigma = 1.0  # Width of the wave packet

# Initialize wave function as complex
psi = initial_wave_packet(x, x0, sigma).astype(np.complex128)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)  # Normalize the wave function

# Time evolution parameters
dt = 0.001  # Time step
t_max = 2  # Maximum time
t_values = np.arange(0, t_max, dt)

# Evolve the wave function to t = 2
for t in t_values:
    psi += -1j * dt * H_operator(psi)  # Imaginary unit for time evolution in quantum mechanics
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)  # Normalize the wave function

# Plot the wave function and potential at t = 2
fig, ax = plt.subplots()
ax.plot(x, np.abs(psi)**2, label='Probability Density at t = 2')
ax.plot(x, V(x), 'r--', label='Potential Energy')
ax.set_title("Quantum Harmonic Oscillator at t = 2")
ax.set_xlabel("Position")
ax.set_ylabel("Probability / Energy")
ax.set_ylim(-1, 1)
ax.legend()
plt.show()


# Parameters
N = 1000  # Number of steps
L = 10    # Range
dx = L / N
x = np.linspace(-L/2, L/2, N)  # Centering the x range around zero

# Potential function (harmonic oscillator potential)
def V(x):
    return 0.5 * x**2

# Hamiltonian operator (kinetic and potential terms)
def H_operator(psi):
    # Second derivative (finite difference method)
    d2psi_dx2 = (np.roll(psi, -1) - 2 * psi + np.roll(psi, 1)) / dx**2
    return -0.5 * d2psi_dx2 + V(x) * psi

# Initial wave function (Gaussian wave packet)
def initial_wave_packet(x, x0, sigma):
    return np.exp(-(x - x0)**2 / (2 * sigma**2))

# Initial parameters for the wave packet
x0 = 0   # Center the initial position
sigma = 1.0  # Width of the wave packet

# Initialize wave function as complex
psi = initial_wave_packet(x, x0, sigma).astype(np.complex128)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)  # Normalize the wave function

# Time evolution parameters
dt = 0.001  # Time step
t_max = 3  # Maximum time
t_values = np.arange(0, t_max, dt)

# Evolve the wave function to t = 3
for t in t_values:
    psi += -1j * dt * H_operator(psi)  # Imaginary unit for time evolution in quantum mechanics
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)  # Normalize the wave function

# Plot the wave function and potential at t = 3
fig, ax = plt.subplots()
ax.plot(x, np.abs(psi)**2, label='Probability Density at t = 3')
ax.plot(x, V(x), 'r--', label='Potential Energy')
ax.set_title("Quantum Harmonic Oscillator at t = 3")
ax.set_xlabel("Position")
ax.set_ylabel("Probability / Energy")
ax.set_ylim(2, 1)
ax.legend()
plt.show()


# Parameters
N = 1000  # Number of steps
L = 10    # Range
dx = L / N
x = np.linspace(-L/2, L/2, N)  # Centering the x range around zero

# Potential function (harmonic oscillator potential)
def V(x):
    return 0.5 * x**2

# Hamiltonian operator (kinetic and potential terms)
def H_operator(psi):
    # Second derivative (finite difference method)
    d2psi_dx2 = (np.roll(psi, -1) - 2 * psi + np.roll(psi, 1)) / dx**2
    return -0.5 * d2psi_dx2 + V(x) * psi

# Initial wave function (Gaussian wave packet)
def initial_wave_packet(x, x0, sigma, k0):
    return np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)

# Initial parameters for the wave packet
x0 = 0   # Center the initial position
sigma = 1.0  # Width of the wave packet
k0 = 5.0  # Initial momentum

# Initialize wave function as complex
psi = initial_wave_packet(x, x0, sigma, k0).astype(np.complex128)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)  # Normalize the wave function

# Time evolution parameters
dt = 0.001  # Time step
t_max = 4  # Maximum time
t_values = np.arange(0, t_max, dt)

# Evolve the wave function to t = 4
for t in t_values:
    psi += -1j * dt * H_operator(psi)  # Imaginary unit for time evolution in quantum mechanics
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)  # Normal

# Plot the wave function and potential at t = 4
fig, ax = plt.subplots()
ax.plot(x, np.abs(psi)**2, label='Probability Density at t = 4')
ax.plot(x, V(x), 'r--', label='Potential Energy')
ax.set_title("Quantum Harmonic Oscillator at t = 4")
ax.set_xlabel("Position")
ax.set_ylabel("Probability / Energy")
ax.set_ylim(0, 1)
ax.legend()
plt.show()


# Parameters
N = 1000  # Number of steps
L = 10    # Range
dx = L / N
x = np.linspace(-L/2, L/2, N)  # Centering the x range around zero

# Potential function (harmonic oscillator potential)
def V(x):
    return 0.5 * x**2

# Hamiltonian operator (kinetic and potential terms)
def H_operator(psi):
    # Second derivative (finite difference method)
    d2psi_dx2 = (np.roll(psi, -1) - 2 * psi + np.roll(psi, 1)) / dx**2
    return -0.5 * d2psi_dx2 + V(x) * psi

# Initial wave function (Gaussian wave packet)
def initial_wave_packet(x, x0, sigma, k0):
    return np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)

# Initial parameters for the wave packet
x0 = 0   # Center the initial position
sigma = 1.0  # Width of the wave packet
k0 = 5.0  # Initial momentum

# Initialize wave function as complex
psi = initial_wave_packet(x, x0, sigma, k0).astype(np.complex128)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)  # Normalize the wave function

# Time evolution parameters
dt = 0.001  # Time step
t_max = 5  # Maximum time
t_values = np.arange(0, t_max + dt, dt)

# List of specific time points to plot
time_points = [1, 2, 4, 5]

# Evolve the wave function and plot at specific time points
for t in t_values:
    psi += -1j * dt * H_operator(psi)  # Imaginary unit for time evolution in quantum mechanics
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)  # Normalize the wave function

if np.isclose(t, time_points).any():
 fig, ax = plt.subplots()
ax.plot(x, np.abs(psi)**2, label=f'Probability Density at t = 5')
ax.plot(x, V(x), 'r--', label='Potential Energy')
ax.set_title(f"Quantum Harmonic Oscillator at t = 5")
ax.set_xlabel("Position")
ax.set_ylabel("Probability / Energy")
ax.set_ylim(-3, 1)
ax.legend()
plt.show()



