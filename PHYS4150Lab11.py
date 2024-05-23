import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

def dst(y):
    """Discrete sine transform of the array y."""
    N = len(y)
    y2 = np.zeros(2 * N + 2, float)
    y2[1:N + 1] = y
    y2[N + 1:2 * N + 1] = -y[::-1]
    theta = np.fft.rfft(y2)
    return np.imag(theta[1:N + 1])

# Define parameters
N = 1000
L = 1.0  # Length of the box
x = np.linspace(0, L, N)  # Grid points

# Define the initial wave function ψ(x, 0) at each grid point
psi_real = np.sin(np.pi * x / L)  # Example real part
psi_imag = np.zeros_like(x)  # Example imaginary part

# Perform discrete sine transforms
alpha = dst(psi_real)[1:]  # Ignore the first element
eta = dst(psi_imag)[1:]  # Ignore the first element

# Print the coefficients
for k in range(len(alpha)):
    print(f"b_{k+1} = {alpha[k]} + {eta[k]}i")


# Constants
N = 1000
L = 1.0  # Length of the box
hbar = 1.0545718e-34  # Planck's constant (Joule second)
M = 9.10938356e-31  # Mass of electron (kg)
t = 1e-16  # Time in seconds

# Grid points
x = np.linspace(0, L, N)

# Initial wave function ψ(x, 0) at each grid point
psi_real = np.sin(np.pi * x / L)  # Example real part
psi_imag = np.zeros_like(x)  # Example imaginary part

# Perform discrete sine transforms
alpha = dst(psi_real)[1:]  # Ignore the first element
eta = dst(psi_imag)[1:]  # Ignore the first element

# Calculate the real part of the wavefunction at time t
real_part_wavefunction = np.zeros_like(x)
for k in range(1, N):
    coefficient = (alpha[k-1] * np.cos((np.pi**2 * hbar * k**2 / (2 * M * L**2)) * t) - 
                   eta[k-1] * np.sin((np.pi**2 * hbar * k**2 / (2 * M * L**2)) * t))
    real_part_wavefunction += coefficient * np.sin(np.pi * k)

# Plot the wavefunction at time t
plt.plot(x, real_part_wavefunction)
plt.xlabel('x')
plt.ylabel('Re ψ(x, t)')
plt.title(f'Real part of the wavefunction at t = {t} s')
plt.show()


# Constants
N = 1000
L = 1.0  # Length of the box
hbar = 1.0545718e-34  # Planck's constant (Joule second)
M = 9.10938356e-31  # Mass of electron (kg)
time_interval = 1e-18  # Time interval between frames in seconds

# Grid points
x = np.linspace(0, L, N)

# Initial wave function ψ(x, 0) at each grid point
psi_real = np.sin(np.pi * x / L)  # Example real part
psi_imag = np.zeros_like(x)  # Example imaginary part

# Perform discrete sine transforms
alpha = dst(psi_real)[1:]  # Ignore the first element
eta = dst(psi_imag)[1:]  # Ignore the first element

# Prepare the plot
fig, ax = plt.subplots()
line, = ax.plot(x, np.zeros_like(x))
ax.set_xlim(0, L)
ax.set_ylim(-1, 1)
ax.set_xlabel('x')
ax.set_ylabel('Re ψ(x, t)')
ax.set_title('Real part of the wavefunction')

def update_wavefunction(t):
    """Calculate the real part of the wavefunction at time t."""
    real_part_wavefunction = np.zeros_like(x)
    for k in range(1, N):
        coefficient = (alpha[k-1] * np.cos((np.pi**2 * hbar * k**2 / (2 * M * L**2)) * t) - 
                       eta[k-1] * np.sin((np.pi**2 * hbar * k**2 / (2 * M * L**2)) * t))
        real_part_wavefunction += coefficient * np.sin(np.pi * k * x / L)
    return real_part_wavefunction

def animate(frame):
    """Update the plot for animation."""
    t = frame * time_interval
    line.set_ydata(update_wavefunction(t))
    ax.set_title(f'Real part of the wavefunction at t = {t:.2e} s')
    return line,

# Create the animation
num_frames = 100  # Adjust as needed
ani = animation.FuncAnimation(fig, animate, frames=num_frames, blit=True, interval=50)

# Display the animation
HTML(ani.to_jshtml())


