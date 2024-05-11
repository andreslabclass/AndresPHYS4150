import numpy as np
import matplotlib.pyplot as plt


# Constants
g = 9.81  # Acceleration due to gravity (m/s^2)
rho = 1.2  # Density of air (kg/m^3)
C = 0.5  # Coefficient of drag
R = 0.1  # Radius of the cannonball (m)
m = 0.5  # Mass of the cannonball (kg)


# Function to compute the derivatives of x and y
def derivatives(state, t):
    x, y, dx, dy = state
    v = np.sqrt(dx**2 + dy**2)
    dvx_dt = -(np.pi * R**2 * rho * C / (2 * m)) * dx * v
    dvy_dt = -g - (np.pi * R**2 * rho * C / (2 * m)) * dy * v
    print(dx,dy)
    return np.array([dx, dy, dvx_dt, dvy_dt])


# Initial conditions
x0 = 0.0  # Initial x position (m)
y0 = 0.0  # Initial y position (m)
v0 = 100.0  # Initial velocity (m/s)
theta = np.pi / 4  # Launch angle (radians)


# Initial velocities
vx0 = v0 * np.cos(theta)
vy0 = v0 * np.sin(theta)

# Initial state [x, y, dx/dt, dy/dt]
initial_state = [x0, y0, vx0, vy0]
print(initial_state)


# Time array
N=1000
tpoints = np.linspace(0, 10, N)
xpoints = []
x = np.array([x0, y0, vx0, vy0])
h = 10/N
f = derivatives
print(tpoints[0:2])


for t in tpoints: 
    xpoints.append(x)
    k1 = h*f(x,t)
    k2 = h*f(x+0.5*k1,t+0.5*h)
    k3 = h*f(x+0.5*k2,t+0.5*h)
    k4 = h*f(x+k3,t+h)
    x  = x + (k1+2*k2+2*k3+k4)/6

print((x,0.1))
print(f(x,0.001))

states=np.array(xpoints)
states.shape

# Solve differential equations
# from scipy.integrate import odeint
# states = odeint(derivatives, initial_state, t)

# Extract positions
x = states[:, 0]
y = states[:, 1]

# Plot the trajectory
plt.figure(figsize=(8, 6))
plt.plot(x,y)
plt.title('Cannonball Trajectory with Air Resistance')
plt.xlabel('Horizontal Distance (m)')
plt.ylabel('Vertical Distance (m)')
plt.grid(True)
plt.show()




import numpy as np
import matplotlib.pyplot as plt


# Constants
g = 9.81  # acceleration due to gravity (m/s^2)
l = 0.1  # length of the pendulum arm (m)
theta_initial = np.radians(179)  # initial angle (radians)
omega_initial = 0.0  # initial angular velocity (rad/s)
t_max = 10.0  # maximum time (s)
dt = 0.01  # time step size (s)



# Function defining the system of differential equations
def pendulum_equations(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -(g / l) * np.sin(theta)
    return [dtheta_dt, domega_dt]


# Fourth-order Runge-Kutta method
def runge_kutta4(f, y0, t0, t_max, dt):
    t = t0
    y = y0
    times = [t0]
    solutions = [y0]
    while t < t_max:
        k1 = dt * np.array(f(t, y))
        k2 = dt * np.array(f(t + 0.5*dt, y + 0.5*k1))
        k3 = dt * np.array(f(t + 0.5*dt, y + 0.5*k2))
        k4 = dt * np.array(f(t + dt, y + k3))
        y = y + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        t += dt
        times.append(t)
        solutions.append(y)
    return np.array(times), np.array(solutions)

# Initial conditions
y0 = [theta_initial, omega_initial]


# Solve the system
times, solutions = runge_kutta4(pendulum_equations, y0, 0, t_max, dt)


# Extract theta values
theta_values = solutions[:, 0]

# Convert time to periods
period = 2 * np.pi * np.sqrt(l / g)
times /= period

# Plot
plt.plot(times, theta_values)
plt.title('Pendulum Motion')
plt.xlabel('Time (periods)')
plt.ylabel('Angle θ (radians)')
plt.grid(True)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Constants
g = 9.81  # acceleration due to gravity (m/s^2)
l = 0.1  # length of the pendulum arm (m)
theta_initial = np.radians(179)  # initial angle (radians)
omega_initial = 0.0  # initial angular velocity (rad/s)
t_max = 10.0  # maximum time (s)
dt = 0.01  # time step size (s)
frames_per_second = 30  # frame rate of the animation


# Function defining the system of differential equations
def pendulum_equations(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -(g / l) * np.sin(theta)
    return [dtheta_dt, domega_dt]


# Fourth-order Runge-Kutta method
def runge_kutta4(f, y0, t0, t_max, dt):
    t = t0
    y = y0
    times = [t0]
    solutions = [y0]
    while t < t_max:
        k1 = dt * np.array(f(t, y))
        k2 = dt * np.array(f(t + 0.5*dt, y + 0.5*k1))
        k3 = dt * np.array(f(t + 0.5*dt, y + 0.5*k2))
        k4 = dt * np.array(f(t + dt, y + k3))
        y = y + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        t += dt
        times.append(t)
        solutions.append(y)
    return np.array(times), np.array(solutions)

# Initial conditions
y0 = [theta_initial, omega_initial]

# Solve the system
times, solutions = runge_kutta4(pendulum_equations, y0, 0, t_max, dt)

# Extract theta values
theta_values = solutions[:, 0]

# Convert time to periods
period = 2 * np.pi * np.sqrt(l / g)
times /= period

# Set up the figure and axes
fig, ax = plt.subplots()
ax.set_xlim(-l, l)
ax.set_ylim(-l, 0.1)
ax.set_aspect('equal')
ax.grid(True)

# Create pendulum arm
arm, = ax.plot([], [], 'b-', lw=2)

# Create pendulum bob
line, = ax.plot([], [], 'ro', markersize=10)


# Create animation
ani = animation.FuncAnimation(fig, update_pendulum, frames=len(times), fargs=(solutions, arm, line),
      interval=1000/frames_per_second, blit=True)

plt.title('Pendulum Motion')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()



