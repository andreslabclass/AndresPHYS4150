import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random 

def brownian_motion(L, steps):
    # Initialize particle position to the center of the grid
    i, j = L // 2, L // 2


# Define directions
directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# Perform steps
i = 0
j = 0
N=10000
new_i=np.zeros(N,int)
new_j=np.zeros(N,int)
for pa in range(N-1):
    # Choose a random direction
    direction = random.choice(directions)
    new_i[pa+1] = new_i[pa] + direction[0]
    new_j[pa+1] = new_j[pa] + direction[1]

# Print the new position
print("pa:", pa+1, "Direction:", (direction[0], direction[1]), "New position:", (new_i, new_j))

np.max(new_i)

fig = plt.figure(figsize=(4,4))
ax = plt.axes(xlim=(-100,100), ylim=(-100, 100))
point = plt.Circle((0, 0), 5.0, fc='b')

def init():
    point.center = (0,0)
    ax.add_patch(point)
    return point,

def animate(i):
    point.center = (new_i[i], new_j[i])
    return point,

anim = animation.FuncAnimation(fig, animate,
init_func=init,frames=360,interval=20,blit=True)
#alternatively save as a gif
writergif = animation.PillowWriter(fps=30)
anim.save('filename.gif',writer=writergif)


fig = plt.figure(figsize=(5,5))
ax = plt.axes(xlim=(-100,100), ylim=(-100, 100))
point = plt.Circle((0, 0), 5.0, fc='b')
point.center = (0,0)
ax.add_patch(point)

