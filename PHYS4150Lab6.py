import numpy as np
import matplotlib.pyplot as plt


E = np.linspace(0.01,19.99,100) 
V = 20
w = 1e-9
m = 9.1e-31
h = 6.26e-34
y1 = np.tan(np.sqrt((w**2*m*E*1.6e-19)/(2*h**2)))
y2 = np.sqrt((V-E)/E)
y3 = - np.sqrt(E/(V-E))

plt.plot(E,(y1))
plt.plot(E,(y2))
plt.plot(E,(y3))
plt.ylim(-3,3)


