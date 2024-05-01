import numpy as np
import matplotlib.pyplot as plt
x = 1

def func(x): 
    return x * (x-1)





def derive(func, x, h):
    return (func(x + h) - func(x)) / h




arr = []
h = [1e-4,1e-6,1e-8,1e-10,1e-12, 1e-14]
#h = [10**-i for i in range(2, 15, 2)]
for i in range(0, len(h)):
    
    arr.append(derive(func, x, h[i]))


print(arr)
plt.scatter(h, arr)
plt.xscale("log")
#plt.yscale("log")
plt.xlabel('h')
plt.ylabel('Derivative')
plt.xlim(1e-16, 1e-0)

 
    



