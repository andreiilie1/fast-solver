from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


A = np.matrix([[2, -1], [3,5]])


def f(x, y):
    return (2 * x ** 2 +  5 * y ** 2 + 2 * x*y)

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)
fig = plt.figure()

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('surface');

plt.show()