import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib import cm

res = 512

X = np.arange(0.5, 1.0, 1/res)
Y = np.arange(0.5, 1.0, 1/res)
x0, x1 = np.meshgrid(X, Y)
obj = np.log(1.0 / (1 - x0) / (1 - x1))
dx0 = (np.roll(obj[::2, ::2], 1, axis=0) - obj[::2, ::2]) * res
dx1 = (np.roll(obj[::2, ::2], 1, axis=1) - obj[::2, ::2]) * res

obj = np.where(0.4 + 0.6 * x0 ** 2 <= x0 * x1, obj, np.nan)
dx0 = np.where(0.4 + 0.6 * x0[::2, ::2] ** 2 <= x0[::2, ::2] * x1[::2, ::2], dx0, np.nan)
dx1 = np.where(0.4 + 0.6 * x0[::2, ::2] ** 2 <= x0[::2, ::2] * x1[::2, ::2], dx1, np.nan)

im = plt.imshow(
    obj,
    extent=(0.5, 1, 0.5, 1),
    origin='lower',
    cmap='coolwarm',
    interpolation='nearest',)
plt.colorbar(im)

# plot gradients as vector field:
plt.quiver(x0[::2, ::2], x1[::2, ::2], dx0, dx1, scale=1e4)

plt.show()
