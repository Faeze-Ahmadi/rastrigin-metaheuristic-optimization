import numpy as np
import matplotlib.pyplot as plt

def rastrigin(x):
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def plot_rastrigin_2d():
    x = np.linspace(-5.12, 5.12, 400)
    y = np.linspace(-5.12, 5.12, 400)
    X, Y = np.meshgrid(x, y)

    Z = np.array([
        rastrigin(np.array([X[i, j], Y[i, j]]))
        for i in range(X.shape[0])
        for j in range(X.shape[1])
    ]).reshape(X.shape)

    plt.figure(figsize=(6, 5))
    contour = plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(contour)
    plt.scatter(0, 0, color="red", marker="x", label="Global Minimum")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
