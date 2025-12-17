import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

from rastrigin import rastrigin
from simulated_annealing import sa_with_restarts


def plot_rastrigin_light(ax):
    x = np.linspace(-5.12, 5.12, 150)
    y = np.linspace(-5.12, 5.12, 150)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = rastrigin(np.array([X[i, j], Y[i, j]]))

    ax.contourf(X, Y, Z, levels=30, cmap="viridis")
    ax.plot(0, 0, "rx", markersize=8, label="Global Minimum")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()


def make_sa_gif():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    VIS_DIR = PROJECT_ROOT / "visualizations"
    VIS_DIR.mkdir(exist_ok=True)

    output_path = VIS_DIR / "sa_convergence.gif"

    rng = np.random.default_rng(123)

    sa_path, best_point, best_val = sa_with_restarts(
        rng,
        n_restarts=10,     
        max_iters=3000     
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_rastrigin_light(ax)

    line, = ax.plot([], [], "r-", lw=2, label="SA Path")
    point, = ax.plot([], [], "ro", markersize=4, label="Current Position")

    tail = 200 
    step = 5     

    def update(frame):
        idx = frame * step
        idx = min(idx, len(sa_path) - 1)

        start = max(0, idx - tail)

        line.set_data(
            sa_path[start:idx, 0],
            sa_path[start:idx, 1]
        )

        point.set_data(
            [sa_path[idx, 0]],
            [sa_path[idx, 1]]
        )

        return line, point

    frames = len(sa_path) // step

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=60,
        blit=False
    )

    writer = animation.PillowWriter(fps=15)
    ani.save(str(output_path), writer=writer)

    plt.close(fig)

    print("[OK] GIF saved to:", output_path)
    print("[OK] Best value:", best_val)
    print("[OK] Best point:", best_point)


if __name__ == "__main__":
    make_sa_gif()
