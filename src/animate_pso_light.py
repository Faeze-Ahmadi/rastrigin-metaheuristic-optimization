import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

from rastrigin import rastrigin
from pso import particle_swarm_optimization

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


def make_pso_gif():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    VIS_DIR = PROJECT_ROOT / "visualizations"
    VIS_DIR.mkdir(exist_ok=True)

    output_path = VIS_DIR / "pso_convergence.gif"

    rng = np.random.default_rng(123)

    history, best_point, best_val = particle_swarm_optimization(
        rng,
        n_particles=20,
        n_iters=1500
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_rastrigin_light(ax)

    scat = ax.scatter([], [], c="red", s=20, label="Particles")
    best_dot, = ax.plot([], [], "bo", markersize=6, label="Global Best")
    ax.legend()

    step = 5 

    def update(frame):
        idx = frame * step
        idx = min(idx, len(history) - 1)

        positions = history[idx]
        scat.set_offsets(positions)
        best_dot.set_data([best_point[0]], [best_point[1]])

        return scat, best_dot

    frames = len(history) // step

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

    print("[OK] PSO GIF saved to:", output_path)
    print("[OK] Best value:", best_val)
    print("[OK] Best point:", best_point)


if __name__ == "__main__":
    make_pso_gif()
