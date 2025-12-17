import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from rastrigin import plot_rastrigin_2d
from simulated_annealing import sa_with_restarts


def make_sa_gif(output_path="../visualizations/sa_convergence.gif"):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(123)

    sa_path, sa_best_point, sa_best_val = sa_with_restarts(
        rng,
        n_restarts=25,
        initial_temp=5.0,
        final_temp=1e-3,
        alpha=0.999,
        step_max=0.8,
        step_min=0.02,
        max_iters=6000
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_rastrigin_2d()

    line, = ax.plot([], [], "r-", lw=2, label="SA Path")
    point, = ax.plot([], [], "ro", markersize=5, label="Current Position")
    ax.legend()

    def update(frame):
        line.set_data(sa_path[:frame + 1, 0], sa_path[:frame + 1, 1])
        point.set_data(
            [sa_path[frame, 0]],
            [sa_path[frame, 1]]
        )
        return line, point

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(sa_path),
        interval=30,
        blit=True
    )

    ani.save(output_path, writer="pillow", dpi=120)
    plt.close(fig)

    print(f"[OK] GIF saved to: {output_path}")
    print(f"[OK] Best value found: {sa_best_val}")
    print(f"[OK] Best point: {sa_best_point}")


if __name__ == "__main__":
    make_sa_gif()
