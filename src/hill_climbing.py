import numpy as np
import matplotlib.pyplot as plt
from rastrigin import rastrigin, plot_rastrigin_2d


def hill_climbing(start, step_size=0.1, max_iters=500):
    current = start.copy()
    current_value = rastrigin(current)
    path = [current.copy()]

    for _ in range(max_iters):
        neighbors = [
            current + np.array([ step_size, 0]),
            current + np.array([-step_size, 0]),
            current + np.array([0,  step_size]),
            current + np.array([0, -step_size])
        ]

        values = [rastrigin(n) for n in neighbors]
        best_idx = np.argmin(values)

        if values[best_idx] >= current_value:
            break

        current = neighbors[best_idx]
        current_value = values[best_idx]
        path.append(current.copy())

    return np.array(path), current_value


def random_restart_hill_climbing(n_restarts=30):
    best_value = float("inf")
    best_path = None

    for _ in range(n_restarts):
        start = np.random.uniform(-5, 5, size=2)
        path, value = hill_climbing(start)

        if value < best_value:
            best_value = value
            best_path = path

    return best_path, best_value

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    from rastrigin import plot_rastrigin_2d
    from utils import run_multiple_times
    from simulated_annealing import sa_with_restarts
    
    def rr_run(rng):
        _, val = random_restart_hill_climbing(n_restarts=30)
        return float(val)

    rr_stats = run_multiple_times(rr_run, n_runs=20, seed=123)
    print("Random Restart stats:", rr_stats)

    def sa_run(rng):
        _, _, val = sa_with_restarts(
            rng,
            n_restarts=25,
            initial_temp=5.0,
            final_temp=1e-3,
            alpha=0.999,
            step_max=0.8,
            step_min=0.02,
            max_iters=6000
        )
        return float(val)

    sa_stats = run_multiple_times(sa_run, n_runs=20, seed=123)
    print("Simulated Annealing stats:", sa_stats)

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

    print("SA best value (visualized):", sa_best_val)

    with open("../results/sa_best.txt", "w") as f:
        f.write(str(sa_best_val))

    plot_rastrigin_2d()
    plt.plot(sa_path[:, 0], sa_path[:, 1], linewidth=1.5, label="SA Path")
    plt.scatter(sa_best_point[0], sa_best_point[1], s=100, label="SA Best")
    plt.legend()
    plt.savefig("../visualizations/simulated_annealing_path.png", dpi=300)
    plt.show()

