import numpy as np
import matplotlib.pyplot as plt
from rastrigin import plot_rastrigin_2d, rastrigin


def hill_climbing(
    start,
    step_size=0.1,
    max_iters=500
):
    """
    Simple Hill Climbing for Rastrigin
    """
    current = start.copy()
    current_value = rastrigin(current)

    path = [current.copy()]

    for _ in range(max_iters):
        neighbors = [
            current + np.array([step_size, 0]),
            current + np.array([-step_size, 0]),
            current + np.array([0,  step_size]),
            current + np.array([0, -step_size])
        ]

        values = [rastrigin(n) for n in neighbors]

        best_idx = np.argmin(values)
        best_neighbor = neighbors[best_idx]
        best_value = values[best_idx]

        # if no improvement, stop
        if best_value >= current_value:
            break

        current = best_neighbor
        current_value = best_value
        path.append(current.copy())

    return np.array(path), current_value

def random_restart_hill_climbing(
    n_restarts=20,
    step_size=0.1,
    max_iters=500
):
    """
    Random Restart Hill Climbing
    """
    best_value = float("inf")
    best_path = None
    best_start = None

    for i in range(n_restarts):
        start = np.random.uniform(-5, 5, size=2)
        path, value = hill_climbing(
            start,
            step_size=step_size,
            max_iters=max_iters
        )

        if value < best_value:
            best_value = value
            best_path = path
            best_start = start

        print(f"Restart {i+1}: final value = {value:.4f}")

    return best_start, best_path, best_value

def plot_path(path):
    x = path[:, 0]
    y = path[:, 1]

    plt.plot(x, y, color="red", marker="o", linewidth=2, label="Hill Climbing Path")
    plt.scatter(x[0], y[0], color="white", edgecolors="black", s=100, label="Start")
    plt.scatter(x[-1], y[-1], color="red", s=100, label="Stop")
    plt.legend()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from rastrigin import plot_rastrigin_2d

    best_start, best_path, best_value = random_restart_hill_climbing(
        n_restarts=30
    )

    print("\nBest result:")
    print("Start:", best_start)
    print("End:", best_path[-1])
    print("Final value:", best_value)

    plot_rastrigin_2d()
    plot_path(best_path)
    plt.show()



