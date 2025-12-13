import numpy as np
from rastrigin import rastrigin
from utils import clamp, BOUNDS

def simulated_annealing(
    start,
    rng,
    initial_temp=5.0,
    final_temp=1e-3,
    alpha=0.999,          # cooling
    step_max=0.8,         # max step at high T
    step_min=0.02,        # min step at low T
    max_iters=6000
):
    """
    Bounded + temperature-adaptive Simulated Annealing (2D)
    - Keeps x in [-5.12, 5.12]
    - Step size shrinks with temperature
    """
    x = clamp(start)
    fx = rastrigin(x)

    best = x.copy()
    best_fx = fx

    T = initial_temp
    path = [x.copy()]

    # map temperature to step size smoothly
    def step_size(Tcur):
        # linear in log-space-ish behavior (simple, stable)
        ratio = (Tcur - final_temp) / max(initial_temp - final_temp, 1e-12)
        ratio = np.clip(ratio, 0.0, 1.0)
        return step_min + (step_max - step_min) * ratio

    for _ in range(max_iters):
        if T < final_temp:
            break

        s = step_size(T)
        # propose
        cand = x + rng.uniform(-s, s, size=2)
        cand = clamp(cand)
        fc = rastrigin(cand)

        dE = fc - fx

        # accept
        if dE <= 0 or rng.random() < np.exp(-dE / T):
            x, fx = cand, fc
            path.append(x.copy())

            if fx < best_fx:
                best, best_fx = x.copy(), fx

        T *= alpha

    return np.array(path), best, float(best_fx)


def sa_with_restarts(rng, n_restarts=25, **sa_kwargs):
    """
    Runs SA from multiple random starts; returns the best solution/path/value.
    This is how you make SA reliably good in practice.
    """
    best_val = float("inf")
    best_point = None
    best_path = None

    for _ in range(n_restarts):
        start = rng.uniform(BOUNDS[0], BOUNDS[1], size=2)
        path, point, val = simulated_annealing(start=start, rng=rng, **sa_kwargs)
        if val < best_val:
            best_val = val
            best_point = point
            best_path = path

    return best_path, best_point, float(best_val)
