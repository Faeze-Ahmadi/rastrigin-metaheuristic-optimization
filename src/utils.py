import numpy as np

BOUNDS = (-5.12, 5.12)

def clamp(x, lo=BOUNDS[0], hi=BOUNDS[1]):
    return np.clip(x, lo, hi)

def run_multiple_times(run_fn, n_runs=20, seed=123):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_runs):
        vals.append(run_fn(rng))
    vals = np.array(vals, dtype=float)
    return {
        "best": float(vals.min()),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=0)),
    }
