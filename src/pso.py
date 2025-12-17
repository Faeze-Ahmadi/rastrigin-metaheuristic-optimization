import numpy as np
from rastrigin import rastrigin
from utils import clamp, BOUNDS


def particle_swarm_optimization(
    rng,
    n_particles=20,
    n_iters=2000,
    w=0.7,      
    c1=1.5,      
    c2=1.5       
):
    """
    Particle Swarm Optimization (2D, bounded)
    Returns:
        - positions history (list of arrays [n_particles, 2])
        - global best position
        - global best value
    """

    positions = rng.uniform(BOUNDS[0], BOUNDS[1], size=(n_particles, 2))
    velocities = rng.uniform(-1.0, 1.0, size=(n_particles, 2))

    pbest_pos = positions.copy()
    pbest_val = np.array([rastrigin(p) for p in positions])

    gbest_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = float(pbest_val[gbest_idx])

    history = []

    for _ in range(n_iters):
        history.append(positions.copy())

        r1 = rng.random(size=(n_particles, 2))
        r2 = rng.random(size=(n_particles, 2))

        velocities = (
            w * velocities
            + c1 * r1 * (pbest_pos - positions)
            + c2 * r2 * (gbest_pos - positions)
        )

        positions = positions + velocities
        positions = clamp(positions)

        values = np.array([rastrigin(p) for p in positions])

        # Update personal best
        improved = values < pbest_val
        pbest_pos[improved] = positions[improved]
        pbest_val[improved] = values[improved]

        # Update global best
        idx = np.argmin(pbest_val)
        if pbest_val[idx] < gbest_val:
            gbest_val = float(pbest_val[idx])
            gbest_pos = pbest_pos[idx].copy()

    return history, gbest_pos, gbest_val
