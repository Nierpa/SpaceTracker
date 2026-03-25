import numpy as np


def generate_debris(n_debris=20, spread_km=2000):
    """
    Generate debris coming from outer space toward Earth.

    Returns a dict of fake objects with trajectories.
    """

    debris_positions = {}

    for i in range(n_debris):

        name = f"DEBRIS_{i}"

        # initial position far from Earth
        start = np.random.uniform(-spread_km, spread_km, size=3)

        # direction toward Earth (0,0,0)
        direction = -start / np.linalg.norm(start)

        # simulate trajectory
        traj = []

        for t in range(100):
            pos = start + direction * t * 20  # speed factor
            traj.append(pos)

        traj = np.array(traj).T  # shape (3, T)

        debris_positions[name] = traj

    return debris_positions
