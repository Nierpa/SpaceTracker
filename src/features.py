import numpy as np


def compute_relative_velocity(posA, posB):
    """
    Compute relative velocity between two objects.
    
    Handles different trajectory lengths by aligning them.
    """

    # Compute velocities (simple derivative)
    velA = np.diff(posA, axis=1)
    velB = np.diff(posB, axis=1)

    # Align lengths (IMPORTANT FIX)
    min_len = min(velA.shape[1], velB.shape[1])

    velA = velA[:, :min_len]
    velB = velB[:, :min_len]

    # Relative velocity
    rel_vel = np.linalg.norm(velA - velB, axis=0)

    return np.percentile(rel_vel, 95)
