import numpy as np


def compute_conjunctions(positions, threshold_km=2, generate_fragments=False):
    """
    Detect close approaches between objects.

    If generate_fragments=True, simulate Kessler effect by creating new debris.
    """

    sats = list(positions.keys())

    events = []
    fragments = {}

    for i in range(len(sats)):
        for j in range(i+1, len(sats)):

            A = positions[sats[i]]
            B = positions[sats[j]]

            # align time dimension
            T = min(A.shape[1], B.shape[1])

            A_cut = A[:, :T]
            B_cut = B[:, :T]

            dist = np.linalg.norm(A_cut - B_cut, axis=0)

            min_dist = np.min(dist)
            time_index = np.argmin(dist)

            if min_dist < threshold_km:

                events.append({
                    "sat1": sats[i],
                    "sat2": sats[j],
                    "min_distance": float(min_dist),
                    "time_index": int(time_index)
                })

                # 💥 Kessler effect
                if generate_fragments:

                    for k in range(3):
                        frag_name = f"FRAG_{i}_{j}_{k}"

                        # explosion = small perturbation
                        noise = np.random.normal(0, 50, size=A.shape)

                        fragments[frag_name] = A + noise

    return events, fragments
