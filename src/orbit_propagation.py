"""
Orbit propagation module using Skyfield.

This module computes satellite positions over time
based on TLE data.
"""

from skyfield.api import load


def propagate_orbits(satellites, minutes=60):
    """
    Propagate satellite orbits over a given time window.

    Parameters
    ----------
    satellites : list
        List of EarthSatellite objects
    minutes : int
        Duration of propagation

    Returns
    -------
    dict
        Dictionary of satellite positions (x, y, z in km)
    """

    # Load timescale (Skyfield time system)
    ts = load.timescale()

    # Generate time points (1 per minute)
    times = ts.utc(2024, 1, 1, 0, range(minutes))

    positions = {}

    for sat in satellites:

        # Compute satellite position at each time
        geocentric = sat.at(times)

        # Extract 3D coordinates (km)
        xyz = geocentric.position.km

        positions[sat.name] = xyz

    return positions
