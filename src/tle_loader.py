"""
Ultra-robust TLE loader.

Handles:
- missing lines
- malformed entries
- empty lines
"""

from skyfield.api import EarthSatellite


def load_satellites(tle_file, max_sat=100):

    sats = []

    # Read and clean lines
    with open(tle_file) as f:
        raw_lines = f.readlines()

    # Remove empty lines
    lines = [l.strip() for l in raw_lines if l.strip()]

    i = 0

    while i < len(lines):

        # Ensure we have at least 3 lines remaining
        if i + 2 >= len(lines):
            break

        name = lines[i]
        line1 = lines[i + 1]
        line2 = lines[i + 2]

        # Validate TLE format
        if line1.startswith("1 ") and line2.startswith("2 "):

            try:
                sat = EarthSatellite(line1, line2, name)
                sats.append(sat)
                i += 3  # move to next satellite
            except Exception:
                i += 1  # skip broken entry

        else:
            i += 1  # skip malformed block

        if len(sats) >= max_sat:
            break

    print(f"Loaded {len(sats)} satellites")

    return sats
    
def load_multiple_tle(files, max_per_file=100):
    """
    Load satellites from multiple TLE files.
    """

    all_sats = []

    for file in files:

        sats = load_satellites(file, max_sat=max_per_file)
        all_sats.extend(sats)

    print(f"Total satellites loaded: {len(all_sats)}")

    return all_sats
